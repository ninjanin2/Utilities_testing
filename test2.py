#!/usr/bin/env python3
"""
CleanUNet Audio Denoising Script
Enhanced script for speech denoising using NVIDIA's CleanUNet DNS-large-high model
Designed for offline use with pretrained model files
"""

import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ===== CONFIGURATION SECTION =====
# Update these paths according to your setup
MODEL_PATH = "./models/DNS-large-high/checkpoint/pretrained.pkl"  # Path to your pretrained.pkl file
INPUT_AUDIO_PATH = "./input/noisy_audio.wav"  # Input noisy audio file
OUTPUT_AUDIO_PATH = "./output/cleaned_audio.wav"  # Output cleaned audio file
SAMPLE_RATE = 16000  # CleanUNet works with 16kHz audio
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== CLEANUNET NETWORK ARCHITECTURE =====
class LinearNorm(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)

class ConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation,
                              bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head
        
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, n_head = self.d_k, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        
        residual = q
        
        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_k)
        
        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.
        
        q = self.attention(q, k, v, mask=mask)
        
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual
        
        q = self.layer_norm(q)
        
        return q

    def attention(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        p_attn = torch.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        
        return torch.matmul(p_attn, value)

class CleanUNet(nn.Module):
    def __init__(self, config):
        super(CleanUNet, self).__init__()
        
        # Default configuration for DNS-large-high
        self.config = {
            'n_layers': 8,
            'channels_input': 1,
            'channels_output': 1,
            'channels_H': 64,
            'max_channels': 768,
            'kernel_size': 3,
            'stride': 2,
            'tsfm_n_layers': 5,
            'tsfm_n_head': 8,
            'tsfm_d_model': 512,
            'tsfm_d_inner': 2048
        }
        self.config.update(config)
        
        # Encoder
        self.encoder = nn.ModuleList()
        ch = self.config['channels_H']
        for i in range(self.config['n_layers']):
            conv = ConvNorm(
                self.config['channels_input'] if i == 0 else ch,
                ch,
                kernel_size=self.config['kernel_size'],
                stride=self.config['stride']
            )
            self.encoder.append(conv)
            ch = min(ch * 2, self.config['max_channels'])
        
        # Transformer layers
        self.transformer = nn.ModuleList()
        for _ in range(self.config['tsfm_n_layers']):
            self.transformer.append(
                MultiHeadAttention(
                    self.config['tsfm_d_model'],
                    self.config['tsfm_n_head']
                )
            )
        
        # Decoder
        self.decoder = nn.ModuleList()
        ch = self.config['max_channels']
        for i in range(self.config['n_layers']):
            conv = nn.ConvTranspose1d(
                ch + (ch // 2 if i > 0 else self.config['channels_H']),
                ch // 2 if i < self.config['n_layers'] - 1 else self.config['channels_output'],
                kernel_size=self.config['kernel_size'],
                stride=self.config['stride'],
                padding=1
            )
            self.decoder.append(conv)
            ch = ch // 2
        
        # Projection for transformer
        self.proj = LinearNorm(ch * 4, self.config['tsfm_d_model'])
        self.proj_back = LinearNorm(self.config['tsfm_d_model'], ch * 4)

    def forward(self, audio):
        # Encoder
        encoder_outputs = []
        x = audio
        for conv in self.encoder:
            x = conv(x)
            x = torch.relu(x)
            encoder_outputs.append(x)
        
        # Prepare for transformer
        B, C, T = x.shape
        x = x.permute(0, 2, 1).reshape(B, T, C)
        x = self.proj(x)
        
        # Transformer layers
        for transformer_layer in self.transformer:
            x = transformer_layer(x, x, x)
        
        # Project back
        x = self.proj_back(x)
        x = x.reshape(B, T, C).permute(0, 2, 1)
        
        # Decoder with skip connections
        for i, conv in enumerate(self.decoder):
            if i > 0:
                x = torch.cat([x, encoder_outputs[-(i+1)]], dim=1)
            x = conv(x)
            if i < len(self.decoder) - 1:
                x = torch.relu(x)
        
        return x

# ===== AUDIO PROCESSING FUNCTIONS =====
def load_audio(file_path, target_sr=16000):
    """Load audio file and resample to target sample rate"""
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if necessary
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
            waveform = resampler(waveform)
        
        return waveform, target_sr
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        return None, None

def save_audio(waveform, file_path, sample_rate=16000):
    """Save audio waveform to file"""
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Normalize audio to prevent clipping
        waveform = waveform / torch.max(torch.abs(waveform))
        
        torchaudio.save(file_path, waveform.cpu(), sample_rate)
        print(f"Cleaned audio saved to: {file_path}")
    except Exception as e:
        print(f"Error saving audio file {file_path}: {e}")

def chunk_audio(waveform, chunk_length=3.0, sample_rate=16000, overlap=0.1):
    """Split audio into overlapping chunks for processing"""
    chunk_samples = int(chunk_length * sample_rate)
    overlap_samples = int(overlap * sample_rate)
    step = chunk_samples - overlap_samples
    
    chunks = []
    for i in range(0, waveform.shape[1] - chunk_samples + 1, step):
        chunk = waveform[:, i:i + chunk_samples]
        chunks.append((chunk, i))
    
    # Handle the last chunk if necessary
    if waveform.shape[1] % step != 0:
        last_chunk = waveform[:, -chunk_samples:]
        chunks.append((last_chunk, waveform.shape[1] - chunk_samples))
    
    return chunks

def reconstruct_audio(chunks, original_length, chunk_length=3.0, sample_rate=16000, overlap=0.1):
    """Reconstruct audio from overlapping chunks"""
    chunk_samples = int(chunk_length * sample_rate)
    overlap_samples = int(overlap * sample_rate)
    step = chunk_samples - overlap_samples
    
    reconstructed = torch.zeros(1, original_length)
    weight_sum = torch.zeros(1, original_length)
    
    for chunk, start_idx in chunks:
        end_idx = min(start_idx + chunk_samples, original_length)
        chunk_len = end_idx - start_idx
        
        # Apply windowing for smooth blending
        window = torch.hann_window(chunk_len)
        windowed_chunk = chunk[:, :chunk_len] * window
        
        reconstructed[:, start_idx:end_idx] += windowed_chunk
        weight_sum[:, start_idx:end_idx] += window
    
    # Normalize by window overlap
    weight_sum = torch.clamp(weight_sum, min=1e-8)
    reconstructed = reconstructed / weight_sum
    
    return reconstructed

# ===== MODEL LOADING AND INFERENCE =====
def load_cleanunet_model(model_path, device="cpu"):
    """Load CleanUNet model from pretrained.pkl file"""
    try:
        print(f"Loading CleanUNet model from: {model_path}")
        
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract model configuration
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            # Default DNS-large-high configuration
            config = {
                'n_layers': 8,
                'channels_input': 1,
                'channels_output': 1,
                'channels_H': 64,
                'max_channels': 768,
                'kernel_size': 3,
                'stride': 2,
                'tsfm_n_layers': 5,
                'tsfm_n_head': 8,
                'tsfm_d_model': 512,
                'tsfm_d_inner': 2048
            }
        
        # Initialize model
        model = CleanUNet(config)
        
        # Load model weights
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        print(f"Model loaded successfully on {device}")
        return model
    
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure the pretrained.pkl file exists and is accessible")
        return None

def denoise_audio_chunk(model, audio_chunk, device="cpu"):
    """Denoise a single audio chunk"""
    with torch.no_grad():
        audio_chunk = audio_chunk.to(device)
        
        # Add batch dimension if not present
        if audio_chunk.dim() == 2:
            audio_chunk = audio_chunk.unsqueeze(0)
        
        # Model inference
        enhanced = model(audio_chunk)
        
        return enhanced.squeeze(0).cpu()

def denoise_full_audio(model, waveform, device="cpu", chunk_length=3.0):
    """Denoise full audio using chunked processing"""
    print("Processing audio in chunks for optimal memory usage...")
    
    original_length = waveform.shape[1]
    chunks = chunk_audio(waveform, chunk_length=chunk_length)
    
    enhanced_chunks = []
    for i, (chunk, start_idx) in enumerate(tqdm(chunks, desc="Denoising chunks")):
        try:
            enhanced_chunk = denoise_audio_chunk(model, chunk, device)
            enhanced_chunks.append((enhanced_chunk, start_idx))
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"GPU memory error on chunk {i}, trying smaller chunk...")
                # Try with smaller chunk
                smaller_chunks = chunk_audio(chunk, chunk_length=1.5)
                for small_chunk, _ in smaller_chunks:
                    enhanced_small = denoise_audio_chunk(model, small_chunk, device)
                    enhanced_chunks.append((enhanced_small, start_idx))
            else:
                raise e
    
    # Reconstruct full audio
    enhanced_audio = reconstruct_audio(enhanced_chunks, original_length, chunk_length)
    
    return enhanced_audio

# ===== MAIN PROCESSING FUNCTION =====
def main():
    """Main processing function"""
    print("=== CleanUNet Audio Denoiser ===")
    print(f"Device: {DEVICE}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Input audio: {INPUT_AUDIO_PATH}")
    print(f"Output audio: {OUTPUT_AUDIO_PATH}")
    print()
    
    # Validate file paths
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please download the DNS-large-high pretrained.pkl file and update MODEL_PATH")
        return
    
    if not os.path.exists(INPUT_AUDIO_PATH):
        print(f"Error: Input audio file not found at {INPUT_AUDIO_PATH}")
        print("Please place your noisy audio file and update INPUT_AUDIO_PATH")
        return
    
    # Load model
    model = load_cleanunet_model(MODEL_PATH, DEVICE)
    if model is None:
        return
    
    # Load input audio
    print("Loading input audio...")
    waveform, sample_rate = load_audio(INPUT_AUDIO_PATH, SAMPLE_RATE)
    if waveform is None:
        return
    
    print(f"Audio loaded: {waveform.shape[1]/sample_rate:.2f} seconds, {sample_rate} Hz")
    
    # Denoise audio
    print("Starting denoising process...")
    try:
        enhanced_audio = denoise_full_audio(model, waveform, DEVICE)
        
        # Save output
        save_audio(enhanced_audio, OUTPUT_AUDIO_PATH, SAMPLE_RATE)
        print("Denoising completed successfully!")
        
    except Exception as e:
        print(f"Error during denoising: {e}")
        print("Try reducing chunk length or using CPU if GPU memory is insufficient")

if __name__ == "__main__":
    # You can also use command line arguments
    parser = argparse.ArgumentParser(description="CleanUNet Audio Denoising")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to pretrained.pkl file")
    parser.add_argument("--input", type=str, default=INPUT_AUDIO_PATH, help="Input noisy audio file")
    parser.add_argument("--output", type=str, default=OUTPUT_AUDIO_PATH, help="Output cleaned audio file")
    parser.add_argument("--device", type=str, default=DEVICE, help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Update global variables with command line arguments
    MODEL_PATH = args.model
    INPUT_AUDIO_PATH = args.input
    OUTPUT_AUDIO_PATH = args.output
    DEVICE = args.device
    
    main()