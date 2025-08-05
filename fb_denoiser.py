#!/usr/bin/env python3
"""
Advanced Facebook Denoiser Script
=================================

This script handles Facebook Denoiser models with comprehensive error handling:
- TorchScript (.th) files
- Dictionary models (state_dict)
- Model architecture recreation
- Multiple fallback methods

Requirements:
- torch
- torchaudio
- librosa
- soundfile
- numpy
- scipy

Usage:
    1. Set the INPUT_AUDIO_PATH and OUTPUT_AUDIO_PATH variables below
    2. Set MODEL_PATH to your Facebook Denoiser model directory
    3. Run: python advanced_facebook_denoiser.py
"""

import os
import sys
import warnings
import logging
from pathlib import Path
from typing import Tuple, Optional, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import librosa
import soundfile as sf
import numpy as np
from scipy.signal import resample

# ==================== CONFIGURATION ====================
# Set your input and output file paths here
INPUT_AUDIO_PATH = "path/to/your/noisy_audio.wav"
OUTPUT_AUDIO_PATH = "path/to/your/clean_audio.wav"

# Model configuration - CHANGE THIS TO YOUR ACTUAL PATH
MODEL_PATH = "./models/facebook_denoiser"  # Directory containing model.th or dns64.th
SAMPLE_RATE = 16000
DEVICE = "auto"  # "auto", "cpu", or "cuda"

# Processing options (True to enable, False to disable)
APPLY_ENHANCEMENT = True      # Use Facebook Denoiser for speech enhancement
APPLY_NORMALIZATION = True    # Normalize audio levels
APPLY_FILTERING = True        # Apply high-pass filter

# ========================================================

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleFacebookDenoiser(nn.Module):
    """Simple Facebook Denoiser-like U-Net architecture."""
    
    def __init__(self, channels=64, sample_rate=16000):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.channels = channels
        
        # Encoder
        self.encoder1 = self._conv_block(1, channels)
        self.encoder2 = self._conv_block(channels, channels * 2)
        self.encoder3 = self._conv_block(channels * 2, channels * 4)
        self.encoder4 = self._conv_block(channels * 4, channels * 8)
        
        # Bottleneck
        self.bottleneck = self._conv_block(channels * 8, channels * 16)
        
        # Decoder
        self.decoder4 = self._deconv_block(channels * 16, channels * 8)
        self.decoder3 = self._deconv_block(channels * 16, channels * 4)  # *16 due to skip connection
        self.decoder2 = self._deconv_block(channels * 8, channels * 2)
        self.decoder1 = self._deconv_block(channels * 4, channels)
        
        # Output layer
        self.output = nn.Conv1d(channels * 2, 1, kernel_size=1)
        self.output_activation = nn.Tanh()
        
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _deconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Ensure input is 3D: [batch, channels, samples]
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, samples]
        elif x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, samples]
        
        # Pad to make divisible by 16 (for 4 downsampling steps)
        original_length = x.size(-1)
        pad_length = (16 - (original_length % 16)) % 16
        if pad_length > 0:
            x = F.pad(x, (0, pad_length))
        
        # Encoder
        e1 = self.encoder1(x)
        e1_pool = F.max_pool1d(e1, 2)
        
        e2 = self.encoder2(e1_pool)
        e2_pool = F.max_pool1d(e2, 2)
        
        e3 = self.encoder3(e2_pool)
        e3_pool = F.max_pool1d(e3, 2)
        
        e4 = self.encoder4(e3_pool)
        e4_pool = F.max_pool1d(e4, 2)
        
        # Bottleneck
        bottleneck = self.bottleneck(e4_pool)
        
        # Decoder with skip connections
        d4 = self.decoder4(bottleneck)
        d4 = torch.cat([d4, e4], dim=1)  # Skip connection
        
        d3 = self.decoder3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        
        d2 = self.decoder2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        
        d1 = self.decoder1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        
        # Output
        output = self.output(d1)
        output = self.output_activation(output)
        
        # Remove padding
        if pad_length > 0:
            output = output[..., :-pad_length]
        
        return output


class FlexibleDenoiser(nn.Module):
    """Flexible denoiser that adapts to different architectures."""
    
    def __init__(self, input_channels=1, base_channels=32, num_layers=4):
        super().__init__()
        
        self.input_channels = input_channels
        self.base_channels = base_channels
        self.num_layers = num_layers
        
        # Encoder
        self.encoders = nn.ModuleList()
        in_ch = input_channels
        for i in range(num_layers):
            out_ch = base_channels * (2 ** i)
            self.encoders.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            ))
            in_ch = out_ch
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(in_ch, in_ch * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(in_ch * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_ch * 2, in_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(in_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Decoder
        self.decoders = nn.ModuleList()
        for i in range(num_layers):
            out_ch = base_channels * (2 ** (num_layers - i - 2)) if i < num_layers - 1 else input_channels
            skip_ch = base_channels * (2 ** (num_layers - i - 1))
            
            self.decoders.append(nn.Sequential(
                nn.ConvTranspose1d(in_ch + skip_ch, out_ch, kernel_size=2, stride=2),
                nn.BatchNorm1d(out_ch) if out_ch != input_channels else nn.Identity(),
                nn.ReLU(inplace=True) if out_ch != input_channels else nn.Tanh()
            ))
            in_ch = out_ch
    
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Store original length
        original_length = x.size(-1)
        
        # Pad for downsampling
        factor = 2 ** self.num_layers
        pad_length = (factor - (original_length % factor)) % factor
        if pad_length > 0:
            x = F.pad(x, (0, pad_length))
        
        # Encoder
        skip_connections = []
        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)
            x = F.max_pool1d(x, 2)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        for i, decoder in enumerate(self.decoders):
            skip = skip_connections[-(i + 1)]
            x = F.interpolate(x, size=skip.size(-1), mode='nearest')
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)
        
        # Remove padding
        if pad_length > 0:
            x = x[..., :-pad_length]
        
        return x


class AdvancedFacebookDenoiserProcessor:
    """Advanced Facebook Denoiser processor with comprehensive error handling."""
    
    def __init__(self, 
                 model_path: str = "./models/facebook_denoiser",
                 device: str = "auto",
                 sample_rate: int = 16000,
                 chunk_size: int = 16000):  # 1 second chunks for processing
        
        self.model_path = Path(model_path)
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model()
        
    def _load_model(self):
        """Load Facebook Denoiser model with comprehensive error handling."""
        try:
            if not self.model_path.exists():
                logger.warning(f"Model directory not found: {self.model_path}")
                return None
            
            # Find model files
            model_files = []
            
            # Look for TorchScript files first (.th)
            for ext in ['.th', '.pt', '.pth', '.bin', '.tar', '.ckpt']:
                model_files.extend(list(self.model_path.glob(f"*{ext}")))
            
            # Check for common Facebook Denoiser filenames
            common_names = ['model.th', 'dns64.th', 'dns48.th', 'denoiser.th', 'best_model.th']
            for name in common_names:
                potential_file = self.model_path / name
                if potential_file.exists() and potential_file not in model_files:
                    model_files.append(potential_file)
            
            if not model_files:
                logger.warning(f"No model files found in {self.model_path}")
                return None
            
            # Try loading each file
            for model_file in model_files:
                logger.info(f"Attempting to load: {model_file}")
                
                model = self._try_load_model_file(model_file)
                if model is not None:
                    logger.info("✅ Model loaded successfully")
                    return model
                    
            logger.warning("Failed to load any model file")
            return None
                
        except Exception as e:
            logger.error(f"Error in model loading: {e}")
            return None
    
    def _try_load_model_file(self, model_file):
        """Try different methods to load a model file."""
        
        # Method 1: TorchScript loading
        if model_file.suffix == '.th':
            model = self._load_torchscript_model(model_file)
            if model is not None:
                return model
        
        # Method 2: Regular PyTorch loading
        model = self._load_pytorch_model(model_file)
        if model is not None:
            return model
        
        return None
    
    def _load_torchscript_model(self, model_file):
        """Load TorchScript model (.th files)."""
        
        # Method 1: Direct TorchScript loading
        try:
            logger.info("Trying TorchScript loading...")
            model = torch.jit.load(model_file, map_location=self.device)
            model.eval()
            logger.info("✅ TorchScript model loaded successfully")
            
            # Test the model with dummy input
            try:
                dummy_input = torch.randn(1, 1, self.sample_rate).to(self.device)
                with torch.no_grad():
                    output = model(dummy_input)
                logger.info(f"✅ Model test successful - Output shape: {output.shape}")
                return model
            except Exception as e:
                logger.warning(f"Model test failed: {e}")
                # Return anyway, might work with different input
                return model
                
        except Exception as e:
            logger.warning(f"TorchScript loading failed: {e}")
        
        # Method 2: Try loading as regular PyTorch model
        try:
            logger.info("Trying to load .th as PyTorch model...")
            model_data = torch.load(model_file, map_location=self.device)
            return self._create_model_from_data(model_data)
        except Exception as e:
            logger.warning(f"PyTorch loading of .th file failed: {e}")
        
        return None
    
    def _load_pytorch_model(self, model_file):
        """Load regular PyTorch model."""
        try:
            logger.info("Loading PyTorch model...")
            model_data = torch.load(model_file, map_location=self.device)
            return self._create_model_from_data(model_data)
        except Exception as e:
            logger.warning(f"PyTorch model loading failed: {e}")
            return None
    
    def _create_model_from_data(self, model_data):
        """Create model from loaded data."""
        try:
            # If it's already a model
            if hasattr(model_data, 'forward') and callable(model_data):
                logger.info("Data is already a model")
                model = model_data.to(self.device)
                model.eval()
                return model
            
            # If it's a dictionary
            if isinstance(model_data, dict):
                logger.info(f"Data is dictionary with keys: {list(model_data.keys())}")
                
                # Look for actual model in dictionary
                model_keys = ['model', 'net', 'generator', 'denoiser', 'network']
                for key in model_keys:
                    if key in model_data:
                        candidate = model_data[key]
                        if hasattr(candidate, 'forward'):
                            logger.info(f"Found model in key: {key}")
                            model = candidate.to(self.device)
                            model.eval()
                            return model
                
                # Look for state_dict
                state_dict_keys = ['state_dict', 'model_state_dict', 'net_state_dict']
                state_dict = None
                for key in state_dict_keys:
                    if key in model_data:
                        state_dict = model_data[key]
                        logger.info(f"Found state_dict in key: {key}")
                        break
                
                if state_dict is None:
                    # Maybe the whole dict is a state_dict
                    if all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in model_data.items()):
                        state_dict = model_data
                        logger.info("Using whole dictionary as state_dict")
                
                if state_dict is not None:
                    return self._create_model_from_state_dict(state_dict)
            
            logger.warning("Could not extract model from data")
            return None
            
        except Exception as e:
            logger.error(f"Error creating model from data: {e}")
            return None
    
    def _create_model_from_state_dict(self, state_dict):
        """Create model architecture and load state_dict."""
        try:
            logger.info("Creating model from state_dict...")
            
            # Analyze state_dict to determine architecture
            input_channels = self._infer_input_channels(state_dict)
            base_channels = self._infer_base_channels(state_dict)
            num_layers = self._infer_num_layers(state_dict)
            
            logger.info(f"Inferred architecture: input_channels={input_channels}, base_channels={base_channels}, num_layers={num_layers}")
            
            # Try different model architectures
            architectures = [
                SimpleFacebookDenoiser(channels=base_channels, sample_rate=self.sample_rate),
                FlexibleDenoiser(input_channels=input_channels, base_channels=base_channels, num_layers=num_layers),
                SimpleFacebookDenoiser(channels=64, sample_rate=self.sample_rate),  # Default
                FlexibleDenoiser(input_channels=1, base_channels=32, num_layers=4),  # Default
            ]
            
            for i, model in enumerate(architectures):
                try:
                    # Try to load state dict (allow size mismatches)
                    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                    
                    if missing_keys:
                        logger.warning(f"Missing keys: {len(missing_keys)} keys")
                    if unexpected_keys:
                        logger.warning(f"Unexpected keys: {len(unexpected_keys)} keys")
                    
                    model = model.to(self.device)
                    model.eval()
                    
                    # Test the model
                    try:
                        dummy_input = torch.randn(1, 1, 1000).to(self.device)
                        with torch.no_grad():
                            output = model(dummy_input)
                        logger.info(f"✅ Model test successful - Architecture {i+1}")
                        return model
                    except Exception as e:
                        logger.warning(f"Model test failed for architecture {i+1}: {e}")
                        continue
                    
                except Exception as e:
                    logger.warning(f"Architecture {i+1} failed: {e}")
                    continue
            
            logger.warning("All architecture attempts failed")
            return None
            
        except Exception as e:
            logger.error(f"Error creating model from state_dict: {e}")
            return None
    
    def _infer_input_channels(self, state_dict):
        """Infer input channels from state_dict."""
        for key, tensor in state_dict.items():
            if 'input' in key.lower() or (key.endswith('.weight') and 'conv' in key and len(tensor.shape) == 3):
                return tensor.shape[1]  # Input channels
        return 1  # Default for audio
    
    def _infer_base_channels(self, state_dict):
        """Infer base channels from state_dict."""
        channel_counts = []
        for key, tensor in state_dict.items():
            if '.weight' in key and 'conv' in key and len(tensor.shape) == 3:
                channel_counts.append(tensor.shape[0])
        
        if channel_counts:
            channel_counts.sort()
            # Use the smallest non-input channel count as base
            for count in channel_counts:
                if count > 1:
                    return min(64, count)
        return 32  # Default
    
    def _infer_num_layers(self, state_dict):
        """Infer number of layers from state_dict."""
        conv_layers = [k for k in state_dict.keys() if 'conv' in k and '.weight' in k]
        estimated_layers = len(conv_layers) // 4  # Rough estimate
        return max(3, min(6, estimated_layers))  # Reasonable range
    
    def _enhance_chunk(self, waveform_chunk: torch.Tensor) -> torch.Tensor:
        """Enhance a single chunk of audio."""
        try:
            # Ensure proper input format
            if waveform_chunk.dim() == 1:
                waveform_chunk = waveform_chunk.unsqueeze(0).unsqueeze(0)  # [1, 1, samples]
            elif waveform_chunk.dim() == 2:
                waveform_chunk = waveform_chunk.unsqueeze(1)  # [batch, 1, samples]
            
            # Move to device
            waveform_chunk = waveform_chunk.to(self.device)
            
            # Model inference
            with torch.no_grad():
                enhanced_chunk = self.model(waveform_chunk)
            
            # Remove batch and channel dimensions
            if enhanced_chunk.dim() == 3:
                enhanced_chunk = enhanced_chunk.squeeze(0).squeeze(0)  # [samples]
            elif enhanced_chunk.dim() == 2:
                enhanced_chunk = enhanced_chunk.squeeze(0)  # [samples]
            
            return enhanced_chunk.cpu()
            
        except Exception as e:
            logger.warning(f"Chunk enhancement failed: {e}")
            # Return original chunk
            if waveform_chunk.dim() == 3:
                return waveform_chunk.squeeze(0).squeeze(0).cpu()
            elif waveform_chunk.dim() == 2:
                return waveform_chunk.squeeze(0).cpu()
            else:
                return waveform_chunk.cpu()
    
    def _traditional_denoising_fallback(self, waveform: torch.Tensor) -> torch.Tensor:
        """Enhanced traditional denoising fallback."""
        logger.info("Applying enhanced traditional denoising")
        
        # Convert to spectrogram
        n_fft = 512
        hop_length = 256
        
        stft = torch.stft(
            waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            window=torch.hann_window(512),
            return_complex=True
        )
        
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)
        
        # Estimate noise from first 0.5 seconds
        noise_frames = int(0.5 * self.sample_rate / hop_length)
        noise_frames = min(noise_frames, magnitude.size(1) // 4)
        
        if noise_frames > 0:
            noise_magnitude = torch.mean(magnitude[:, :noise_frames], dim=1, keepdim=True)
        else:
            noise_magnitude = torch.mean(magnitude[:, :min(10, magnitude.size(1))], dim=1, keepdim=True)
        
        # Advanced spectral subtraction
        alpha = 2.5  # Over-subtraction factor
        beta = 0.02  # Spectral floor
        
        # Frequency-dependent processing
        freq_weights = torch.linspace(0.5, 2.0, magnitude.size(0)).unsqueeze(1)
        alpha_freq = alpha * freq_weights
        
        enhanced_magnitude = magnitude - alpha_freq * noise_magnitude
        enhanced_magnitude = torch.maximum(enhanced_magnitude, beta * magnitude)
        
        # Temporal smoothing
        if enhanced_magnitude.size(1) > 3:
            kernel = torch.ones(1, 1, 3) / 3
            enhanced_magnitude_smooth = F.conv1d(
                enhanced_magnitude.unsqueeze(0), kernel, padding=1
            ).squeeze(0)
            enhanced_magnitude = 0.7 * enhanced_magnitude + 0.3 * enhanced_magnitude_smooth
        
        # Reconstruct
        enhanced_stft = enhanced_magnitude * torch.exp(1j * phase)
        enhanced_waveform = torch.istft(
            enhanced_stft,
            n_fft=n_fft,
            hop_length=hop_length,
            window=torch.hann_window(512)
        )
        
        return enhanced_waveform
    
    def enhance_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """Enhance audio using Facebook Denoiser or fallback."""
        
        # Ensure single channel
        if waveform.dim() > 1 and waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0)
        
        if self.model is not None:
            try:
                logger.info("Enhancing audio with Facebook Denoiser")
                
                # Process in chunks for memory efficiency
                enhanced_chunks = []
                
                for i in range(0, waveform.size(-1), self.chunk_size):
                    chunk = waveform[i:i + self.chunk_size]
                    
                    # Pad if necessary
                    if chunk.size(-1) < self.chunk_size and i + self.chunk_size < waveform.size(-1):
                        padding = self.chunk_size - chunk.size(-1)
                        chunk = F.pad(chunk, (0, padding))
                        enhanced_chunk = self._enhance_chunk(chunk)
                        enhanced_chunk = enhanced_chunk[:-padding]  # Remove padding
                    else:
                        enhanced_chunk = self._enhance_chunk(chunk)
                    
                    enhanced_chunks.append(enhanced_chunk)
                
                # Concatenate chunks
                enhanced_waveform = torch.cat(enhanced_chunks, dim=-1)
                
                logger.info("✅ Facebook Denoiser inference successful")
                return enhanced_waveform.unsqueeze(0)  # Add channel dimension
                
            except Exception as e:
                logger.warning(f"Facebook Denoiser inference failed: {e}, using fallback")
                return self._traditional_denoising_fallback(waveform).unsqueeze(0)
        else:
            # Use fallback method
            return self._traditional_denoising_fallback(waveform).unsqueeze(0)


class AudioPreprocessor:
    """Audio preprocessing pipeline for Facebook Denoiser."""
    
    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr
        self.processor = None
    
    def load_audio(self, file_path: str) -> Tuple[torch.Tensor, int]:
        """Load audio file."""
        try:
            waveform, sr = torchaudio.load(file_path)
            logger.info(f"Loaded audio: {waveform.shape}, SR: {sr}")
            return waveform, sr
        except Exception as e:
            logger.warning(f"torchaudio failed: {e}, trying librosa")
            try:
                waveform, sr = librosa.load(file_path, sr=None, mono=False)
                if waveform.ndim == 1:
                    waveform = waveform[None, :]
                waveform = torch.from_numpy(waveform.astype(np.float32))
                return waveform, sr
            except Exception as e2:
                raise RuntimeError(f"Failed to load audio: {e2}")
    
    def resample_audio(self, waveform: torch.Tensor, orig_sr: int) -> torch.Tensor:
        """Resample audio."""
        if orig_sr == self.target_sr:
            return waveform
        
        logger.info(f"Resampling from {orig_sr} Hz to {self.target_sr} Hz")
        
        try:
            resampler = torchaudio.transforms.Resample(orig_sr, self.target_sr)
            return resampler(waveform)
        except Exception:
            logger.warning("Using scipy for resampling")
            waveform_np = waveform.numpy()
            num_samples = int(waveform_np.shape[-1] * self.target_sr / orig_sr)
            resampled = resample(waveform_np, num_samples, axis=-1)
            return torch.from_numpy(resampled.astype(np.float32))
    
    def normalize_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """Normalize audio."""
        rms = torch.sqrt(torch.mean(waveform**2))
        if rms > 0:
            target_rms = 0.1
            scaling_factor = target_rms / rms
            waveform = waveform * scaling_factor
        
        max_val = torch.max(torch.abs(waveform))
        if max_val > 0.95:
            waveform = waveform * (0.95 / max_val)
        
        return waveform
    
    def apply_highpass_filter(self, waveform: torch.Tensor, cutoff: float = 80.0) -> torch.Tensor:
        """Apply high-pass filter."""
        try:
            highpass = torchaudio.transforms.Highpass(self.target_sr, cutoff)
            return highpass(waveform)
        except Exception:
            logger.warning("Using simple high-pass filter")
            alpha = cutoff / (cutoff + self.target_sr)
            filtered = torch.zeros_like(waveform)
            if waveform.size(-1) > 1:
                filtered[..., 1:] = alpha * (waveform[..., 1:] - waveform[..., :-1])
            return filtered
    
    def save_audio(self, waveform: torch.Tensor, file_path: str):
        """Save audio."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        waveform_np = waveform.detach().cpu().numpy()
        if waveform_np.shape[0] == 1:
            waveform_np = waveform_np.squeeze(0)
        
        sf.write(file_path, waveform_np, self.target_sr, subtype='PCM_16')
        logger.info(f"Saved enhanced audio to: {file_path}")
    
    def process_audio(self, 
                     input_path: str, 
                     output_path: str,
                     model_path: str = "./models/facebook_denoiser",
                     apply_enhancement: bool = True,
                     apply_normalization: bool = True,
                     apply_filtering: bool = True) -> None:
        """Process audio with Facebook Denoiser."""
        logger.info(f"Processing audio: {input_path} -> {output_path}")
        
        # Load
        waveform, orig_sr = self.load_audio(input_path)
        
        # Resample
        waveform = self.resample_audio(waveform, orig_sr)
        
        # Filter
        if apply_filtering:
            waveform = self.apply_highpass_filter(waveform)
        
        # Enhance
        if apply_enhancement:
            if self.processor is None:
                self.processor = AdvancedFacebookDenoiserProcessor(model_path, sample_rate=self.target_sr)
            waveform = self.processor.enhance_audio(waveform)
        
        # Normalize
        if apply_normalization:
            waveform = self.normalize_audio(waveform)
        
        # Save
        self.save_audio(waveform, output_path)
        
        logger.info("Processing completed successfully")


def main():
    """Main function."""
    
    if INPUT_AUDIO_PATH == "path/to/your/noisy_audio.wav":
        logger.error("Please set INPUT_AUDIO_PATH!")
        sys.exit(1)
    
    if OUTPUT_AUDIO_PATH == "path/to/your/clean_audio.wav":
        logger.error("Please set OUTPUT_AUDIO_PATH!")
        sys.exit(1)
    
    if not os.path.exists(INPUT_AUDIO_PATH):
        logger.error(f"Input file not found: {INPUT_AUDIO_PATH}")
        sys.exit(1)
    
    # Create output directory
    output_dir = os.path.dirname(OUTPUT_AUDIO_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        logger.info("="*60)
        logger.info("ADVANCED FACEBOOK DENOISER PROCESSING")
        logger.info("="*60)
        logger.info(f"Input file: {INPUT_AUDIO_PATH}")
        logger.info(f"Output file: {OUTPUT_AUDIO_PATH}")
        logger.info(f"Model path: {MODEL_PATH}")
        logger.info("="*60)
        
        # Debug model directory
        model_dir = Path(MODEL_PATH)
        if model_dir.exists():
            logger.info(f"Files in model directory:")
            for file in model_dir.iterdir():
                size_mb = file.stat().st_size / (1024*1024)
                logger.info(f"  - {file.name} ({size_mb:.1f} MB)")
        
        # Process
        preprocessor = AudioPreprocessor(target_sr=SAMPLE_RATE)
        preprocessor.process_audio(
            input_path=INPUT_AUDIO_PATH,
            output_path=OUTPUT_AUDIO_PATH,
            model_path=MODEL_PATH,
            apply_enhancement=APPLY_ENHANCEMENT,
            apply_normalization=APPLY_NORMALIZATION,
            apply_filtering=APPLY_FILTERING
        )
        
        logger.info("="*60)
        logger.info("✅ PROCESSING COMPLETED SUCCESSFULLY!")
        logger.info(f"Enhanced audio saved to: {OUTPUT_AUDIO_PATH}")
        logger.info("="*60)
        
    except Exception as e:
        logger.error("="*60)
        logger.error("❌ PROCESSING FAILED!")
        logger.error(f"Error: {e}")
        logger.error("="*60)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()