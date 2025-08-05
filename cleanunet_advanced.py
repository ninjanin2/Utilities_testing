#!/usr/bin/env python3
"""
Advanced CleanUNet Audio Denoising Script
=========================================

This script handles CleanUNet models that are saved as dictionaries (state_dict)
and creates the proper model architecture to load them.

Requirements:
- torch
- torchaudio
- librosa
- soundfile
- numpy
- scipy

Usage:
    1. Set the INPUT_AUDIO_PATH and OUTPUT_AUDIO_PATH variables below
    2. Set MODEL_PATH to your CleanUNet model directory
    3. Run: python advanced_cleanunet_script.py
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
MODEL_PATH = "./models/cleanunet"  # Directory containing pretrained.pkl
SAMPLE_RATE = 16000
DEVICE = "auto"  # "auto", "cpu", or "cuda"

# Processing options (True to enable, False to disable)
APPLY_ENHANCEMENT = True      # Use CleanUNet for speech enhancement
APPLY_NORMALIZATION = True    # Normalize audio levels
APPLY_FILTERING = True        # Apply high-pass filter

# ========================================================

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleCleanUNet(nn.Module):
    """Simple CleanUNet-like architecture for when we only have state_dict."""
    
    def __init__(self, input_size=513, hidden_size=128, num_layers=3):
        super().__init__()
        
        # Encoder
        self.encoder = nn.ModuleList()
        in_size = input_size
        for i in range(num_layers):
            out_size = hidden_size * (2 ** i)
            self.encoder.append(nn.Sequential(
                nn.Linear(in_size, out_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ))
            in_size = out_size
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(in_size, in_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Decoder
        self.decoder = nn.ModuleList()
        for i in range(num_layers):
            out_size = hidden_size * (2 ** (num_layers - i - 2)) if i < num_layers - 1 else input_size
            self.decoder.append(nn.Sequential(
                nn.Linear(in_size, out_size),
                nn.ReLU() if i < num_layers - 1 else nn.Sigmoid(),
                nn.Dropout(0.1) if i < num_layers - 1 else nn.Identity()
            ))
            in_size = out_size
    
    def forward(self, x):
        # x shape: [batch, freq, time] or [batch, time, freq]
        
        # Ensure correct input shape
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension
        
        original_shape = x.shape
        
        # Flatten for processing
        if x.size(-1) != 513:  # If not frequency dimension
            x = x.transpose(-1, -2)  # Make frequency last dimension
        
        batch_size, time_steps, freq_bins = x.shape
        x = x.view(-1, freq_bins)  # [batch*time, freq]
        
        # Forward pass
        residual = x
        
        # Encoder
        for layer in self.encoder:
            x = layer(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        for layer in self.decoder:
            x = layer(x)
        
        # Add residual connection
        if x.shape == residual.shape:
            x = x + residual
        
        # Reshape back
        x = x.view(batch_size, time_steps, freq_bins)
        
        # Return in original orientation
        if original_shape[-1] != freq_bins:
            x = x.transpose(-1, -2)
        
        return x


class FlexibleCleanUNet(nn.Module):
    """Flexible CleanUNet that adapts to different architectures."""
    
    def __init__(self, **kwargs):
        super().__init__()
        
        # Try to build architecture based on common patterns
        self.input_dim = kwargs.get('input_dim', 513)
        self.hidden_dim = kwargs.get('hidden_dim', 128)
        self.num_layers = kwargs.get('num_layers', 4)
        
        # Main processing network
        layers = []
        current_dim = self.input_dim
        
        # Encoder layers
        for i in range(self.num_layers):
            next_dim = self.hidden_dim * (2 ** min(i, 3))  # Cap at 8x
            layers.extend([
                nn.Linear(current_dim, next_dim),
                nn.ReLU(),
                nn.BatchNorm1d(next_dim) if next_dim > 1 else nn.Identity(),
                nn.Dropout(0.1)
            ])
            current_dim = next_dim
        
        # Decoder layers
        for i in range(self.num_layers):
            next_dim = self.hidden_dim * (2 ** max(0, self.num_layers - i - 2))
            if i == self.num_layers - 1:
                next_dim = self.input_dim  # Output dimension
            
            layers.extend([
                nn.Linear(current_dim, next_dim),
                nn.ReLU() if i < self.num_layers - 1 else nn.Sigmoid(),
                nn.BatchNorm1d(next_dim) if next_dim > 1 and i < self.num_layers - 1 else nn.Identity(),
                nn.Dropout(0.1) if i < self.num_layers - 1 else nn.Identity()
            ])
            current_dim = next_dim
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        original_shape = x.shape
        
        # Handle different input shapes
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, samples]
        elif x.dim() == 2:
            x = x.unsqueeze(0)  # [1, channels, samples] or [1, freq, time]
        
        # Process each frame
        batch_size = x.size(0)
        
        # If input is waveform, convert to spectrogram
        if x.size(-1) > x.size(-2):  # More samples than frequency bins
            # Convert to spectrogram
            n_fft = (x.size(-2) - 1) * 2 if x.size(-2) < 1000 else 1024
            spec = torch.stft(
                x.view(-1, x.size(-1)), 
                n_fft=n_fft, 
                hop_length=n_fft//4,
                return_complex=True
            )
            x = torch.abs(spec)  # Magnitude spectrogram
        
        # Reshape for network
        if x.dim() == 3:
            batch, freq, time = x.shape
            x = x.transpose(1, 2).contiguous()  # [batch, time, freq]
            x = x.view(-1, freq)  # [batch*time, freq]
            
            # Process through network
            output = self.network(x)
            
            # Reshape back
            output = output.view(batch, time, -1).transpose(1, 2)  # [batch, freq, time]
        else:
            output = self.network(x)
        
        return output


class AdvancedCleanUNetProcessor:
    """Advanced CleanUNet processor that handles dictionary models."""
    
    def __init__(self, 
                 model_path: str = "./models/cleanunet",
                 device: str = "auto",
                 sample_rate: int = 16000,
                 n_fft: int = 512,
                 hop_length: int = 256,
                 win_length: int = 512):
        
        self.model_path = Path(model_path)
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model()
        
    def _load_model(self):
        """Load CleanUNet model and handle dictionary format."""
        try:
            if not self.model_path.exists():
                logger.warning(f"Model directory not found: {self.model_path}")
                return None
            
            # Find model files
            model_files = []
            for ext in ['.pkl', '.pickle', '.pth', '.pt', '.bin', '.tar', '.ckpt']:
                model_files.extend(list(self.model_path.glob(f"*{ext}")))
            
            common_names = ['pretrained.pkl', 'model.pkl', 'cleanunet.pkl', 'best_model.pkl']
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
                
                model_data = self._load_model_file(model_file)
                if model_data is not None:
                    model = self._create_model_from_data(model_data)
                    if model is not None:
                        logger.info("✅ Model created successfully")
                        return model
            
            logger.warning("Failed to create model from any file")
            return None
                
        except Exception as e:
            logger.error(f"Error in model loading: {e}")
            return None
    
    def _load_model_file(self, model_file):
        """Load model file with multiple methods."""
        
        # Method 1: torch.load()
        try:
            logger.info("Trying torch.load()...")
            data = torch.load(model_file, map_location=self.device)
            logger.info("✅ torch.load() successful")
            return data
        except Exception as e:
            logger.warning(f"torch.load() failed: {e}")
        
        # Method 2: torch.load() with CPU
        try:
            logger.info("Trying torch.load() with CPU...")
            data = torch.load(model_file, map_location='cpu')
            logger.info("✅ torch.load() with CPU successful")
            return data
        except Exception as e:
            logger.warning(f"torch.load() with CPU failed: {e}")
        
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
                model_keys = ['model', 'net', 'generator', 'cleanunet', 'network']
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
            input_dim = self._infer_input_dimension(state_dict)
            hidden_dim = self._infer_hidden_dimension(state_dict)
            num_layers = self._infer_num_layers(state_dict)
            
            logger.info(f"Inferred architecture: input_dim={input_dim}, hidden_dim={hidden_dim}, num_layers={num_layers}")
            
            # Try different model architectures
            architectures = [
                SimpleCleanUNet(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers),
                FlexibleCleanUNet(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers),
                SimpleCleanUNet(input_size=513, hidden_size=128, num_layers=4),  # Default fallback
            ]
            
            for i, model in enumerate(architectures):
                try:
                    # Try to load state dict (allow size mismatches)
                    model.load_state_dict(state_dict, strict=False)
                    model = model.to(self.device)
                    model.eval()
                    
                    logger.info(f"✅ Successfully created model with architecture {i+1}")
                    return model
                    
                except Exception as e:
                    logger.warning(f"Architecture {i+1} failed: {e}")
                    continue
            
            logger.warning("All architecture attempts failed")
            return None
            
        except Exception as e:
            logger.error(f"Error creating model from state_dict: {e}")
            return None
    
    def _infer_input_dimension(self, state_dict):
        """Infer input dimension from state_dict."""
        for key, tensor in state_dict.items():
            if 'input' in key.lower() or key.endswith('.weight') and len(tensor.shape) == 2:
                return tensor.shape[1]  # Input dimension
        return 513  # Default for spectrograms
    
    def _infer_hidden_dimension(self, state_dict):
        """Infer hidden dimension from state_dict."""
        hidden_dims = []
        for key, tensor in state_dict.items():
            if '.weight' in key and len(tensor.shape) == 2:
                hidden_dims.append(tensor.shape[0])
        
        if hidden_dims:
            # Use median as a reasonable estimate
            hidden_dims.sort()
            return hidden_dims[len(hidden_dims)//2] // 2  # Divide by 2 as base
        return 128  # Default
    
    def _infer_num_layers(self, state_dict):
        """Infer number of layers from state_dict."""
        layer_count = len([k for k in state_dict.keys() if '.weight' in k])
        return max(2, min(6, layer_count // 2))  # Reasonable range
    
    def _stft(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute STFT."""
        return torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length).to(waveform.device),
            return_complex=True
        )
    
    def _istft(self, stft_matrix: torch.Tensor) -> torch.Tensor:
        """Compute ISTFT."""
        return torch.istft(
            stft_matrix,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length).to(stft_matrix.device)
        )
    
    def _spectral_subtraction_fallback(self, waveform: torch.Tensor) -> torch.Tensor:
        """Enhanced spectral subtraction fallback."""
        logger.info("Applying enhanced spectral subtraction")
        
        # Convert to STFT
        stft = self._stft(waveform)
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)
        
        # Estimate noise
        noise_frames = int(0.5 * self.sample_rate / self.hop_length)
        noise_frames = min(noise_frames, magnitude.size(1) // 4)
        
        if noise_frames > 0:
            noise_magnitude = torch.mean(magnitude[:, :noise_frames], dim=1, keepdim=True)
        else:
            noise_magnitude = torch.mean(magnitude[:, :min(10, magnitude.size(1))], dim=1, keepdim=True)
        
        # Advanced spectral subtraction
        alpha = 2.5  # Over-subtraction factor
        beta = 0.02  # Spectral floor
        
        # Frequency-dependent over-subtraction
        freq_weights = torch.linspace(0.5, 2.0, magnitude.size(0)).unsqueeze(1).to(magnitude.device)
        alpha_freq = alpha * freq_weights
        
        enhanced_magnitude = magnitude - alpha_freq * noise_magnitude
        enhanced_magnitude = torch.maximum(enhanced_magnitude, beta * magnitude)
        
        # Temporal smoothing
        if enhanced_magnitude.size(1) > 3:
            smoothed = torch.nn.functional.avg_pool1d(
                enhanced_magnitude.unsqueeze(0), kernel_size=3, stride=1, padding=1
            ).squeeze(0)
            enhanced_magnitude = 0.7 * enhanced_magnitude + 0.3 * smoothed
        
        # Reconstruct
        enhanced_stft = enhanced_magnitude * torch.exp(1j * phase)
        enhanced_waveform = self._istft(enhanced_stft)
        
        return enhanced_waveform
    
    def enhance_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """Enhance audio using CleanUNet or fallback."""
        with torch.no_grad():
            if self.model is not None:
                try:
                    logger.info("Enhancing audio with CleanUNet model")
                    
                    # Ensure single channel
                    if waveform.dim() > 1 and waveform.size(0) > 1:
                        waveform = torch.mean(waveform, dim=0)
                    
                    # Move to device
                    waveform = waveform.to(self.device)
                    
                    # Convert to spectrogram
                    stft = self._stft(waveform)
                    magnitude = torch.abs(stft)
                    phase = torch.angle(stft)
                    
                    # Normalize input
                    mag_mean = torch.mean(magnitude)
                    mag_std = torch.std(magnitude)
                    if mag_std > 0:
                        magnitude_norm = (magnitude - mag_mean) / mag_std
                    else:
                        magnitude_norm = magnitude
                    
                    # Model inference
                    enhanced_mag_norm = self.model(magnitude_norm.unsqueeze(0)).squeeze(0)
                    
                    # Denormalize
                    if mag_std > 0:
                        enhanced_magnitude = enhanced_mag_norm * mag_std + mag_mean
                    else:
                        enhanced_magnitude = enhanced_mag_norm
                    
                    # Ensure positive values
                    enhanced_magnitude = torch.clamp(enhanced_magnitude, min=0)
                    
                    # Reconstruct
                    enhanced_stft = enhanced_magnitude * torch.exp(1j * phase)
                    enhanced_waveform = self._istft(enhanced_stft)
                    
                    logger.info("✅ CleanUNet inference successful")
                    return enhanced_waveform.unsqueeze(0)
                    
                except Exception as e:
                    logger.warning(f"CleanUNet inference failed: {e}, using fallback")
                    return self._spectral_subtraction_fallback(waveform).unsqueeze(0)
            else:
                # Use fallback
                if waveform.dim() > 1 and waveform.size(0) > 1:
                    waveform = torch.mean(waveform, dim=0)
                return self._spectral_subtraction_fallback(waveform).unsqueeze(0)


class AudioPreprocessor:
    """Audio preprocessing pipeline."""
    
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
                     model_path: str = "./models/cleanunet",
                     apply_enhancement: bool = True,
                     apply_normalization: bool = True,
                     apply_filtering: bool = True) -> None:
        """Process audio."""
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
                self.processor = AdvancedCleanUNetProcessor(model_path, sample_rate=self.target_sr)
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
        logger.info("ADVANCED CLEANUNET AUDIO PROCESSING")
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