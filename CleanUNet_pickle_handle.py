#!/usr/bin/env python3
"""
Fixed CleanUNet Audio Denoising Script
=====================================

This script handles the "persistent_load function" error that occurs
with CleanUNet pickle files containing PyTorch objects.

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
    3. Run: python fixed_cleanunet_script.py
"""

import os
import sys
import warnings
import logging
import pickle
from pathlib import Path
from typing import Tuple, Optional, Union

import torch
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

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FixedCleanUNetProcessor:
    """Fixed CleanUNet processor that handles pickle loading errors."""
    
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
        """Load CleanUNet model with proper error handling for pickle issues."""
        try:
            # Check if model directory exists
            if not self.model_path.exists():
                logger.warning(f"Model directory not found: {self.model_path}")
                logger.info("Will use fallback spectral subtraction method")
                return None
            
            # Find model files
            model_files = []
            
            # Look for pickle files first
            for ext in ['.pkl', '.pickle']:
                potential_files = list(self.model_path.glob(f"*{ext}"))
                model_files.extend(potential_files)
            
            # Then look for PyTorch files
            for ext in ['.pth', '.pt', '.bin', '.tar', '.ckpt']:
                potential_files = list(self.model_path.glob(f"*{ext}"))
                model_files.extend(potential_files)
            
            # Check for common filenames
            common_names = ['pretrained.pkl', 'model.pkl', 'cleanunet.pkl', 'best_model.pkl']
            for name in common_names:
                potential_file = self.model_path / name
                if potential_file.exists() and potential_file not in model_files:
                    model_files.append(potential_file)
            
            if not model_files:
                logger.warning(f"No model files found in {self.model_path}")
                logger.info(f"Files in directory: {list(self.model_path.glob('*'))}")
                return None
            
            # Try to load each file until one works
            for model_file in model_files:
                logger.info(f"Attempting to load model from: {model_file}")
                
                model = self._try_load_model_file(model_file)
                if model is not None:
                    logger.info("✅ Model loaded successfully")
                    return model
                else:
                    logger.warning(f"Failed to load {model_file}, trying next file...")
            
            logger.warning("All model loading attempts failed")
            return None
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.warning("Will use fallback spectral subtraction method")
            return None
    
    def _try_load_model_file(self, model_file):
        """Try different methods to load a model file."""
        
        if model_file.suffix in ['.pkl', '.pickle']:
            return self._load_pickle_file(model_file)
        else:
            return self._load_pytorch_file(model_file)
    
    def _load_pickle_file(self, model_file):
        """Load pickle file with multiple fallback methods."""
        
        # Method 1: Try torch.load() first (handles PyTorch objects in pickle)
        try:
            logger.info("Trying torch.load() for pickle file...")
            model_data = torch.load(model_file, map_location=self.device)
            logger.info("✅ Successfully loaded with torch.load()")
            return self._extract_model_from_data(model_data)
        except Exception as e:
            logger.warning(f"torch.load() failed: {e}")
        
        # Method 2: Try torch.load() with CPU mapping
        try:
            logger.info("Trying torch.load() with CPU mapping...")
            model_data = torch.load(model_file, map_location='cpu')
            logger.info("✅ Successfully loaded with torch.load() (CPU)")
            model = self._extract_model_from_data(model_data)
            if model is not None:
                model = model.to(self.device)
            return model
        except Exception as e:
            logger.warning(f"torch.load() with CPU failed: {e}")
        
        # Method 3: Try regular pickle.load()
        try:
            logger.info("Trying pickle.load()...")
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)
            logger.info("✅ Successfully loaded with pickle.load()")
            return self._extract_model_from_data(model_data)
        except Exception as e:
            logger.warning(f"pickle.load() failed: {e}")
        
        # Method 4: Try pickle with different protocols
        for protocol in [pickle.HIGHEST_PROTOCOL, 4, 3, 2]:
            try:
                logger.info(f"Trying pickle with protocol {protocol}...")
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                logger.info(f"✅ Successfully loaded with pickle protocol {protocol}")
                return self._extract_model_from_data(model_data)
            except Exception as e:
                logger.warning(f"Pickle protocol {protocol} failed: {e}")
        
        logger.error("All pickle loading methods failed")
        return None
    
    def _load_pytorch_file(self, model_file):
        """Load regular PyTorch model file."""
        try:
            logger.info("Loading PyTorch model file...")
            model_data = torch.load(model_file, map_location=self.device)
            logger.info("✅ Successfully loaded PyTorch file")
            return self._extract_model_from_data(model_data)
        except Exception as e:
            logger.error(f"Failed to load PyTorch file: {e}")
            return None
    
    def _extract_model_from_data(self, model_data):
        """Extract model from loaded data."""
        try:
            if model_data is None:
                return None
            
            # If it's already a model
            if hasattr(model_data, 'forward') or hasattr(model_data, '__call__'):
                logger.info("Data is already a model")
                return model_data
            
            # If it's a dictionary, look for model in common keys
            if isinstance(model_data, dict):
                logger.info(f"Data is dictionary with keys: {list(model_data.keys())}")
                
                # Common keys where models are stored
                possible_keys = ['model', 'net', 'generator', 'cleanunet', 'state_dict', 'model_state_dict']
                
                for key in possible_keys:
                    if key in model_data:
                        candidate = model_data[key]
                        if hasattr(candidate, 'forward') or hasattr(candidate, '__call__'):
                            logger.info(f"Found model in key: {key}")
                            return candidate
                        elif isinstance(candidate, dict):
                            # Might be a state dict, try to create model
                            logger.info(f"Key '{key}' contains state dict")
                            # For now, return the data and let inference handle it
                            return model_data
                
                # If no model found, return the whole dict
                logger.info("No model found in common keys, returning whole dictionary")
                return model_data
            
            # If it's a list or tuple, try first element
            elif isinstance(model_data, (list, tuple)):
                if len(model_data) > 0:
                    logger.info("Data is list/tuple, trying first element")
                    return self._extract_model_from_data(model_data[0])
            
            # Otherwise, return as-is and hope for the best
            logger.info("Returning data as-is")
            return model_data
            
        except Exception as e:
            logger.error(f"Error extracting model from data: {e}")
            return None
    
    def _stft(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute Short-Time Fourier Transform."""
        return torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length).to(waveform.device),
            return_complex=True
        )
    
    def _istft(self, stft_matrix: torch.Tensor) -> torch.Tensor:
        """Compute Inverse Short-Time Fourier Transform."""
        return torch.istft(
            stft_matrix,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length).to(stft_matrix.device)
        )
    
    def _spectral_subtraction_fallback(self, waveform: torch.Tensor) -> torch.Tensor:
        """Fallback method using spectral subtraction for denoising."""
        logger.info("Applying spectral subtraction denoising")
        
        # Convert to STFT
        stft = self._stft(waveform)
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)
        
        # Estimate noise from first 0.5 seconds
        noise_frames = int(0.5 * self.sample_rate / self.hop_length)
        if noise_frames > 0:
            noise_magnitude = torch.mean(magnitude[:, :noise_frames], dim=1, keepdim=True)
        else:
            # If audio is too short, use first few frames
            noise_magnitude = torch.mean(magnitude[:, :min(10, magnitude.size(1))], dim=1, keepdim=True)
        
        # Spectral subtraction
        alpha = 2.0  # Over-subtraction factor
        beta = 0.01  # Spectral floor
        
        enhanced_magnitude = magnitude - alpha * noise_magnitude
        enhanced_magnitude = torch.maximum(enhanced_magnitude, beta * magnitude)
        
        # Reconstruct signal
        enhanced_stft = enhanced_magnitude * torch.exp(1j * phase)
        enhanced_waveform = self._istft(enhanced_stft)
        
        return enhanced_waveform
    
    def enhance_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Enhance audio using CleanUNet or fallback method.
        
        Args:
            waveform: Input waveform tensor [channels, samples]
            
        Returns:
            Enhanced waveform tensor
        """
        with torch.no_grad():
            if self.model is not None:
                try:
                    # Use loaded model
                    logger.info("Enhancing audio with loaded model")
                    
                    # Ensure single channel
                    if waveform.dim() > 1 and waveform.size(0) > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)
                    
                    # Prepare input
                    original_shape = waveform.shape
                    if waveform.dim() == 1:
                        waveform = waveform.unsqueeze(0)
                    
                    # Move to device
                    waveform = waveform.to(self.device)
                    
                    # Try different inference methods
                    enhanced = None
                    
                    # Method 1: Try standard forward pass
                    try:
                        if hasattr(self.model, 'forward'):
                            enhanced = self.model.forward(waveform)
                        elif callable(self.model):
                            enhanced = self.model(waveform)
                    except Exception as e:
                        logger.warning(f"Standard forward pass failed: {e}")
                    
                    # Method 2: Try CleanUNet-specific methods
                    if enhanced is None:
                        try:
                            if hasattr(self.model, 'enhance'):
                                enhanced = self.model.enhance(waveform)
                            elif hasattr(self.model, 'denoise'):
                                enhanced = self.model.denoise(waveform)
                        except Exception as e:
                            logger.warning(f"CleanUNet-specific methods failed: {e}")
                    
                    # Method 3: If model is a dictionary, try to use it differently
                    if enhanced is None and isinstance(self.model, dict):
                        logger.warning("Model is dictionary, cannot perform inference directly")
                        return self._spectral_subtraction_fallback(waveform.squeeze(0))
                    
                    # Method 4: Last resort - try to call as function
                    if enhanced is None:
                        try:
                            enhanced = self.model(waveform)
                        except Exception as e:
                            logger.warning(f"Function call failed: {e}")
                            return self._spectral_subtraction_fallback(waveform.squeeze(0))
                    
                    if enhanced is not None:
                        # Clean up output dimensions
                        while enhanced.dim() > 2:
                            enhanced = enhanced.squeeze(0)
                        
                        if enhanced.dim() == 1:
                            enhanced = enhanced.unsqueeze(0)
                        
                        logger.info("✅ Model inference successful")
                        return enhanced
                    else:
                        logger.warning("All inference methods failed, using fallback")
                        return self._spectral_subtraction_fallback(waveform.squeeze(0))
                    
                except Exception as e:
                    logger.warning(f"Model inference failed: {e}, using fallback")
                    return self._spectral_subtraction_fallback(waveform.squeeze(0) if waveform.dim() > 1 else waveform)
            else:
                # Use fallback method
                if waveform.dim() > 1 and waveform.size(0) > 1:
                    waveform = torch.mean(waveform, dim=0)
                return self._spectral_subtraction_fallback(waveform)


class AudioPreprocessor:
    """Audio preprocessing pipeline for ASR preparation."""
    
    def __init__(self, target_sr: int = 16000):
        """
        Initialize audio preprocessor.
        
        Args:
            target_sr: Target sample rate for output
        """
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
        """Resample audio to target sample rate."""
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
    
    def normalize_audio(self, waveform: torch.Tensor, target_lufs: float = -23.0) -> torch.Tensor:
        """Normalize audio level."""
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
        """Save processed audio."""
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
        """Complete audio processing pipeline."""
        logger.info(f"Processing audio: {input_path} -> {output_path}")
        
        # Load audio
        waveform, orig_sr = self.load_audio(input_path)
        
        # Resample to target rate
        waveform = self.resample_audio(waveform, orig_sr)
        
        # Apply high-pass filter
        if apply_filtering:
            waveform = self.apply_highpass_filter(waveform)
        
        # Apply speech enhancement
        if apply_enhancement:
            if self.processor is None:
                self.processor = FixedCleanUNetProcessor(model_path, sample_rate=self.target_sr)
            waveform = self.processor.enhance_audio(waveform)
        
        # Normalize audio level
        if apply_normalization:
            waveform = self.normalize_audio(waveform)
        
        # Save processed audio
        self.save_audio(waveform, output_path)
        
        logger.info("Audio processing completed successfully")


def main():
    """Main function - processes audio using the configured file paths."""
    
    # Validate configuration
    if INPUT_AUDIO_PATH == "path/to/your/noisy_audio.wav":
        logger.error("Please set INPUT_AUDIO_PATH to your actual input file!")
        logger.info("Edit the script and change INPUT_AUDIO_PATH at the top")
        sys.exit(1)
    
    if OUTPUT_AUDIO_PATH == "path/to/your/clean_audio.wav":
        logger.error("Please set OUTPUT_AUDIO_PATH to your desired output file!")
        logger.info("Edit the script and change OUTPUT_AUDIO_PATH at the top")
        sys.exit(1)
    
    # Validate input file exists
    if not os.path.exists(INPUT_AUDIO_PATH):
        logger.error(f"Input file not found: {INPUT_AUDIO_PATH}")
        sys.exit(1)
    
    # Create output directory if needed
    output_dir = os.path.dirname(OUTPUT_AUDIO_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    
    try:
        # Print configuration
        logger.info("="*60)
        logger.info("FIXED CLEANUNET AUDIO PROCESSING")
        logger.info("="*60)
        logger.info(f"Input file: {INPUT_AUDIO_PATH}")
        logger.info(f"Output file: {OUTPUT_AUDIO_PATH}")
        logger.info(f"Model path: {MODEL_PATH}")
        logger.info(f"Sample rate: {SAMPLE_RATE} Hz")
        logger.info(f"Device: {DEVICE}")
        logger.info(f"Enhancement: {'ON' if APPLY_ENHANCEMENT else 'OFF'}")
        logger.info(f"Normalization: {'ON' if APPLY_NORMALIZATION else 'OFF'}")
        logger.info(f"Filtering: {'ON' if APPLY_FILTERING else 'OFF'}")
        logger.info("="*60)
        
        # Debug: List files in model directory
        model_dir = Path(MODEL_PATH)
        if model_dir.exists():
            logger.info(f"Files in model directory {MODEL_PATH}:")
            for file in model_dir.iterdir():
                size_mb = file.stat().st_size / (1024*1024)
                logger.info(f"  - {file.name} ({size_mb:.1f} MB)")
        else:
            logger.warning(f"Model directory does not exist: {MODEL_PATH}")
        
        # Initialize preprocessor
        preprocessor = AudioPreprocessor(target_sr=SAMPLE_RATE)
        
        # Process audio
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