#!/usr/bin/env python3
"""
CleanUNet Audio Denoising and Speech Enhancement Script
======================================================

This script uses CleanUNet for offline speech enhancement and denoising
of noisy call audio recordings. The output is clean audio ready for ASR.

Requirements:
- torch
- torchaudio
- librosa
- soundfile
- numpy
- scipy

Usage:
    1. Set the INPUT_AUDIO_PATH and OUTPUT_AUDIO_PATH variables below
    2. Run: python cleanunet_denoiser.py
"""

import os
import sys
import warnings
import logging
from pathlib import Path
from typing import Tuple, Optional, Union
import math

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

# Model configuration
MODEL_PATH = "./models/cleanunet"
SAMPLE_RATE = 16000
DEVICE = "auto"  # "auto", "cpu", or "cuda"

# Processing options (True to enable, False to disable)
APPLY_ENHANCEMENT = True      # Use CleanUNet for speech enhancement
APPLY_NORMALIZATION = True    # Normalize audio levels
APPLY_FILTERING = True        # Apply high-pass filter

# CleanUNet-specific parameters
DIFFUSION_STEPS = 8          # Number of diffusion steps (higher = better quality, slower)
CHUNK_SIZE = 8192            # Process audio in chunks (samples)
OVERLAP = 1024               # Overlap between chunks

# ========================================================

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CleanUNetProcessor:
    """CleanUNet processor for speech enhancement and denoising."""
    
    def __init__(self, 
                 model_path: str = "./models/cleanunet",
                 device: str = "auto",
                 sample_rate: int = 16000,
                 diffusion_steps: int = 8,
                 chunk_size: int = 8192,
                 overlap: int = 1024):
        """
        Initialize CleanUNet processor.
        
        Args:
            model_path: Path to the CleanUNet model directory
            device: Device to run inference on ('auto', 'cpu', 'cuda')
            sample_rate: Target sample rate for processing
            diffusion_steps: Number of diffusion denoising steps
            chunk_size: Size of audio chunks for processing
            overlap: Overlap between chunks for smooth reconstruction
        """
        self.model_path = Path(model_path)
        self.sample_rate = sample_rate
        self.diffusion_steps = diffusion_steps
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model()
        
    def _load_model(self):
        """Load CleanUNet model from local directory."""
        try:
            # Check if model directory exists
            if not self.model_path.exists():
                logger.warning(f"Model directory not found: {self.model_path}")
                logger.info("Will use fallback wiener filtering method")
                return None
            
            # Look for model files
            model_file = None
            for ext in ['.pth', '.pt', '.bin', '.tar']:
                potential_files = list(self.model_path.glob(f"*{ext}"))
                if potential_files:
                    model_file = potential_files[0]
                    break
            
            if model_file is None:
                logger.warning(f"No model file found in {self.model_path}")
                logger.info("Will use fallback wiener filtering method")
                return None
            
            logger.info(f"Loading model from: {model_file}")
            
            # Load model checkpoint
            checkpoint = torch.load(model_file, map_location=self.device)
            
            # Create model instance (you may need to adjust this based on your model structure)
            try:
                from cleanunet.model import CleanUNet
                
                model = CleanUNet(
                    channels=[32, 64, 128, 256, 512],
                    factors=[4, 4, 4, 2, 2],
                    items=[4, 4, 4, 4, 4]
                )
            except ImportError:
                logger.warning("Could not import CleanUNet model class")
                logger.info("Will use fallback wiener filtering method")
                return None
            
            # Load state dict
            try:
                if 'model' in checkpoint:
                    model.load_state_dict(checkpoint['model'])
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            except Exception as e:
                logger.warning(f"Could not load model state dict: {e}")
                return None
            
            model.to(self.device)
            model.eval()
            
            logger.info("CleanUNet model loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.warning("Will use fallback wiener filtering method")
            return None
    
    def _apply_window(self, waveform: torch.Tensor, window_type: str = "hann") -> torch.Tensor:
        """Apply window function to reduce boundary artifacts."""
        if window_type == "hann":
            window = torch.hann_window(waveform.size(-1)).to(waveform.device)
        else:
            window = torch.ones(waveform.size(-1)).to(waveform.device)
        
        return waveform * window
    
    def _wiener_filter_fallback(self, waveform: torch.Tensor) -> torch.Tensor:
        """Fallback method using Wiener filtering for denoising."""
        logger.info("Applying Wiener filter denoising")
        
        # Convert to frequency domain
        fft = torch.fft.rfft(waveform, dim=-1)
        magnitude = torch.abs(fft)
        phase = torch.angle(fft)
        
        # Estimate noise from first 0.5 seconds
        noise_frames = int(0.5 * self.sample_rate)
        if noise_frames > waveform.size(-1):
            noise_frames = waveform.size(-1) // 4
        
        noise_fft = torch.fft.rfft(waveform[..., :noise_frames], dim=-1)
        noise_power = torch.mean(torch.abs(noise_fft)**2, dim=-1, keepdim=True)
        
        # Signal power estimation
        signal_power = magnitude**2
        
        # Wiener filter
        wiener_gain = signal_power / (signal_power + noise_power + 1e-10)
        
        # Apply filter
        enhanced_fft = fft * wiener_gain
        enhanced_waveform = torch.fft.irfft(enhanced_fft, n=waveform.size(-1), dim=-1)
        
        return enhanced_waveform
    
    def _chunk_audio(self, waveform: torch.Tensor) -> list:
        """Split audio into overlapping chunks for processing."""
        chunks = []
        step = self.chunk_size - self.overlap
        
        for i in range(0, waveform.size(-1), step):
            end = min(i + self.chunk_size, waveform.size(-1))
            chunk = waveform[..., i:end]
            
            # Pad if necessary
            if chunk.size(-1) < self.chunk_size:
                padding = self.chunk_size - chunk.size(-1)
                chunk = torch.nn.functional.pad(chunk, (0, padding))
            
            chunks.append((chunk, i, end))
        
        return chunks
    
    def _reconstruct_from_chunks(self, chunks: list, original_length: int) -> torch.Tensor:
        """Reconstruct audio from overlapping chunks."""
        reconstructed = torch.zeros(chunks[0][0].shape[:-1] + (original_length,), 
                                  device=chunks[0][0].device)
        weights = torch.zeros_like(reconstructed)
        
        step = self.chunk_size - self.overlap
        
        for chunk, start_idx, end_idx in chunks:
            actual_end = min(start_idx + chunk.size(-1), original_length)
            actual_chunk_size = actual_end - start_idx
            
            # Apply window to reduce boundary artifacts
            if actual_chunk_size == self.chunk_size:
                window = torch.hann_window(self.overlap).to(chunk.device)
                # Fade in at the beginning
                if start_idx > 0:
                    chunk[..., :self.overlap//2] *= window[:self.overlap//2]
                # Fade out at the end
                if actual_end < original_length:
                    chunk[..., -self.overlap//2:] *= window[self.overlap//2:]
            
            reconstructed[..., start_idx:actual_end] += chunk[..., :actual_chunk_size]
            weights[..., start_idx:actual_end] += 1.0
        
        # Normalize by overlap count
        weights = torch.clamp(weights, min=1.0)
        reconstructed = reconstructed / weights
        
        return reconstructed
    
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
                    # Use CleanUNet model
                    logger.info(f"Enhancing audio with CleanUNet ({self.diffusion_steps} steps)")
                    
                    # Ensure single channel
                    if waveform.dim() > 1 and waveform.size(0) > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)
                    
                    # Move to device
                    waveform = waveform.to(self.device)
                    original_length = waveform.size(-1)
                    
                    # Process in chunks for memory efficiency
                    if original_length > self.chunk_size:
                        chunks = self._chunk_audio(waveform)
                        enhanced_chunks = []
                        
                        for i, (chunk, start_idx, end_idx) in enumerate(chunks):
                            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                            
                            # Add batch dimension if needed
                            if chunk.dim() == 2:
                                chunk = chunk.unsqueeze(0)
                            
                            # Run CleanUNet inference
                            enhanced_chunk = self.model.enhance(
                                chunk, 
                                num_steps=self.diffusion_steps
                            )
                            
                            if enhanced_chunk.dim() == 3:
                                enhanced_chunk = enhanced_chunk.squeeze(0)
                            
                            enhanced_chunks.append((enhanced_chunk, start_idx, end_idx))
                        
                        # Reconstruct full audio
                        enhanced = self._reconstruct_from_chunks(enhanced_chunks, original_length)
                    else:
                        # Process entire audio at once
                        if waveform.dim() == 2:
                            waveform = waveform.unsqueeze(0)
                        
                        enhanced = self.model.enhance(
                            waveform,
                            num_steps=self.diffusion_steps
                        )
                        
                        if enhanced.dim() == 3:
                            enhanced = enhanced.squeeze(0)
                    
                    return enhanced
                    
                except Exception as e:
                    logger.warning(f"CleanUNet inference failed: {e}, using fallback")
                    if waveform.dim() > 1 and waveform.size(0) > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)
                    return self._wiener_filter_fallback(waveform)
            else:
                # Use fallback method
                if waveform.dim() > 1 and waveform.size(0) > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                return self._wiener_filter_fallback(waveform)


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
        """
        Load audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (waveform, sample_rate)
        """
        try:
            # Try with torchaudio first
            waveform, sr = torchaudio.load(file_path)
            logger.info(f"Loaded audio: {waveform.shape}, SR: {sr}")
            return waveform, sr
            
        except Exception as e:
            logger.warning(f"torchaudio failed: {e}, trying librosa")
            try:
                # Fallback to librosa
                waveform, sr = librosa.load(file_path, sr=None, mono=False)
                if waveform.ndim == 1:
                    waveform = waveform[None, :]  # Add channel dimension
                waveform = torch.from_numpy(waveform)
                return waveform, sr
            except Exception as e2:
                raise RuntimeError(f"Failed to load audio with both torchaudio and librosa: {e2}")
    
    def resample_audio(self, waveform: torch.Tensor, orig_sr: int) -> torch.Tensor:
        """
        Resample audio to target sample rate.
        
        Args:
            waveform: Input waveform
            orig_sr: Original sample rate
            
        Returns:
            Resampled waveform
        """
        if orig_sr == self.target_sr:
            return waveform
        
        logger.info(f"Resampling from {orig_sr} Hz to {self.target_sr} Hz")
        
        try:
            # Use torchaudio resampler
            resampler = torchaudio.transforms.Resample(orig_sr, self.target_sr)
            return resampler(waveform)
        except:
            # Fallback to scipy
            logger.warning("Using scipy for resampling")
            waveform_np = waveform.numpy()
            num_samples = int(waveform_np.shape[-1] * self.target_sr / orig_sr)
            resampled = resample(waveform_np, num_samples, axis=-1)
            return torch.from_numpy(resampled)
    
    def normalize_audio(self, waveform: torch.Tensor, target_lufs: float = -23.0) -> torch.Tensor:
        """
        Normalize audio level.
        
        Args:
            waveform: Input waveform
            target_lufs: Target loudness in LUFS
            
        Returns:
            Normalized waveform
        """
        # Simple RMS normalization (for more accurate LUFS, use pyloudnorm)
        rms = torch.sqrt(torch.mean(waveform**2))
        if rms > 0:
            target_rms = 0.1  # Approximate target RMS
            scaling_factor = target_rms / rms
            waveform = waveform * scaling_factor
        
        # Prevent clipping
        max_val = torch.max(torch.abs(waveform))
        if max_val > 0.95:
            waveform = waveform * (0.95 / max_val)
        
        return waveform
    
    def apply_highpass_filter(self, waveform: torch.Tensor, cutoff: float = 80.0) -> torch.Tensor:
        """
        Apply high-pass filter to remove low-frequency noise.
        
        Args:
            waveform: Input waveform
            cutoff: Cutoff frequency in Hz
            
        Returns:
            Filtered waveform
        """
        try:
            # Use torchaudio's high-pass filter
            highpass = torchaudio.transforms.Highpass(self.target_sr, cutoff)
            return highpass(waveform)
        except:
            # Simple high-pass using difference
            logger.warning("Using simple high-pass filter")
            alpha = cutoff / (cutoff + self.target_sr)
            filtered = torch.zeros_like(waveform)
            filtered[:, 1:] = alpha * (waveform[:, 1:] - waveform[:, :-1])
            return filtered
    
    def save_audio(self, waveform: torch.Tensor, file_path: str):
        """
        Save processed audio.
        
        Args:
            waveform: Processed waveform
            file_path: Output file path
        """
        # Ensure waveform is in correct format for saving
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Convert to numpy and transpose for soundfile
        waveform_np = waveform.cpu().numpy()
        if waveform_np.shape[0] == 1:
            waveform_np = waveform_np.squeeze(0)  # Remove channel dimension for mono
        
        # Save using soundfile
        sf.write(file_path, waveform_np, self.target_sr, subtype='PCM_16')
        logger.info(f"Saved enhanced audio to: {file_path}")
    
    def process_audio(self, 
                     input_path: str, 
                     output_path: str,
                     model_path: str = "./models/cleanunet",
                     diffusion_steps: int = 8,
                     chunk_size: int = 8192,
                     overlap: int = 1024,
                     apply_enhancement: bool = True,
                     apply_normalization: bool = True,
                     apply_filtering: bool = True) -> None:
        """
        Complete audio processing pipeline.
        
        Args:
            input_path: Input audio file path
            output_path: Output audio file path
            model_path: Path to CleanUNet model
            diffusion_steps: Number of diffusion denoising steps
            chunk_size: Size of audio chunks for processing
            overlap: Overlap between chunks
            apply_enhancement: Whether to apply speech enhancement
            apply_normalization: Whether to normalize audio level
            apply_filtering: Whether to apply high-pass filtering
        """
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
                self.processor = CleanUNetProcessor(
                    model_path=model_path,
                    sample_rate=self.target_sr,
                    diffusion_steps=diffusion_steps,
                    chunk_size=chunk_size,
                    overlap=overlap
                )
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
        logger.info("CLEANUNET AUDIO PROCESSING")
        logger.info("="*60)
        logger.info(f"Input file: {INPUT_AUDIO_PATH}")
        logger.info(f"Output file: {OUTPUT_AUDIO_PATH}")
        logger.info(f"Model path: {MODEL_PATH}")
        logger.info(f"Sample rate: {SAMPLE_RATE} Hz")
        logger.info(f"Device: {DEVICE}")
        logger.info(f"Diffusion steps: {DIFFUSION_STEPS}")
        logger.info(f"Chunk size: {CHUNK_SIZE}")
        logger.info(f"Overlap: {OVERLAP}")
        logger.info(f"Enhancement: {'ON' if APPLY_ENHANCEMENT else 'OFF'}")
        logger.info(f"Normalization: {'ON' if APPLY_NORMALIZATION else 'OFF'}")
        logger.info(f"Filtering: {'ON' if APPLY_FILTERING else 'OFF'}")
        logger.info("="*60)
        
        # Initialize preprocessor
        preprocessor = AudioPreprocessor(target_sr=SAMPLE_RATE)
        
        # Process audio
        preprocessor.process_audio(
            input_path=INPUT_AUDIO_PATH,
            output_path=OUTPUT_AUDIO_PATH,
            model_path=MODEL_PATH,
            diffusion_steps=DIFFUSION_STEPS,
            chunk_size=CHUNK_SIZE,
            overlap=OVERLAP,
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
        sys.exit(1)


if __name__ == "__main__":
    main()
