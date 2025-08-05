#!/usr/bin/env python3
"""
Facebook Denoiser Audio Enhancement Script
==========================================

This script uses Facebook's Denoiser (DNS/Demucs) for offline speech enhancement 
and denoising of noisy call audio recordings. The output is clean audio ready for ASR.

Requirements:
- torch
- torchaudio
- librosa
- soundfile
- numpy
- scipy
- denoiser (pip install denoiser)

Usage:
    1. Set the INPUT_AUDIO_PATH and OUTPUT_AUDIO_PATH variables below
    2. Run: python facebook_denoiser.py
"""

import os
import sys
import warnings
import logging
from pathlib import Path
from typing import Tuple, Optional, Union
import tempfile

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
MODEL_PATH = "./models/facebook_denoiser"
MODEL_NAME = "dns64"  # Options: "dns48", "dns64", "master64"
SAMPLE_RATE = 16000
DEVICE = "auto"  # "auto", "cpu", or "cuda"

# Processing options (True to enable, False to disable)
APPLY_ENHANCEMENT = True      # Use Facebook Denoiser for speech enhancement
APPLY_NORMALIZATION = True    # Normalize audio levels
APPLY_FILTERING = True        # Apply high-pass filter

# Facebook Denoiser-specific parameters
STREAMING = False            # Use streaming mode for real-time processing
SEGMENT_LENGTH = 10.0        # Length of segments in seconds for processing
USE_VAD = True              # Use Voice Activity Detection for better results

# ========================================================

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FacebookDenoiserProcessor:
    """Facebook Denoiser processor for speech enhancement and noise suppression."""
    
    def __init__(self, 
                 model_path: str = "./models/facebook_denoiser",
                 model_name: str = "dns64",
                 device: str = "auto",
                 sample_rate: int = 16000,
                 streaming: bool = False,
                 segment_length: float = 10.0,
                 use_vad: bool = True):
        """
        Initialize Facebook Denoiser processor.
        
        Args:
            model_path: Path to the Facebook Denoiser model directory
            model_name: Model variant ("dns48", "dns64", "master64")
            device: Device to run inference on ('auto', 'cpu', 'cuda')
            sample_rate: Target sample rate for processing
            streaming: Whether to use streaming mode
            segment_length: Length of audio segments for processing (seconds)
            use_vad: Whether to use Voice Activity Detection
        """
        self.model_path = Path(model_path)
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.streaming = streaming
        self.segment_length = segment_length
        self.use_vad = use_vad
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model()
        self.vad_model = self._load_vad_model() if use_vad else None
        
    def _load_model(self):
        """Load Facebook Denoiser model."""
        try:
            # Try to import denoiser package
            try:
                from denoiser import pretrained
                from denoiser.demucs import Demucs
                logger.info("Using official denoiser package")
                
                # Load pretrained model
                model = pretrained.get_model(self.model_name).to(self.device)
                model.eval()
                
                logger.info(f"Loaded Facebook Denoiser model: {self.model_name}")
                return model
                
            except ImportError:
                logger.warning("denoiser package not found, trying manual loading")
                
                # Check if model directory exists
                if not self.model_path.exists():
                    logger.warning(f"Model directory not found: {self.model_path}")
                    logger.info("Will use fallback RNNoise-style filtering")
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
                    logger.info("Will use fallback RNNoise-style filtering")
                    return None
                
                logger.info(f"Loading model from: {model_file}")
                
                # Load model checkpoint
                checkpoint = torch.load(model_file, map_location=self.device)
                
                # Create model instance manually
                try:
                    from denoiser.demucs import Demucs
                    
                    # Default Demucs configuration for DNS
                    model = Demucs(
                        chin=1,
                        chout=1,
                        hidden=64,
                        depth=5,
                        kernel_size=8,
                        stride=4,
                        causal=True,
                        resample=4,
                        growth=2,
                        max_hidden=10_000,
                        normalize=True,
                        glu=True,
                        rescale=0.1,
                        floor=1e-3
                    )
                except ImportError:
                    logger.warning("Could not import Demucs model class")
                    logger.info("Will use fallback RNNoise-style filtering")
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
                
                logger.info("Facebook Denoiser model loaded successfully")
                return model
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.warning("Will use fallback RNNoise-style filtering")
            return None
    
    def _load_vad_model(self):
        """Load Voice Activity Detection model."""
        try:
            # Try to load WebRTC VAD or similar
            import webrtcvad
            return webrtcvad.Vad(2)  # Aggressiveness level 2
        except ImportError:
            try:
                # Alternative: use silero VAD
                import torch
                model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                            model='silero_vad',
                                            force_reload=False)
                return model
            except:
                logger.warning("Could not load VAD model")
                return None
    
    def _apply_vad(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply Voice Activity Detection to focus processing on speech regions."""
        if self.vad_model is None:
            return waveform
        
        try:
            # Simple energy-based VAD as fallback
            frame_length = int(0.025 * self.sample_rate)  # 25ms frames
            hop_length = int(0.010 * self.sample_rate)    # 10ms hop
            
            # Calculate frame energy
            frames = waveform.unfold(-1, frame_length, hop_length)
            energy = torch.mean(frames**2, dim=-1)
            
            # Threshold for voice activity
            threshold = torch.quantile(energy, 0.3)
            voice_mask = energy > threshold
            
            # Expand mask to waveform
            expanded_mask = torch.repeat_interleave(voice_mask, hop_length, dim=-1)
            
            # Trim to match waveform length
            if expanded_mask.size(-1) > waveform.size(-1):
                expanded_mask = expanded_mask[..., :waveform.size(-1)]
            elif expanded_mask.size(-1) < waveform.size(-1):
                padding = waveform.size(-1) - expanded_mask.size(-1)
                expanded_mask = torch.nn.functional.pad(expanded_mask, (0, padding), value=False)
            
            return waveform * expanded_mask.float()
            
        except Exception as e:
            logger.warning(f"VAD processing failed: {e}")
            return waveform
    
    def _rnnoise_style_fallback(self, waveform: torch.Tensor) -> torch.Tensor:
        """Fallback method using RNNoise-style spectral gating."""
        logger.info("Applying RNNoise-style spectral gating")
        
        # Parameters for spectral gating
        frame_size = 480  # 30ms at 16kHz
        hop_size = 160    # 10ms at 16kHz
        
        # Convert to frequency domain
        frames = waveform.unfold(-1, frame_size, hop_size)
        windowed_frames = frames * torch.hann_window(frame_size).to(frames.device)
        
        fft_frames = torch.fft.rfft(windowed_frames, dim=-1)
        magnitude = torch.abs(fft_frames)
        phase = torch.angle(fft_frames)
        
        # Estimate noise floor (first 10 frames)
        noise_frames = magnitude[..., :10, :]
        noise_floor = torch.mean(noise_frames, dim=-2, keepdim=True)
        
        # Spectral gating
        snr_threshold = 3.0  # 3 dB threshold
        gate_ratio = 0.1     # Reduce noise by 90%
        
        snr = magnitude / (noise_floor + 1e-8)
        gain = torch.where(snr > snr_threshold, 
                          torch.ones_like(snr),
                          gate_ratio * torch.ones_like(snr))
        
        # Apply gain
        enhanced_magnitude = magnitude * gain
        enhanced_fft = enhanced_magnitude * torch.exp(1j * phase)
        
        # Convert back to time domain
        enhanced_frames = torch.fft.irfft(enhanced_fft, n=frame_size, dim=-1)
        enhanced_frames = enhanced_frames * torch.hann_window(frame_size).to(enhanced_frames.device)
        
        # Overlap-add reconstruction
        output_length = waveform.size(-1)
        enhanced_waveform = torch.zeros_like(waveform)
        
        for i, frame in enumerate(enhanced_frames.unbind(-2)):
            start = i * hop_size
            end = min(start + frame_size, output_length)
            enhanced_waveform[..., start:end] += frame[..., :end-start]
        
        return enhanced_waveform
    
    def _process_segments(self, waveform: torch.Tensor) -> torch.Tensor:
        """Process audio in segments for memory efficiency."""
        segment_samples = int(self.segment_length * self.sample_rate)
        
        if waveform.size(-1) <= segment_samples:
            return self.model(waveform.unsqueeze(0)).squeeze(0)
        
        # Process in overlapping segments
        overlap_samples = int(0.5 * self.sample_rate)  # 0.5 second overlap
        step_size = segment_samples - overlap_samples
        
        segments = []
        for i in range(0, waveform.size(-1), step_size):
            end = min(i + segment_samples, waveform.size(-1))
            segment = waveform[..., i:end]
            
            # Pad if necessary
            if segment.size(-1) < segment_samples:
                padding = segment_samples - segment.size(-1)
                segment = torch.nn.functional.pad(segment, (0, padding))
            
            # Process segment
            with torch.no_grad():
                enhanced_segment = self.model(segment.unsqueeze(0)).squeeze(0)
            
            # Apply fade-in/fade-out for smooth transitions
            if i > 0:  # Fade in
                fade_samples = overlap_samples // 2
                fade_in = torch.linspace(0, 1, fade_samples).to(enhanced_segment.device)
                enhanced_segment[..., :fade_samples] *= fade_in
            
            if end < waveform.size(-1):  # Fade out
                fade_samples = overlap_samples // 2
                fade_out = torch.linspace(1, 0, fade_samples).to(enhanced_segment.device)
                enhanced_segment[..., -fade_samples:] *= fade_out
            
            segments.append((enhanced_segment, i, end))
        
        # Reconstruct full audio
        enhanced_waveform = torch.zeros_like(waveform)
        for segment, start_idx, end_idx in segments:
            actual_length = min(end_idx - start_idx, segment.size(-1))
            enhanced_waveform[..., start_idx:start_idx + actual_length] += segment[..., :actual_length]
        
        return enhanced_waveform
    
    def enhance_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Enhance audio using Facebook Denoiser or fallback method.
        
        Args:
            waveform: Input waveform tensor [channels, samples]
            
        Returns:
            Enhanced waveform tensor
        """
        with torch.no_grad():
            if self.model is not None:
                try:
                    # Use Facebook Denoiser model
                    logger.info(f"Enhancing audio with Facebook Denoiser ({self.model_name})")
                    
                    # Ensure single channel
                    if waveform.dim() > 1 and waveform.size(0) > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)
                    
                    # Move to device
                    waveform = waveform.to(self.device)
                    
                    # Apply VAD preprocessing if enabled
                    if self.use_vad:
                        waveform = self._apply_vad(waveform)
                    
                    # Process audio
                    if self.streaming or waveform.size(-1) > int(self.segment_length * self.sample_rate):
                        enhanced = self._process_segments(waveform)
                    else:
                        # Process entire audio at once
                        enhanced = self.model(waveform.unsqueeze(0)).squeeze(0)
                    
                    return enhanced
                    
                except Exception as e:
                    logger.warning(f"Facebook Denoiser inference failed: {e}, using fallback")
                    if waveform.dim() > 1 and waveform.size(0) > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)
                    return self._rnnoise_style_fallback(waveform)
            else:
                # Use fallback method
                if waveform.dim() > 1 and waveform.size(0) > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                return self._rnnoise_style_fallback(waveform)


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
                     model_path: str = "./models/facebook_denoiser",
                     model_name: str = "dns64",
                     streaming: bool = False,
                     segment_length: float = 10.0,
                     use_vad: bool = True,
                     apply_enhancement: bool = True,
                     apply_normalization: bool = True,
                     apply_filtering: bool = True) -> None:
        """
        Complete audio processing pipeline.
        
        Args:
            input_path: Input audio file path
            output_path: Output audio file path
            model_path: Path to Facebook Denoiser model
            model_name: Model variant name
            streaming: Whether to use streaming mode
            segment_length: Length of segments for processing
            use_vad: Whether to use Voice Activity Detection
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
                self.processor = FacebookDenoiserProcessor(
                    model_path=model_path,
                    model_name=model_name,
                    sample_rate=self.target_sr,
                    streaming=streaming,
                    segment_length=segment_length,
                    use_vad=use_vad
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
        logger.info("FACEBOOK DENOISER AUDIO PROCESSING")
        logger.info("="*60)
        logger.info(f"Input file: {INPUT_AUDIO_PATH}")
        logger.info(f"Output file: {OUTPUT_AUDIO_PATH}")
        logger.info(f"Model path: {MODEL_PATH}")
        logger.info(f"Model name: {MODEL_NAME}")
        logger.info(f"Sample rate: {SAMPLE_RATE} Hz")
        logger.info(f"Device: {DEVICE}")
        logger.info(f"Streaming: {'ON' if STREAMING else 'OFF'}")
        logger.info(f"Segment length: {SEGMENT_LENGTH}s")
        logger.info(f"VAD: {'ON' if USE_VAD else 'OFF'}")
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
            model_name=MODEL_NAME,
            streaming=STREAMING,
            segment_length=SEGMENT_LENGTH,
            use_vad=USE_VAD,
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
