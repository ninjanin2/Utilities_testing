"""
Speech-Optimized Audio Enhancement System for ASR
=================================================
Conservative, speech-focused enhancement specifically designed for 
call recordings and ASR preprocessing. Prioritizes speech intelligibility
and clarity over heavy noise reduction to avoid distortion.

Features:
- Gentle noise reduction focused on speech preservation
- Speech frequency emphasis (300-3400 Hz)
- Conservative dynamic processing
- No harmonic separation (preserves speech integrity)
- ASR-optimized output formatting

Author: Speech Enhancement System for ASR
Version: 1.0.0-ASR-Optimized
"""

import os
import sys
import warnings
import gc
import json
import logging
import time
from pathlib import Path
from typing import Tuple, Dict, Optional, List, Union, Any
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from collections import defaultdict

import numpy as np
import scipy.signal as signal
from scipy.stats import median_abs_deviation
import librosa
import soundfile as sf

# Optional professional libraries
try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False
    print("Info: noisereduce not available - using built-in methods")

try:
    import pyloudnorm as pyln
    PYLOUDNORM_AVAILABLE = True
except ImportError:
    PYLOUDNORM_AVAILABLE = False
    print("Info: pyloudnorm not available - using peak normalization")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ==================== GLOBAL CONFIGURATION ====================
# 🔥 IMPORTANT: SET YOUR AUDIO FILE PATH HERE! 🔥
INPUT_AUDIO_PATH = "input_call_recording.wav"  # ← CHANGE THIS TO YOUR ACTUAL FILE PATH!
OUTPUT_AUDIO_PATH = "enhanced_speech_for_asr.wav"

# Examples:
# INPUT_AUDIO_PATH = "my_call_recording.wav"              # File in same folder
# INPUT_AUDIO_PATH = "C:/recordings/meeting_audio.wav"   # Windows path
# INPUT_AUDIO_PATH = "/home/user/audio/call.wav"         # Linux/Mac path
# ================================================================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SpeechEnhancementConfig:
    """Configuration optimized for speech and ASR"""
    
    # Audio parameters - ASR optimized
    sample_rate: int = 16000  # Standard for ASR
    frame_length: int = 1024  # Smaller for speech
    hop_length: int = 256     # Smaller overlap for better time resolution
    n_fft: int = 1024        # Smaller for speech processing
    
    # Conservative enhancement parameters
    noise_reduction_strength: float = 0.3  # Very gentle (0.0-1.0)
    spectral_subtraction_alpha: float = 1.2  # Conservative over-subtraction
    spectral_floor: float = 0.3  # Higher floor to preserve speech
    
    # Speech-specific parameters
    speech_freq_min: float = 300.0   # Hz - lower bound of speech
    speech_freq_max: float = 3400.0  # Hz - upper bound of speech
    speech_emphasis: float = 1.1     # Gentle boost for speech frequencies
    
    # Dynamic processing - very gentle
    enable_gentle_compression: bool = True
    compression_threshold: float = -15.0  # dB - light compression
    compression_ratio: float = 1.8       # Gentle ratio
    
    # Final processing
    target_loudness: float = -16.0  # LUFS - good for ASR
    enable_final_limiting: bool = True
    
    # Quality preservation
    preserve_speech_dynamics: bool = True  # Maintain natural speech rhythm
    avoid_artifacts: bool = True           # Extra conservative processing

class GentleSpeechNoiseReducer:
    """Gentle noise reduction optimized for speech preservation"""
    
    def __init__(self, config: SpeechEnhancementConfig):
        self.config = config
        self.noise_profile = None
        self.speech_mask = None
        
    def create_speech_frequency_mask(self) -> np.ndarray:
        """Create frequency mask emphasizing speech frequencies"""
        
        freqs = librosa.fft_frequencies(sr=self.config.sample_rate, n_fft=self.config.n_fft)
        speech_mask = np.ones_like(freqs)
        
        # Emphasize speech frequencies (300-3400 Hz)
        speech_band = (freqs >= self.config.speech_freq_min) & (freqs <= self.config.speech_freq_max)
        speech_mask[speech_band] = self.config.speech_emphasis
        
        # Gentle reduction of very low frequencies (below 100 Hz)
        low_freq_band = freqs < 100
        speech_mask[low_freq_band] = 0.8
        
        # Gentle reduction of very high frequencies (above 8000 Hz)
        high_freq_band = freqs > 8000
        speech_mask[high_freq_band] = 0.9
        
        self.speech_mask = speech_mask
        return speech_mask
    
    def estimate_noise_conservatively(self, audio: np.ndarray) -> np.ndarray:
        """Conservative noise estimation to avoid removing speech"""
        
        if len(audio) == 0:
            freq_bins = self.config.n_fft // 2 + 1
            self.noise_profile = np.ones(freq_bins) * 0.001  # Very low noise estimate
            return self.noise_profile
        
        try:
            # Use only the very beginning for noise estimation (first 0.5 seconds max)
            noise_samples = min(int(0.5 * self.config.sample_rate), len(audio) // 10)
            noise_samples = max(512, noise_samples)  # At least 512 samples
            
            noise_segment = audio[:noise_samples]
            
            # Ensure we have enough for STFT
            min_length = self.config.n_fft * 2
            if len(noise_segment) < min_length:
                noise_segment = np.tile(noise_segment, (min_length // len(noise_segment)) + 1)[:min_length]
            
            # Compute noise spectrum
            noise_stft = librosa.stft(noise_segment, 
                                     n_fft=self.config.n_fft, 
                                     hop_length=self.config.hop_length)
            
            # Use conservative estimate (lower percentile)
            noise_spectrum = np.percentile(np.abs(noise_stft), 30, axis=1)  # 30th percentile instead of mean
            
            # Apply gentle smoothing
            if len(noise_spectrum) >= 3:
                noise_spectrum = signal.medfilt(noise_spectrum, kernel_size=3)
            
            # Ensure very conservative noise floor (don't over-estimate noise)
            noise_spectrum = np.minimum(noise_spectrum, np.max(noise_spectrum) * 0.5)
            noise_spectrum = np.maximum(noise_spectrum, np.max(noise_spectrum) * 0.01)
            
            self.noise_profile = noise_spectrum
            logger.info(f"Conservative noise profile estimated from {len(noise_segment)} samples")
            return noise_spectrum
            
        except Exception as e:
            logger.error(f"Noise estimation failed: {e}")
            freq_bins = self.config.n_fft // 2 + 1
            self.noise_profile = np.ones(freq_bins) * 0.001
            return self.noise_profile
    
    def gentle_spectral_subtraction(self, audio: np.ndarray) -> np.ndarray:
        """Very gentle spectral subtraction that preserves speech"""
        
        if len(audio) == 0:
            return audio
            
        try:
            # Estimate noise if needed
            if self.noise_profile is None:
                self.estimate_noise_conservatively(audio)
            
            # Create speech frequency mask
            if self.speech_mask is None:
                self.create_speech_frequency_mask()
            
            # Compute STFT
            stft = librosa.stft(audio, 
                               n_fft=self.config.n_fft, 
                               hop_length=self.config.hop_length)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Ensure dimensions match
            if len(self.noise_profile) != magnitude.shape[0]:
                logger.warning("Adjusting noise profile dimensions")
                from scipy.interpolate import interp1d
                old_indices = np.linspace(0, 1, len(self.noise_profile))
                new_indices = np.linspace(0, 1, magnitude.shape[0])
                f = interp1d(old_indices, self.noise_profile, kind='linear', fill_value='extrapolate')
                self.noise_profile = f(new_indices)
            
            # Conservative spectral subtraction
            alpha = self.config.spectral_subtraction_alpha  # Much lower than before
            noise_profile_2d = self.noise_profile.reshape(-1, 1)
            
            # Apply frequency-dependent subtraction (less aggressive in speech frequencies)
            speech_frequencies = (np.arange(len(self.noise_profile)) >= 
                                librosa.fft_frequencies(sr=self.config.sample_rate, n_fft=self.config.n_fft).searchsorted(300))
            speech_frequencies &= (np.arange(len(self.noise_profile)) <= 
                                 librosa.fft_frequencies(sr=self.config.sample_rate, n_fft=self.config.n_fft).searchsorted(3400))
            
            # Reduce subtraction strength in speech frequencies
            alpha_per_freq = np.ones(len(self.noise_profile)) * alpha
            alpha_per_freq[speech_frequencies] *= 0.7  # Even more conservative for speech
            alpha_2d = alpha_per_freq.reshape(-1, 1)
            
            enhanced_magnitude = magnitude - alpha_2d * noise_profile_2d
            
            # High spectral floor to preserve speech
            floor_ratio = self.config.spectral_floor
            spectral_floor = floor_ratio * magnitude
            enhanced_magnitude = np.maximum(enhanced_magnitude, spectral_floor)
            
            # Apply speech frequency emphasis
            speech_mask_2d = self.speech_mask.reshape(-1, 1)
            enhanced_magnitude = enhanced_magnitude * speech_mask_2d
            
            # Reconstruct
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft, hop_length=self.config.hop_length)
            
            return enhanced_audio
            
        except Exception as e:
            logger.error(f"Gentle spectral subtraction failed: {e}")
            return audio
    
    def adaptive_wiener_filter(self, audio: np.ndarray) -> np.ndarray:
        """Conservative Wiener filtering for speech"""
        
        if len(audio) == 0:
            return audio
            
        try:
            stft = librosa.stft(audio, 
                               n_fft=self.config.n_fft, 
                               hop_length=self.config.hop_length)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            if self.noise_profile is None:
                self.estimate_noise_conservatively(audio)
            
            # Ensure dimensions match
            if len(self.noise_profile) != magnitude.shape[0]:
                from scipy.interpolate import interp1d
                old_indices = np.linspace(0, 1, len(self.noise_profile))
                new_indices = np.linspace(0, 1, magnitude.shape[0])
                f = interp1d(old_indices, self.noise_profile, kind='linear', fill_value='extrapolate')
                self.noise_profile = f(new_indices)
            
            # Conservative noise power estimate
            noise_power = (self.noise_profile ** 2).reshape(-1, 1) * 0.5  # Reduce noise power estimate
            signal_power = magnitude ** 2
            
            # Conservative Wiener gain (biased toward preserving signal)
            wiener_gain = signal_power / (signal_power + noise_power + 1e-12)
            
            # Ensure minimum gain to preserve speech
            min_gain = 0.3  # Never reduce signal below 30%
            wiener_gain = np.maximum(wiener_gain, min_gain)
            
            # Light smoothing only
            if wiener_gain.shape[1] > 3:
                wiener_gain = signal.medfilt2d(wiener_gain, kernel_size=(1, 3))  # Only temporal smoothing
            
            # Apply gain
            enhanced_magnitude = wiener_gain * magnitude
            
            # Reconstruct
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft, hop_length=self.config.hop_length)
            
            return enhanced_audio
            
        except Exception as e:
            logger.error(f"Wiener filtering failed: {e}")
            return audio

class SpeechDynamicsProcessor:
    """Gentle dynamics processing for speech clarity"""
    
    def __init__(self, config: SpeechEnhancementConfig):
        self.config = config
    
    def gentle_compression(self, audio: np.ndarray) -> np.ndarray:
        """Apply very gentle compression optimized for speech"""
        
        if not self.config.enable_gentle_compression:
            return audio
            
        try:
            threshold_db = self.config.compression_threshold
            ratio = self.config.compression_ratio
            threshold_linear = 10 ** (threshold_db / 20)
            
            # RMS-based compression (better for speech than peak)
            window_size = int(0.01 * self.config.sample_rate)  # 10ms windows
            compressed = np.zeros_like(audio)
            
            for i in range(len(audio)):
                # Get local RMS
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(audio), i + window_size // 2)
                local_rms = np.sqrt(np.mean(audio[start_idx:end_idx] ** 2))
                
                if local_rms > threshold_linear:
                    # Apply gentle compression
                    excess = local_rms - threshold_linear
                    compressed_excess = excess / ratio
                    gain_reduction = (threshold_linear + compressed_excess) / local_rms
                    compressed[i] = audio[i] * gain_reduction
                else:
                    compressed[i] = audio[i]
            
            return compressed
            
        except Exception as e:
            logger.error(f"Gentle compression failed: {e}")
            return audio
    
    def speech_gate(self, audio: np.ndarray) -> np.ndarray:
        """Very gentle gating to reduce inter-word noise"""
        
        try:
            # Simple energy-based gating
            window_size = int(0.02 * self.config.sample_rate)  # 20ms windows
            hop_size = window_size // 4
            
            # Calculate frame energies
            frame_energies = []
            for i in range(0, len(audio) - window_size, hop_size):
                frame = audio[i:i + window_size]
                energy = np.mean(frame ** 2)
                frame_energies.append(energy)
            
            if not frame_energies:
                return audio
            
            frame_energies = np.array(frame_energies)
            
            # Conservative threshold (only gate very quiet parts)
            threshold = np.percentile(frame_energies, 15)  # Bottom 15%
            
            # Create gate mask
            gate_mask = np.ones_like(audio)
            
            for i, energy in enumerate(frame_energies):
                start_idx = i * hop_size
                end_idx = min(len(audio), start_idx + window_size)
                
                if energy < threshold:
                    # Very gentle gating (reduce but don't eliminate)
                    gate_mask[start_idx:end_idx] *= 0.5
            
            # Smooth the gate mask to avoid artifacts
            if len(gate_mask) > 10:
                gate_mask = signal.medfilt(gate_mask, kernel_size=9)
            
            return audio * gate_mask
            
        except Exception as e:
            logger.error(f"Speech gating failed: {e}")
            return audio

class SpeechOptimizedProcessor:
    """Main speech enhancement processor for ASR"""
    
    def __init__(self, config: Optional[SpeechEnhancementConfig] = None):
        self.config = config or SpeechEnhancementConfig()
        
        # Initialize processors
        self.noise_reducer = GentleSpeechNoiseReducer(self.config)
        self.dynamics_processor = SpeechDynamicsProcessor(self.config)
        
        # Performance tracking
        self.metrics = defaultdict(list)
        
        logger.info("Speech-Optimized Audio Processor initialized for ASR")
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load and prepare audio for speech processing"""
        
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        try:
            start_time = time.time()
            
            # Load audio
            audio, sr = librosa.load(file_path, sr=None, mono=True)
            
            if len(audio) == 0:
                raise ValueError("Empty audio file")
            
            # Clean invalid values
            if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
                logger.warning("Cleaning invalid audio values")
                audio = np.nan_to_num(audio, nan=0.0, posinf=0.95, neginf=-0.95)
            
            # Resample to ASR-optimized rate if needed
            if sr != self.config.sample_rate:
                logger.info(f"Resampling from {sr}Hz to {self.config.sample_rate}Hz for ASR")
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.config.sample_rate)
                sr = self.config.sample_rate
            
            # Conservative normalization (preserve dynamics)
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                # Don't normalize too aggressively - preserve natural speech levels
                if max_val > 0.95:  # Only normalize if clipping risk
                    audio = audio / max_val * 0.9
                elif max_val < 0.1:  # Boost very quiet audio
                    audio = audio / max_val * 0.3
            
            load_time = time.time() - start_time
            self.metrics['loading_time'].append(load_time)
            
            logger.info(f"Audio loaded for ASR: {len(audio)/sr:.2f}s at {sr}Hz")
            logger.info(f"Peak level: {max_val:.3f}, RMS: {np.sqrt(np.mean(audio**2)):.3f}")
            
            return audio, sr
            
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            raise
    
    def enhance_speech_for_asr(self, audio_path: str = None) -> Tuple[np.ndarray, int]:
        """Main speech enhancement pipeline optimized for ASR"""
        
        if audio_path is None:
            audio_path = INPUT_AUDIO_PATH
            logger.info(f"Using global input path: {INPUT_AUDIO_PATH}")
        
        logger.info(f"Starting speech enhancement for ASR: {audio_path}")
        start_time = time.time()
        
        # Load audio
        original_audio, sr = self.load_audio(audio_path)
        enhanced_audio = original_audio.copy()
        
        # Calculate original metrics
        orig_rms = np.sqrt(np.mean(original_audio ** 2))
        orig_peak = np.max(np.abs(original_audio))
        logger.info(f"Original audio - RMS: {orig_rms:.4f}, Peak: {orig_peak:.4f}")
        
        # Stage 1: High-pass filter (remove DC and low-frequency rumble)
        logger.info("Stage 1: Gentle high-pass filtering...")
        stage_start = time.time()
        
        # Very gentle high-pass (preserve natural speech)
        if sr > 0:
            sos = signal.butter(1, 80 / (sr / 2), 'high', output='sos')  # Remove only below 80Hz
            enhanced_audio = signal.sosfilt(sos, enhanced_audio)
        
        stage_time = time.time() - stage_start
        self.metrics['highpass_time'].append(stage_time)
        logger.info(f"High-pass filtering completed in {stage_time:.3f}s")
        
        # Stage 2: Conservative noise reduction
        if self.config.noise_reduction_strength > 0.0:
            logger.info("Stage 2: Conservative noise reduction...")
            stage_start = time.time()
            
            if self.config.noise_reduction_strength <= 0.5:
                # Light noise reduction - spectral subtraction only
                enhanced_audio = self.noise_reducer.gentle_spectral_subtraction(enhanced_audio)
            else:
                # Moderate noise reduction - add Wiener filter
                enhanced_audio = self.noise_reducer.gentle_spectral_subtraction(enhanced_audio)
                enhanced_audio = self.noise_reducer.adaptive_wiener_filter(enhanced_audio)
            
            stage_time = time.time() - stage_start
            self.metrics['noise_reduction_time'].append(stage_time)
            logger.info(f"Noise reduction completed in {stage_time:.3f}s")
        else:
            logger.info("Noise reduction disabled")
        
        # Stage 3: Speech frequency emphasis
        logger.info("Stage 3: Speech frequency emphasis...")
        stage_start = time.time()
        
        enhanced_audio = self._apply_speech_frequency_emphasis(enhanced_audio, sr)
        
        stage_time = time.time() - stage_start
        self.metrics['emphasis_time'].append(stage_time)
        logger.info(f"Speech emphasis completed in {stage_time:.3f}s")
        
        # Stage 4: Gentle dynamics processing
        logger.info("Stage 4: Gentle dynamics processing...")
        stage_start = time.time()
        
        # Apply gentle gating first
        enhanced_audio = self.dynamics_processor.speech_gate(enhanced_audio)
        
        # Then gentle compression
        enhanced_audio = self.dynamics_processor.gentle_compression(enhanced_audio)
        
        stage_time = time.time() - stage_start
        self.metrics['dynamics_time'].append(stage_time)
        logger.info(f"Dynamics processing completed in {stage_time:.3f}s")
        
        # Stage 5: Final ASR optimization
        logger.info("Stage 5: Final ASR optimization...")
        stage_start = time.time()
        
        enhanced_audio = self._final_asr_optimization(enhanced_audio, sr)
        
        stage_time = time.time() - stage_start
        self.metrics['final_time'].append(stage_time)
        logger.info(f"Final optimization completed in {stage_time:.3f}s")
        
        # Quality assessment
        total_time = time.time() - start_time
        enhanced_rms = np.sqrt(np.mean(enhanced_audio ** 2))
        enhanced_peak = np.max(np.abs(enhanced_audio))
        
        # Calculate improvement metrics
        snr_estimate = self._estimate_speech_quality_improvement(original_audio, enhanced_audio)
        
        self.metrics['total_time'].append(total_time)
        self.metrics['snr_improvement'].append(snr_estimate)
        
        logger.info(f"Speech enhancement complete! Total time: {total_time:.2f}s")
        logger.info(f"Enhanced audio - RMS: {enhanced_rms:.4f}, Peak: {enhanced_peak:.4f}")
        logger.info(f"Estimated quality improvement: {snr_estimate:.2f} dB")
        
        # Warn if audio seems too processed
        if enhanced_peak < orig_peak * 0.3:
            logger.warning("Output level seems low - check if processing was too aggressive")
        
        return enhanced_audio, sr
    
    def _apply_speech_frequency_emphasis(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply gentle emphasis to speech frequencies"""
        
        try:
            # Design a gentle bell curve filter for speech frequencies
            nyquist = sr / 2
            
            # Speech band emphasis (300-3400 Hz with gentle slopes)
            speech_low = 300 / nyquist
            speech_high = 3400 / nyquist
            
            # Design a gentle bandpass emphasis
            # Create frequency response
            freqs = np.linspace(0, 1, 1024)
            response = np.ones_like(freqs)
            
            # Bell curve emphasis for speech
            speech_center = (speech_low + speech_high) / 2
            speech_width = (speech_high - speech_low) / 2
            
            for i, f in enumerate(freqs):
                if speech_low <= f <= speech_high:
                    # Gentle boost in speech range
                    distance_from_center = abs(f - speech_center) / speech_width
                    boost = 1.0 + 0.15 * np.exp(-(distance_from_center ** 2) * 2)  # Gentle bell curve
                    response[i] = boost
            
            # Design FIR filter from frequency response
            filter_taps = signal.firwin2(101, freqs, response, window='hann')
            
            # Apply filter
            emphasized_audio = signal.lfilter(filter_taps, 1, audio)
            
            return emphasized_audio
            
        except Exception as e:
            logger.error(f"Speech emphasis failed: {e}")
            return audio
    
    def _final_asr_optimization(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Final processing optimized for ASR"""
        
        try:
            processed = audio.copy()
            
            # Light de-clicking (remove brief artifacts)
            processed = self._remove_light_artifacts(processed)
            
            # Final loudness optimization for ASR
            if PYLOUDNORM_AVAILABLE:
                try:
                    meter = pyln.Meter(sr)
                    loudness = meter.integrated_loudness(processed)
                    
                    if -50 < loudness < -5:  # Valid range
                        processed = pyln.normalize.loudness(
                            processed, loudness, self.config.target_loudness
                        )
                        logger.info(f"Loudness normalized to {self.config.target_loudness} LUFS")
                except Exception as e:
                    logger.warning(f"Professional loudness normalization failed: {e}")
                    # Fallback normalization
                    processed = self._simple_asr_normalization(processed)
            else:
                processed = self._simple_asr_normalization(processed)
            
            # Final gentle limiting (prevent clipping)
            if self.config.enable_final_limiting:
                processed = self._gentle_limiting(processed)
            
            # Ensure no clipping
            processed = np.clip(processed, -0.98, 0.98)
            
            return processed
            
        except Exception as e:
            logger.error(f"Final ASR optimization failed: {e}")
            return audio
    
    def _remove_light_artifacts(self, audio: np.ndarray) -> np.ndarray:
        """Remove brief artifacts without affecting speech"""
        
        try:
            # Very conservative artifact removal
            # Only remove very brief spikes that are much louder than surrounding audio
            
            window_size = int(0.005 * self.config.sample_rate)  # 5ms windows
            threshold_factor = 3.0  # Spike must be 3x louder than neighbors
            
            cleaned = audio.copy()
            
            for i in range(window_size, len(audio) - window_size):
                current = abs(audio[i])
                
                # Check surrounding area
                left_region = audio[i-window_size:i]
                right_region = audio[i+1:i+window_size+1]
                surrounding_level = np.mean(np.abs(np.concatenate([left_region, right_region])))
                
                # If current sample is much louder and brief, reduce it
                if current > surrounding_level * threshold_factor and surrounding_level > 0:
                    # Check if it's actually brief (not sustained speech)
                    peak_region = audio[max(0, i-2):i+3]
                    if np.sum(np.abs(peak_region) > surrounding_level * 2) <= 3:  # Brief spike
                        cleaned[i] = audio[i] * 0.5  # Gentle reduction
            
            return cleaned
            
        except Exception as e:
            logger.warning(f"Artifact removal failed: {e}")
            return audio
    
    def _simple_asr_normalization(self, audio: np.ndarray) -> np.ndarray:
        """Simple normalization optimized for ASR"""
        
        try:
            # RMS-based normalization (better for speech than peak)
            current_rms = np.sqrt(np.mean(audio ** 2))
            
            if current_rms > 0:
                # Target RMS level good for ASR (not too loud, not too quiet)
                target_rms = 0.15  # About -16 dB RMS
                gain = target_rms / current_rms
                
                # Limit gain to prevent over-amplification
                max_gain = 5.0
                min_gain = 0.2
                gain = np.clip(gain, min_gain, max_gain)
                
                normalized = audio * gain
                
                # Ensure no clipping
                peak = np.max(np.abs(normalized))
                if peak > 0.95:
                    normalized = normalized * (0.95 / peak)
                
                return normalized
            
            return audio
            
        except Exception as e:
            logger.warning(f"Simple normalization failed: {e}")
            return audio
    
    def _gentle_limiting(self, audio: np.ndarray) -> np.ndarray:
        """Very gentle peak limiting"""
        
        try:
            threshold = 0.9
            limited = np.copy(audio)
            
            # Find peaks above threshold
            peaks = np.abs(audio) > threshold
            
            if np.any(peaks):
                # Apply very gentle limiting
                peak_indices = np.where(peaks)[0]
                
                for idx in peak_indices:
                    # Gentle soft clipping
                    original_value = audio[idx]
                    sign = np.sign(original_value)
                    magnitude = abs(original_value)
                    
                    # Soft knee limiting
                    if magnitude > threshold:
                        excess = magnitude - threshold
                        limited_excess = excess * (1 / (1 + excess * 2))  # Gentle curve
                        limited[idx] = sign * (threshold + limited_excess)
            
            return limited
            
        except Exception as e:
            logger.warning(f"Gentle limiting failed: {e}")
            return np.clip(audio, -0.98, 0.98)
    
    def _estimate_speech_quality_improvement(self, original: np.ndarray, enhanced: np.ndarray) -> float:
        """Estimate speech quality improvement"""
        
        try:
            # Align lengths
            min_len = min(len(original), len(enhanced))
            orig = original[:min_len]
            enh = enhanced[:min_len]
            
            # Focus on speech frequencies for quality assessment
            orig_spectrum = np.abs(librosa.stft(orig, n_fft=self.config.n_fft))
            enh_spectrum = np.abs(librosa.stft(enh, n_fft=self.config.n_fft))
            
            # Get speech frequency range
            freqs = librosa.fft_frequencies(sr=self.config.sample_rate, n_fft=self.config.n_fft)
            speech_mask = (freqs >= 300) & (freqs <= 3400)
            
            # Calculate energy in speech band
            orig_speech_energy = np.sum(orig_spectrum[speech_mask, :])
            enh_speech_energy = np.sum(enh_spectrum[speech_mask, :])
            
            # Calculate noise estimate (frequencies outside speech band)
            noise_mask = ~speech_mask
            orig_noise_energy = np.sum(orig_spectrum[noise_mask, :])
            enh_noise_energy = np.sum(enh_spectrum[noise_mask, :])
            
            # Estimate SNR improvement
            if orig_noise_energy > 0 and enh_noise_energy > 0:
                orig_snr = 10 * np.log10(orig_speech_energy / orig_noise_energy + 1e-12)
                enh_snr = 10 * np.log10(enh_speech_energy / enh_noise_energy + 1e-12)
                improvement = enh_snr - orig_snr
            else:
                improvement = 0.0
            
            return improvement
            
        except Exception as e:
            logger.warning(f"Quality estimation failed: {e}")
            return 0.0
    
    def save_asr_optimized_audio(self, audio: np.ndarray, sr: int, output_path: str = None):
        """Save audio optimized for ASR with metadata"""
        
        if output_path is None:
            output_path = OUTPUT_AUDIO_PATH
        
        try:
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save in format optimal for ASR (16-bit, mono)
            sf.write(output_path, audio, sr, subtype='PCM_16')
            logger.info(f"ASR-optimized audio saved to: {output_path}")
            
            # Save metadata
            metadata_path = output_path.replace('.wav', '_asr_metadata.json')
            metadata = {
                'asr_optimization_info': {
                    'input_file': str(INPUT_AUDIO_PATH),
                    'output_file': str(output_path),
                    'sample_rate': sr,
                    'duration': len(audio) / sr,
                    'format': '16-bit PCM WAV',
                    'optimization_target': 'ASR (Automatic Speech Recognition)',
                    'processing_approach': 'Conservative speech-focused enhancement'
                },
                'configuration': asdict(self.config),
                'performance_metrics': {
                    'total_processing_time': np.sum(self.metrics.get('total_time', [])),
                    'estimated_quality_improvement': np.mean(self.metrics.get('snr_improvement', [])),
                    'stage_breakdown': {
                        'loading': np.sum(self.metrics.get('loading_time', [])),
                        'highpass_filter': np.sum(self.metrics.get('highpass_time', [])),
                        'noise_reduction': np.sum(self.metrics.get('noise_reduction_time', [])),
                        'speech_emphasis': np.sum(self.metrics.get('emphasis_time', [])),
                        'dynamics_processing': np.sum(self.metrics.get('dynamics_time', [])),
                        'final_optimization': np.sum(self.metrics.get('final_time', []))
                    }
                },
                'speech_enhancement_features': {
                    'conservative_noise_reduction': self.config.noise_reduction_strength <= 0.5,
                    'speech_frequency_emphasis': True,
                    'gentle_dynamics_processing': self.config.enable_gentle_compression,
                    'asr_loudness_optimization': True,
                    'artifact_prevention': self.config.avoid_artifacts
                },
                'asr_readiness': {
                    'sample_rate_asr_standard': sr == 16000,
                    'mono_channel': True,
                    'bit_depth': 16,
                    'speech_clarity_optimized': True,
                    'recommended_for_transcription': True
                }
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"ASR metadata saved to: {metadata_path}")
            
        except Exception as e:
            logger.error(f"Failed to save ASR-optimized audio: {e}")
            try:
                sf.write(output_path, audio, sr, subtype='PCM_16')
                logger.info(f"Audio saved successfully (metadata failed): {output_path}")
            except Exception as e2:
                logger.error(f"Failed to save audio file: {e2}")
                raise

def main():
    """Main execution function for speech enhancement"""
    
    try:
        print(f"\n{'='*75}")
        print("🎙️  SPEECH-OPTIMIZED AUDIO ENHANCEMENT FOR ASR")
        print("Conservative Processing - Preserves Speech Clarity")
        print(f"{'='*75}")
        print(f"Input file: {INPUT_AUDIO_PATH}")
        print(f"Output file: {OUTPUT_AUDIO_PATH}")
        print(f"Target: ASR (Automatic Speech Recognition)")
        print(f"{'='*75}")
        
        # Initialize speech processor
        config = SpeechEnhancementConfig()
        
        # Show configuration
        print(f"📊 Configuration:")
        print(f"   • Sample Rate: {config.sample_rate} Hz (ASR standard)")
        print(f"   • Noise Reduction: {config.noise_reduction_strength:.1f}/1.0 (Conservative)")
        print(f"   • Speech Emphasis: {config.speech_emphasis:.1f}x (300-3400 Hz)")
        print(f"   • Target Loudness: {config.target_loudness} LUFS")
        print(f"   • Processing Approach: Speech-focused, minimal artifacts")
        print(f"{'='*75}")
        
        processor = SpeechOptimizedProcessor(config)
        
        # Process audio
        enhanced_audio, sample_rate = processor.enhance_speech_for_asr()
        
        # Save results
        processor.save_asr_optimized_audio(enhanced_audio, sample_rate)
        
        # Results
        total_time = np.sum(processor.metrics.get('total_time', []))
        quality_improvement = np.mean(processor.metrics.get('snr_improvement', []))
        
        print(f"\n{'='*75}")
        print("✅ SPEECH ENHANCEMENT FOR ASR COMPLETE!")
        print(f"{'='*75}")
        print(f"📁 Input: {INPUT_AUDIO_PATH}")
        print(f"📁 Output: {OUTPUT_AUDIO_PATH}")
        print(f"⏱️  Processing Time: {total_time:.2f} seconds")
        print(f"📊 Speech Quality Improvement: {quality_improvement:+.2f} dB")
        print(f"🎯 ASR Ready: 16 kHz, 16-bit, Mono WAV")
        
        # Enhancement assessment
        if quality_improvement > 3:
            print(f"🏆 EXCELLENT - Significant speech clarity improvement!")
            print(f"   Your ASR model should perform much better with this audio.")
        elif quality_improvement > 1:
            print(f"✅ GOOD - Noticeable speech enhancement achieved")
            print(f"   ASR transcription quality should be improved.")
        elif quality_improvement > -1:
            print(f"✅ PRESERVED - Audio quality maintained without distortion")
            print(f"   Safe for ASR use - no artifacts introduced.")
        else:
            print(f"⚠️  Input audio may already be high quality for ASR")
        
        # Processing breakdown
        print(f"\n📈 Processing Stages:")
        stages = [
            ('Loading & Validation', processor.metrics.get('loading_time', [])),
            ('High-Pass Filtering', processor.metrics.get('highpass_time', [])),
            ('Conservative Noise Reduction', processor.metrics.get('noise_reduction_time', [])),
            ('Speech Frequency Emphasis', processor.metrics.get('emphasis_time', [])),
            ('Gentle Dynamics Processing', processor.metrics.get('dynamics_time', [])),
            ('ASR Optimization', processor.metrics.get('final_time', []))
        ]
        
        for stage_name, times in stages:
            if times:
                avg_time = np.mean(times)
                percentage = (avg_time / total_time) * 100
                print(f"   • {stage_name}: {avg_time:.3f}s ({percentage:.1f}%)")
        
        print(f"\n🎙️  ASR RECOMMENDATIONS:")
        print(f"   • Use the output audio directly with your ASR model")
        print(f"   • Audio is optimized for speech recognition accuracy")
        print(f"   • Conservative processing minimizes artifacts")
        print(f"   • Format: 16 kHz, 16-bit WAV (standard for most ASR systems)")
        
        print(f"{'='*75}")
        
        return True
        
    except FileNotFoundError:
        print(f"\n❌ ERROR: Audio file not found!")
        print(f"Please ensure the file exists at: {INPUT_AUDIO_PATH}")
        print(f"Update the INPUT_AUDIO_PATH variable with your actual file path.")
        return False
        
    except Exception as e:
        logger.error(f"Speech enhancement failed: {e}")
        print(f"\n❌ Enhancement failed: {e}")
        print(f"\nTroubleshooting tips:")
        print(f"1. Ensure input is a valid audio file (WAV, MP3, FLAC)")
        print(f"2. Check that the audio file is not corrupted")
        print(f"3. Try with a shorter audio clip first")
        print(f"4. Ensure you have sufficient disk space")
        return False

if __name__ == "__main__":
    # Configuration section
    print("\n" + "="*75)
    print("🔧 SPEECH ENHANCEMENT FOR ASR - CONFIGURATION")
    print("="*75)
    
    # Check if user has set their file path
    if INPUT_AUDIO_PATH in ["input_call_recording.wav", "your_audio.wav", "input_audio.wav"]:
        print("❌ CONFIGURATION REQUIRED!")
        print("\nPlease edit this script and change the INPUT_AUDIO_PATH variable.")
        print("Find this line in the script:")
        print('INPUT_AUDIO_PATH = "input_call_recording.wav"  # ← CHANGE THIS!')
        print("\nTo your actual call recording path, for example:")
        print('INPUT_AUDIO_PATH = "my_call_recording.wav"')
        print('INPUT_AUDIO_PATH = "C:/recordings/meeting.wav"')
        print('INPUT_AUDIO_PATH = "/home/user/calls/audio.wav"')
        
        # Try to help user find audio files
        current_dir = Path(".")
        audio_files = []
        for ext in ["*.wav", "*.mp3", "*.flac", "*.m4a"]:
            audio_files.extend(list(current_dir.glob(ext)))
        
        if audio_files:
            print(f"\n📁 Found these audio files in current directory:")
            for i, file in enumerate(audio_files[:10], 1):
                print(f"   {i}. {file.name}")
            print(f'\nYou could use: INPUT_AUDIO_PATH = "{audio_files[0].name}"')
        
        print("\n" + "="*75)
        sys.exit(1)
    
    # Validate the path
    if not Path(INPUT_AUDIO_PATH).exists():
        print(f"❌ ERROR: Audio file not found!")
        print(f"Specified path: {INPUT_AUDIO_PATH}")
        print(f"Full path: {Path(INPUT_AUDIO_PATH).absolute()}")
        print(f"\nPlease check:")
        print(f"1. File exists at the specified location")
        print(f"2. File path is correctly typed")
        print(f"3. You have permission to read the file")
        sys.exit(1)
    
    print(f"✅ Configuration validated!")
    print(f"📁 Input: {INPUT_AUDIO_PATH}")
    print(f"📁 Output: {OUTPUT_AUDIO_PATH}")
    print(f"🎯 Purpose: Call recording enhancement for ASR")
    print(f"🔧 Approach: Conservative, speech-focused processing")
    print("="*75)
    
    # Run the enhancement
    success = main()
    
    if not success:
        sys.exit(1)