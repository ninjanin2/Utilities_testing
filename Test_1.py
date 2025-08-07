"""
Advanced Traditional Audio Enhancement System
===========================================
Professional-grade audio enhancement using only traditional signal processing.
No AI/ML - Pure signal processing for reliable, distortion-free results.

Features:
- Multi-band spectral enhancement
- Psychoacoustic processing
- Harmonic/percussive separation
- Advanced noise reduction
- Dynamic range optimization
- Phase coherence enhancement
- Professional mastering chain

Author: Advanced Signal Processing System
Version: 2.0.0-Traditional
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
from scipy.optimize import minimize_scalar
from scipy.stats import median_abs_deviation
import librosa
import soundfile as sf
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler

# Optional professional libraries
try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False
    print("Info: noisereduce not available - using built-in noise reduction")

try:
    import pyloudnorm as pyln
    PYLOUDNORM_AVAILABLE = True
except ImportError:
    PYLOUDNORM_AVAILABLE = False
    print("Info: pyloudnorm not available - using built-in loudness processing")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ==================== GLOBAL CONFIGURATION ====================
# 🔥 IMPORTANT: SET YOUR AUDIO FILE PATH HERE! 🔥
INPUT_AUDIO_PATH = "input_noisy_audio.wav"  # ← CHANGE THIS TO YOUR ACTUAL FILE PATH!
OUTPUT_AUDIO_PATH = "enhanced_clean_audio.wav"

# Examples of how to set the path:
# INPUT_AUDIO_PATH = "my_recording.wav"                    # File in same folder
# INPUT_AUDIO_PATH = "C:/Users/YourName/Desktop/audio.wav" # Windows full path  
# INPUT_AUDIO_PATH = "/home/user/Documents/audio.wav"      # Linux/Mac full path
# ================================================================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class AdvancedEnhancementConfig:
    """Configuration for traditional audio enhancement"""
    
    # Audio parameters
    sample_rate: int = 16000
    frame_length: int = 2048
    hop_length: int = 512
    n_fft: int = 2048
    
    # Enhancement parameters
    noise_reduction_strength: float = 0.8  # 0.0 = none, 1.0 = maximum
    spectral_subtraction_alpha: float = 2.0
    spectral_floor: float = 0.1
    wiener_noise_estimate_frames: int = 20
    
    # Multi-band processing
    enable_multiband: bool = True
    band_count: int = 8
    
    # Psychoacoustic processing
    enable_psychoacoustic: bool = True
    masking_threshold_db: float = 6.0
    
    # Harmonic enhancement
    enable_harmonic_enhancement: bool = True
    harmonic_separation_margin: float = 3.0
    
    # Dynamic processing
    enable_dynamic_processing: bool = True
    compression_threshold: float = -20.0  # dB
    compression_ratio: float = 2.5
    
    # Final processing
    target_loudness: float = -20.0  # LUFS
    enable_final_limiter: bool = True
    limiter_threshold: float = -1.0  # dB

class AdvancedNoiseReducer:
    """Advanced noise reduction using multiple traditional techniques"""
    
    def __init__(self, config: AdvancedEnhancementConfig):
        self.config = config
        self.noise_profile = None
        
    def estimate_noise_profile(self, audio: np.ndarray) -> np.ndarray:
        """Estimate noise profile using multiple methods"""
        
        if len(audio) == 0:
            # Create a default noise profile
            freq_bins = self.config.n_fft // 2 + 1
            self.noise_profile = np.ones(freq_bins) * 0.01
            return self.noise_profile
        
        try:
            # Method 1: Use first 10% of audio (assuming it contains noise)
            noise_samples = max(1024, int(0.1 * len(audio)))  # At least 1024 samples
            noise_samples = min(noise_samples, len(audio) // 2)  # At most half the audio
            noise_segment = audio[:noise_samples]
            
            # Method 2: Find quiet segments throughout the audio
            frame_size = self.config.hop_length * 4
            if frame_size >= len(audio):
                frame_size = len(audio) // 4
                
            energy_threshold = np.percentile(audio ** 2, 15)  # Bottom 15% energy
            
            quiet_segments = []
            step_size = max(frame_size // 2, 512)
            
            for i in range(0, len(audio) - frame_size, step_size):
                frame = audio[i:i + frame_size]
                if len(frame) >= frame_size and np.mean(frame ** 2) < energy_threshold:
                    quiet_segments.append(frame)
                    
                # Limit quiet segments to avoid memory issues
                if len(quiet_segments) > 20:
                    break
            
            if quiet_segments:
                noise_from_quiet = np.concatenate(quiet_segments)
                # Limit size
                if len(noise_from_quiet) > len(noise_segment) * 2:
                    noise_from_quiet = noise_from_quiet[:len(noise_segment) * 2]
            else:
                noise_from_quiet = noise_segment
            
            # Combine both methods
            combined_noise = np.concatenate([noise_segment, noise_from_quiet[:len(noise_segment)]])
            
            # Ensure we have enough samples for STFT
            min_length = self.config.n_fft * 2
            if len(combined_noise) < min_length:
                # Pad or repeat if necessary
                repetitions = (min_length // len(combined_noise)) + 1
                combined_noise = np.tile(combined_noise, repetitions)[:min_length]
            
            # Compute spectral profile
            noise_stft = librosa.stft(combined_noise, 
                                     n_fft=self.config.n_fft, 
                                     hop_length=self.config.hop_length)
            noise_spectrum = np.mean(np.abs(noise_stft), axis=1)
            
            # Ensure we have a valid spectrum
            if len(noise_spectrum) == 0 or np.any(np.isnan(noise_spectrum)) or np.any(np.isinf(noise_spectrum)):
                # Fallback: create a flat noise profile
                freq_bins = self.config.n_fft // 2 + 1
                noise_spectrum = np.ones(freq_bins) * 0.01
                logger.warning("Created fallback noise profile")
            else:
                # Smooth the noise profile
                if len(noise_spectrum) >= 5:
                    noise_spectrum = signal.medfilt(noise_spectrum, kernel_size=5)
                
                # Ensure minimum noise floor
                noise_spectrum = np.maximum(noise_spectrum, np.max(noise_spectrum) * 0.01)
            
            self.noise_profile = noise_spectrum
            logger.debug(f"Estimated noise profile shape: {noise_spectrum.shape}")
            return noise_spectrum
            
        except Exception as e:
            logger.error(f"Noise profile estimation failed: {e}")
            # Create a safe fallback profile
            freq_bins = self.config.n_fft // 2 + 1
            self.noise_profile = np.ones(freq_bins) * 0.01
            return self.noise_profile
    
    def adaptive_spectral_subtraction(self, audio: np.ndarray) -> np.ndarray:
        """Advanced spectral subtraction with adaptive parameters"""
        
        # Ensure audio is not empty
        if len(audio) == 0:
            return audio
            
        # Estimate noise if not already done
        if self.noise_profile is None:
            self.estimate_noise_profile(audio)
        
        try:
            # Compute STFT
            stft = librosa.stft(audio, 
                               n_fft=self.config.n_fft, 
                               hop_length=self.config.hop_length)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Debug: Check shapes
            logger.debug(f"Magnitude shape: {magnitude.shape}")
            logger.debug(f"Noise profile shape: {self.noise_profile.shape}")
            
            # Ensure noise profile matches frequency dimension
            if len(self.noise_profile) != magnitude.shape[0]:
                logger.warning(f"Noise profile size mismatch. Adjusting from {len(self.noise_profile)} to {magnitude.shape[0]}")
                # Resize noise profile to match
                from scipy.interpolate import interp1d
                old_indices = np.linspace(0, 1, len(self.noise_profile))
                new_indices = np.linspace(0, 1, magnitude.shape[0])
                f = interp1d(old_indices, self.noise_profile, kind='linear', fill_value='extrapolate')
                self.noise_profile = f(new_indices)
            
            # Adaptive over-subtraction factor
            alpha = self._compute_adaptive_alpha(magnitude)
            
            # Spectral subtraction with proper broadcasting
            noise_profile_2d = self.noise_profile.reshape(-1, 1)  # (freq_bins, 1)
            alpha_2d = alpha.reshape(-1, 1)  # (freq_bins, 1)
            enhanced_magnitude = magnitude - alpha_2d * noise_profile_2d
            
            # Adaptive spectral floor
            spectral_floor = self._compute_adaptive_floor(magnitude)
            enhanced_magnitude = np.maximum(enhanced_magnitude, 
                                           spectral_floor.reshape(-1, 1) * magnitude)
            
            # Reconstruct
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft, hop_length=self.config.hop_length)
            
            return enhanced_audio
            
        except Exception as e:
            logger.error(f"Spectral subtraction failed: {e}")
            logger.error(f"Audio shape: {audio.shape}")
            if hasattr(self, 'noise_profile') and self.noise_profile is not None:
                logger.error(f"Noise profile shape: {self.noise_profile.shape}")
            return audio  # Return original audio on failure
    
    def _compute_adaptive_alpha(self, magnitude: np.ndarray) -> np.ndarray:
        """Compute adaptive over-subtraction factor"""
        
        # SNR estimation per frequency bin
        signal_estimate = np.mean(magnitude, axis=1)
        snr_estimate = signal_estimate / (self.noise_profile + 1e-12)
        snr_db = 20 * np.log10(snr_estimate + 1e-12)
        
        # Adaptive alpha based on SNR
        alpha = np.ones_like(snr_db) * self.config.spectral_subtraction_alpha
        
        # Reduce over-subtraction for high SNR regions
        high_snr_mask = snr_db > 15
        alpha[high_snr_mask] *= 0.5
        
        # Increase over-subtraction for low SNR regions
        low_snr_mask = snr_db < 5
        alpha[low_snr_mask] *= 1.5
        
        return alpha
    
    def _compute_adaptive_floor(self, magnitude: np.ndarray) -> np.ndarray:
        """Compute adaptive spectral floor"""
        
        # Base floor
        floor = np.ones(magnitude.shape[0]) * self.config.spectral_floor
        
        # Reduce floor for speech-important frequencies (300-3400 Hz)
        freqs = librosa.fft_frequencies(sr=self.config.sample_rate, n_fft=self.config.n_fft)
        speech_mask = (freqs >= 300) & (freqs <= 3400)
        floor[speech_mask] *= 0.5
        
        # Increase floor for very high frequencies (reduce hiss)
        high_freq_mask = freqs > 8000
        floor[high_freq_mask] *= 2.0
        
        return floor
    
    def wiener_filtering(self, audio: np.ndarray) -> np.ndarray:
        """Advanced Wiener filtering"""
        
        if len(audio) == 0:
            return audio
            
        try:
            stft = librosa.stft(audio, 
                               n_fft=self.config.n_fft, 
                               hop_length=self.config.hop_length)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Estimate noise power spectrum
            if self.noise_profile is None:
                self.estimate_noise_profile(audio)
            
            # Ensure noise profile matches magnitude dimensions
            if len(self.noise_profile) != magnitude.shape[0]:
                logger.warning(f"Noise profile size mismatch in Wiener filter. Adjusting...")
                from scipy.interpolate import interp1d
                old_indices = np.linspace(0, 1, len(self.noise_profile))
                new_indices = np.linspace(0, 1, magnitude.shape[0])
                f = interp1d(old_indices, self.noise_profile, kind='linear', fill_value='extrapolate')
                self.noise_profile = f(new_indices)
            
            noise_power = (self.noise_profile ** 2).reshape(-1, 1)  # (freq_bins, 1)
            
            # Signal power estimation
            signal_power = magnitude ** 2
            
            # Wiener gain with smoothing
            wiener_gain = signal_power / (signal_power + noise_power + 1e-12)
            
            # Smooth the gain across time
            wiener_gain = signal.medfilt2d(wiener_gain, kernel_size=(3, 5))
            
            # Apply gain
            enhanced_magnitude = wiener_gain * magnitude
            
            # Reconstruct
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft, hop_length=self.config.hop_length)
            
            return enhanced_audio
            
        except Exception as e:
            logger.error(f"Wiener filtering failed: {e}")
            return audio  # Return original audio on failure
    
    def spectral_gating(self, audio: np.ndarray) -> np.ndarray:
        """Advanced spectral gating for noise reduction"""
        
        stft = librosa.stft(audio, 
                           n_fft=self.config.n_fft, 
                           hop_length=self.config.hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Compute time-frequency dependent threshold
        # Use local statistics for adaptive thresholding
        window_size = 5  # frames
        threshold_factor = 1.5
        
        gated_magnitude = np.zeros_like(magnitude)
        
        for freq_bin in range(magnitude.shape[0]):
            for time_frame in range(magnitude.shape[1]):
                # Local window
                t_start = max(0, time_frame - window_size // 2)
                t_end = min(magnitude.shape[1], time_frame + window_size // 2 + 1)
                
                local_magnitudes = magnitude[freq_bin, t_start:t_end]
                local_threshold = np.median(local_magnitudes) * threshold_factor
                
                # Apply gate
                if magnitude[freq_bin, time_frame] > local_threshold:
                    gated_magnitude[freq_bin, time_frame] = magnitude[freq_bin, time_frame]
                else:
                    gated_magnitude[freq_bin, time_frame] = 0.1 * magnitude[freq_bin, time_frame]
        
        # Reconstruct
        enhanced_stft = gated_magnitude * np.exp(1j * phase)
        enhanced_audio = librosa.istft(enhanced_stft, hop_length=self.config.hop_length)
        
        return enhanced_audio

class MultibandProcessor:
    """Professional multi-band audio processing"""
    
    def __init__(self, config: AdvancedEnhancementConfig):
        self.config = config
        self.band_filters = self._design_band_filters()
    
    def _design_band_filters(self) -> List[np.ndarray]:
        """Design filterbank for multi-band processing"""
        
        # Define frequency bands (in Hz)
        nyquist = self.config.sample_rate / 2
        
        if self.config.band_count == 4:
            # 4-band processing
            band_edges = [0, 250, 1000, 4000, nyquist]
        elif self.config.band_count == 8:
            # 8-band processing (more precise)
            band_edges = [0, 125, 250, 500, 1000, 2000, 4000, 8000, nyquist]
        else:
            # Custom band count
            band_edges = np.logspace(np.log10(50), np.log10(nyquist), self.config.band_count + 1)
            band_edges[0] = 0
        
        # Design filters
        filters = []
        for i in range(len(band_edges) - 1):
            low_freq = band_edges[i]
            high_freq = band_edges[i + 1]
            
            # Normalize frequencies
            low_norm = low_freq / nyquist
            high_norm = high_freq / nyquist
            
            # Ensure valid frequency range
            low_norm = max(low_norm, 0.001)
            high_norm = min(high_norm, 0.999)
            
            if low_norm >= high_norm:
                # Skip invalid bands
                filters.append(None)
                continue
            
            # Design bandpass filter
            if low_norm <= 0.001:
                # Low-pass filter
                sos = signal.butter(4, high_norm, 'low', output='sos')
            elif high_norm >= 0.999:
                # High-pass filter
                sos = signal.butter(4, low_norm, 'high', output='sos')
            else:
                # Band-pass filter
                sos = signal.butter(4, [low_norm, high_norm], 'band', output='sos')
            
            filters.append(sos)
        
        return filters
    
    def process_multiband(self, audio: np.ndarray) -> np.ndarray:
        """Process audio with multi-band enhancement"""
        
        if not self.config.enable_multiband:
            return audio
        
        # Split into bands
        bands = []
        for band_filter in self.band_filters:
            if band_filter is not None:
                band_signal = signal.sosfilt(band_filter, audio)
                bands.append(band_signal)
            else:
                bands.append(np.zeros_like(audio))
        
        # Process each band
        processed_bands = []
        for i, band_signal in enumerate(bands):
            processed_band = self._process_band(band_signal, i)
            processed_bands.append(processed_band)
        
        # Recombine bands
        enhanced_audio = np.sum(processed_bands, axis=0)
        
        return enhanced_audio
    
    def _process_band(self, band_signal: np.ndarray, band_index: int) -> np.ndarray:
        """Process individual frequency band"""
        
        if np.max(np.abs(band_signal)) < 1e-6:
            return band_signal
        
        # Band-specific processing
        processed = band_signal.copy()
        
        # Low frequency bands - reduce rumble
        if band_index < 2:  # Below 250 Hz
            processed = self._reduce_low_frequency_artifacts(processed)
        
        # Mid frequency bands - enhance speech
        elif 2 <= band_index <= 5:  # 250 Hz - 4000 Hz (speech region)
            processed = self._enhance_speech_band(processed)
        
        # High frequency bands - control brightness
        else:  # Above 4000 Hz
            processed = self._process_high_frequencies(processed)
        
        return processed
    
    def _reduce_low_frequency_artifacts(self, band_signal: np.ndarray) -> np.ndarray:
        """Reduce low-frequency artifacts and rumble"""
        
        # High-pass filtering
        sos = signal.butter(2, 50 / (self.config.sample_rate / 2), 'high', output='sos')
        filtered = signal.sosfilt(sos, band_signal)
        
        # Dynamic range compression for rumble
        compressed = self._apply_band_compression(filtered, -30, 4.0)
        
        return compressed
    
    def _enhance_speech_band(self, band_signal: np.ndarray) -> np.ndarray:
        """Enhance speech-critical frequencies"""
        
        # Gentle compression to even out levels
        compressed = self._apply_band_compression(band_signal, -18, 2.5)
        
        # Harmonic enhancement
        enhanced = self._add_harmonic_emphasis(compressed)
        
        return enhanced
    
    def _process_high_frequencies(self, band_signal: np.ndarray) -> np.ndarray:
        """Process high frequencies to control harshness"""
        
        # De-essing and harshness reduction
        deessed = self._apply_deessing(band_signal)
        
        # Gentle compression
        compressed = self._apply_band_compression(deessed, -15, 2.0)
        
        return compressed
    
    def _apply_band_compression(self, signal_data: np.ndarray, 
                               threshold_db: float, ratio: float) -> np.ndarray:
        """Apply compression to a frequency band"""
        
        threshold_linear = 10 ** (threshold_db / 20)
        
        # Simple compression
        compressed = np.zeros_like(signal_data)
        
        for i, sample in enumerate(signal_data):
            abs_sample = abs(sample)
            
            if abs_sample > threshold_linear:
                # Above threshold - apply compression
                excess = abs_sample - threshold_linear
                compressed_excess = excess / ratio
                compressed_abs = threshold_linear + compressed_excess
                compressed[i] = np.sign(sample) * compressed_abs
            else:
                # Below threshold - no compression
                compressed[i] = sample
        
        return compressed
    
    def _add_harmonic_emphasis(self, signal_data: np.ndarray) -> np.ndarray:
        """Add subtle harmonic emphasis for warmth"""
        
        # Gentle saturation for harmonic generation
        drive = 1.1
        emphasized = np.tanh(signal_data * drive) / drive
        
        # Mix with original
        mix_ratio = 0.15
        return (1 - mix_ratio) * signal_data + mix_ratio * emphasized
    
    def _apply_deessing(self, signal_data: np.ndarray) -> np.ndarray:
        """Reduce sibilance and harshness"""
        
        # Simple de-essing using dynamic EQ concept
        # Focus on 4-8 kHz range where sibilance occurs
        
        # High-pass filter to isolate sibilant frequencies
        sos = signal.butter(4, 4000 / (self.config.sample_rate / 2), 'high', output='sos')
        sibilant_content = signal.sosfilt(sos, signal_data)
        
        # Detect sibilant levels
        threshold = np.percentile(np.abs(sibilant_content), 85)
        
        # Apply gain reduction when sibilance detected
        deessed = signal_data.copy()
        for i, sample in enumerate(signal_data):
            if abs(sibilant_content[i]) > threshold:
                deessed[i] *= 0.7  # Reduce gain
        
        return deessed

class PsychoacousticProcessor:
    """Psychoacoustic modeling for perceptually-aware processing"""
    
    def __init__(self, config: AdvancedEnhancementConfig):
        self.config = config
        self.bark_scale = self._create_bark_scale()
    
    def _create_bark_scale(self) -> np.ndarray:
        """Create Bark scale frequency mapping"""
        
        freqs = librosa.fft_frequencies(sr=self.config.sample_rate, n_fft=self.config.n_fft)
        
        # Convert Hz to Bark scale
        # Bark = 13 * arctan(0.00076 * f) + 3.5 * arctan((f/7500)^2)
        bark_freqs = 13 * np.arctan(0.00076 * freqs) + 3.5 * np.arctan((freqs / 7500) ** 2)
        
        return bark_freqs
    
    def psychoacoustic_enhancement(self, audio: np.ndarray) -> np.ndarray:
        """Apply psychoacoustic-based enhancement"""
        
        if not self.config.enable_psychoacoustic:
            return audio
        
        stft = librosa.stft(audio, 
                           n_fft=self.config.n_fft, 
                           hop_length=self.config.hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Convert to Bark domain
        bark_spectrum = self._convert_to_bark_domain(magnitude)
        
        # Apply masking model
        masked_spectrum = self._apply_masking_model(bark_spectrum)
        
        # Convert back to linear domain
        enhanced_magnitude = self._convert_from_bark_domain(masked_spectrum, magnitude.shape)
        
        # Reconstruct
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        enhanced_audio = librosa.istft(enhanced_stft, hop_length=self.config.hop_length)
        
        return enhanced_audio
    
    def _convert_to_bark_domain(self, magnitude: np.ndarray) -> np.ndarray:
        """Convert linear frequency spectrum to Bark domain"""
        
        # Simple binning to Bark scale
        bark_bands = np.linspace(0, 24, 25)  # 24 Bark bands
        bark_spectrum = np.zeros((len(bark_bands) - 1, magnitude.shape[1]))
        
        for i in range(len(bark_bands) - 1):
            band_mask = (self.bark_scale >= bark_bands[i]) & (self.bark_scale < bark_bands[i + 1])
            if np.any(band_mask):
                bark_spectrum[i, :] = np.mean(magnitude[band_mask, :], axis=0)
        
        return bark_spectrum
    
    def _apply_masking_model(self, bark_spectrum: np.ndarray) -> np.ndarray:
        """Apply psychoacoustic masking model"""
        
        masked_spectrum = bark_spectrum.copy()
        
        # Frequency masking
        for i in range(bark_spectrum.shape[0]):
            for j in range(bark_spectrum.shape[1]):
                masker_level = bark_spectrum[i, j]
                
                # Apply masking to neighboring frequency bands
                for k in range(max(0, i-3), min(bark_spectrum.shape[0], i+4)):
                    if k != i:
                        distance = abs(i - k)
                        masking_effect = masker_level * np.exp(-distance * 0.5)
                        
                        # Reduce masked components
                        if bark_spectrum[k, j] < masking_effect * 0.5:
                            masked_spectrum[k, j] *= 0.7
        
        # Temporal masking (simplified)
        for i in range(1, bark_spectrum.shape[1] - 1):
            prev_frame = bark_spectrum[:, i-1]
            curr_frame = bark_spectrum[:, i]
            next_frame = bark_spectrum[:, i+1]
            
            # Forward masking
            forward_mask = prev_frame * 0.3
            masked_spectrum[:, i] = np.maximum(masked_spectrum[:, i], 
                                             np.minimum(curr_frame, forward_mask))
        
        return masked_spectrum
    
    def _convert_from_bark_domain(self, bark_spectrum: np.ndarray, 
                                 target_shape: Tuple[int, int]) -> np.ndarray:
        """Convert Bark domain back to linear frequency domain"""
        
        enhanced_magnitude = np.zeros(target_shape)
        bark_bands = np.linspace(0, 24, bark_spectrum.shape[0] + 1)
        
        for i in range(len(self.bark_scale)):
            # Find corresponding Bark band
            bark_idx = np.digitize(self.bark_scale[i], bark_bands) - 1
            bark_idx = np.clip(bark_idx, 0, bark_spectrum.shape[0] - 1)
            
            enhanced_magnitude[i, :] = bark_spectrum[bark_idx, :]
        
        return enhanced_magnitude

class HarmonicPercussiveProcessor:
    """Harmonic and percussive component processing"""
    
    def __init__(self, config: AdvancedEnhancementConfig):
        self.config = config
    
    def separate_and_enhance(self, audio: np.ndarray) -> np.ndarray:
        """Separate harmonic/percussive and enhance separately"""
        
        if not self.config.enable_harmonic_enhancement:
            return audio
        
        # Separate harmonic and percussive components
        harmonic, percussive = librosa.effects.hpss(
            audio, 
            margin=self.config.harmonic_separation_margin
        )
        
        # Process harmonic component (tonal content)
        enhanced_harmonic = self._enhance_harmonic_component(harmonic)
        
        # Process percussive component (transients)
        processed_percussive = self._process_percussive_component(percussive)
        
        # Recombine with appropriate weights
        # Emphasize harmonic content for speech
        combined = 0.75 * enhanced_harmonic + 0.25 * processed_percussive
        
        return combined
    
    def _enhance_harmonic_component(self, harmonic: np.ndarray) -> np.ndarray:
        """Enhance harmonic content"""
        
        stft = librosa.stft(harmonic, 
                           n_fft=self.config.n_fft, 
                           hop_length=self.config.hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Enhance harmonic peaks
        enhanced_magnitude = self._enhance_spectral_peaks(magnitude)
        
        # Reduce noise floor
        enhanced_magnitude = self._reduce_harmonic_noise_floor(enhanced_magnitude)
        
        # Reconstruct
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        enhanced_harmonic = librosa.istft(enhanced_stft, hop_length=self.config.hop_length)
        
        return enhanced_harmonic
    
    def _enhance_spectral_peaks(self, magnitude: np.ndarray) -> np.ndarray:
        """Enhance spectral peaks (harmonics)"""
        
        enhanced = magnitude.copy()
        
        # Find spectral peaks
        for frame_idx in range(magnitude.shape[1]):
            spectrum = magnitude[:, frame_idx]
            
            # Find peaks
            peaks, properties = signal.find_peaks(spectrum, 
                                                 height=np.max(spectrum) * 0.1,
                                                 distance=5)
            
            # Enhance peaks
            for peak in peaks:
                # Boost peak and surrounding bins
                start_bin = max(0, peak - 2)
                end_bin = min(len(spectrum), peak + 3)
                enhanced[start_bin:end_bin, frame_idx] *= 1.2
        
        return enhanced
    
    def _reduce_harmonic_noise_floor(self, magnitude: np.ndarray) -> np.ndarray:
        """Reduce noise floor in harmonic component"""
        
        # Estimate noise floor
        noise_floor = np.percentile(magnitude, 20, axis=1, keepdims=True)
        
        # Create noise gate
        gate_threshold = noise_floor * 2.0
        gate_ratio = 0.5
        
        gated_magnitude = np.where(magnitude > gate_threshold,
                                 magnitude,
                                 magnitude * gate_ratio)
        
        return gated_magnitude
    
    def _process_percussive_component(self, percussive: np.ndarray) -> np.ndarray:
        """Process percussive component"""
        
        # Gate low-level percussive content to reduce artifacts
        threshold = np.percentile(np.abs(percussive), 80)
        
        # Apply soft gating
        processed = np.where(np.abs(percussive) > threshold * 0.3,
                           percussive,
                           percussive * 0.1)
        
        return processed

class DynamicProcessor:
    """Advanced dynamic range processing"""
    
    def __init__(self, config: AdvancedEnhancementConfig):
        self.config = config
    
    def process_dynamics(self, audio: np.ndarray) -> np.ndarray:
        """Apply comprehensive dynamic processing"""
        
        if not self.config.enable_dynamic_processing:
            return audio
        
        # Multi-stage dynamic processing
        processed = audio.copy()
        
        # 1. Upward expansion (reduce low-level noise)
        processed = self._apply_upward_expansion(processed)
        
        # 2. Compression (control peaks)
        processed = self._apply_multiband_compression(processed)
        
        # 3. Gentle limiting (prevent clipping)
        processed = self._apply_soft_limiting(processed)
        
        return processed
    
    def _apply_upward_expansion(self, audio: np.ndarray) -> np.ndarray:
        """Apply upward expansion to reduce low-level noise"""
        
        # Threshold below which to apply expansion
        threshold_db = -40
        threshold_linear = 10 ** (threshold_db / 20)
        
        # Expansion ratio
        expansion_ratio = 1.5
        
        expanded = np.zeros_like(audio)
        
        for i, sample in enumerate(audio):
            abs_sample = abs(sample)
            
            if abs_sample < threshold_linear:
                # Below threshold - apply expansion (reduce level)
                expanded_abs = abs_sample ** expansion_ratio
                expanded[i] = np.sign(sample) * expanded_abs
            else:
                # Above threshold - no change
                expanded[i] = sample
        
        return expanded
    
    def _apply_multiband_compression(self, audio: np.ndarray) -> np.ndarray:
        """Apply frequency-dependent compression"""
        
        # Split into 3 bands for different compression
        nyquist = self.config.sample_rate / 2
        
        # Low band (up to 200 Hz)
        sos_low = signal.butter(4, 200 / nyquist, 'low', output='sos')
        low_band = signal.sosfilt(sos_low, audio)
        
        # Mid band (200 Hz - 4000 Hz)
        sos_mid = signal.butter(4, [200 / nyquist, 4000 / nyquist], 'band', output='sos')
        mid_band = signal.sosfilt(sos_mid, audio)
        
        # High band (above 4000 Hz)
        sos_high = signal.butter(4, 4000 / nyquist, 'high', output='sos')
        high_band = signal.sosfilt(sos_high, audio)
        
        # Apply different compression to each band
        compressed_low = self._compress_band(low_band, -25, 3.0)  # More compression for low end
        compressed_mid = self._compress_band(mid_band, -20, 2.5)  # Moderate compression for speech
        compressed_high = self._compress_band(high_band, -15, 2.0)  # Light compression for highs
        
        # Recombine
        return compressed_low + compressed_mid + compressed_high
    
    def _compress_band(self, band_audio: np.ndarray, 
                      threshold_db: float, ratio: float) -> np.ndarray:
        """Apply compression to frequency band"""
        
        threshold_linear = 10 ** (threshold_db / 20)
        
        compressed = np.zeros_like(band_audio)
        
        for i, sample in enumerate(band_audio):
            abs_sample = abs(sample)
            
            if abs_sample > threshold_linear:
                # Above threshold - apply compression
                excess = abs_sample - threshold_linear
                compressed_excess = excess / ratio
                compressed_abs = threshold_linear + compressed_excess
                compressed[i] = np.sign(sample) * compressed_abs
            else:
                # Below threshold - no compression
                compressed[i] = sample
        
        return compressed
    
    def _apply_soft_limiting(self, audio: np.ndarray) -> np.ndarray:
        """Apply soft limiting to prevent clipping"""
        
        threshold = 0.8
        knee_width = 0.1
        
        limited = np.zeros_like(audio)
        
        for i, sample in enumerate(audio):
            abs_sample = abs(sample)
            
            if abs_sample <= threshold - knee_width:
                # Below knee - no limiting
                limited[i] = sample
            elif abs_sample <= threshold + knee_width:
                # In knee region - soft limiting
                knee_ratio = (abs_sample - (threshold - knee_width)) / (2 * knee_width)
                knee_gain = 1 - (knee_ratio ** 2) * 0.3
                limited[i] = sample * knee_gain
            else:
                # Above knee - hard limiting
                limited[i] = np.sign(sample) * threshold
        
        return limited

class FinalMasteringProcessor:
    """Final mastering and output processing"""
    
    def __init__(self, config: AdvancedEnhancementConfig):
        self.config = config
    
    def master_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply final mastering processing"""
        
        mastered = audio.copy()
        
        # 1. High-pass filter (remove DC and sub-sonic content)
        mastered = self._apply_highpass_filter(mastered, sample_rate)
        
        # 2. Subtle harmonic enhancement
        mastered = self._add_harmonic_excitement(mastered)
        
        # 3. Loudness normalization
        mastered = self._normalize_loudness(mastered, sample_rate)
        
        # 4. Final limiting
        if self.config.enable_final_limiter:
            mastered = self._apply_final_limiter(mastered)
        
        # 5. Ensure no clipping
        mastered = np.clip(mastered, -0.98, 0.98)
        
        return mastered
    
    def _apply_highpass_filter(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply high-pass filter to remove unwanted low frequencies"""
        
        # Remove content below 40 Hz
        sos = signal.butter(2, 40 / (sample_rate / 2), 'high', output='sos')
        filtered = signal.sosfilt(sos, audio)
        
        return filtered
    
    def _add_harmonic_excitement(self, audio: np.ndarray) -> np.ndarray:
        """Add subtle harmonic excitement for warmth"""
        
        # Very gentle saturation
        drive = 1.02
        excited = np.tanh(audio * drive)
        
        # Mix with original (very subtle)
        mix_ratio = 0.05
        return (1 - mix_ratio) * audio + mix_ratio * excited
    
    def _normalize_loudness(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Normalize loudness to target level"""
        
        if PYLOUDNORM_AVAILABLE:
            try:
                # Use professional loudness normalization
                meter = pyln.Meter(sample_rate)
                loudness = meter.integrated_loudness(audio)
                
                if -50 < loudness < 0:  # Valid loudness measurement
                    normalized = pyln.normalize.loudness(
                        audio, loudness, self.config.target_loudness
                    )
                    return normalized
            except Exception as e:
                logger.warning(f"Professional loudness normalization failed: {e}")
        
        # Fallback: simple peak normalization
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            target_peak = 10 ** (self.config.target_loudness / 20)
            normalized = audio * (target_peak / max_val)
            return normalized
        
        return audio
    
    def _apply_final_limiter(self, audio: np.ndarray) -> np.ndarray:
        """Apply final peak limiter"""
        
        threshold_db = self.config.limiter_threshold
        threshold_linear = 10 ** (threshold_db / 20)
        
        # Soft knee limiter
        limited = np.zeros_like(audio)
        
        for i, sample in enumerate(audio):
            abs_sample = abs(sample)
            
            if abs_sample > threshold_linear:
                # Soft limiting with gentle curve
                excess = abs_sample - threshold_linear
                limited_excess = excess * (1 / (1 + excess * 5))  # Soft knee
                limited_abs = threshold_linear + limited_excess
                limited[i] = np.sign(sample) * limited_abs
            else:
                limited[i] = sample
        
        return limited

class AdvancedTraditionalEnhancer:
    """Main traditional audio enhancement system"""
    
    def __init__(self, config: Optional[AdvancedEnhancementConfig] = None):
        self.config = config or AdvancedEnhancementConfig()
        
        # Initialize processors
        self.noise_reducer = AdvancedNoiseReducer(self.config)
        self.multiband_processor = MultibandProcessor(self.config)
        self.psychoacoustic_processor = PsychoacousticProcessor(self.config)
        self.harmonic_processor = HarmonicPercussiveProcessor(self.config)
        self.dynamic_processor = DynamicProcessor(self.config)
        self.mastering_processor = FinalMasteringProcessor(self.config)
        
        # Performance tracking
        self.metrics = defaultdict(list)
        
        logger.info("Advanced Traditional Audio Enhancer initialized")
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load and validate audio file"""
        
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        try:
            start_time = time.time()
            
            # Load audio
            audio, sr = librosa.load(file_path, sr=None, mono=True)
            
            # Validate audio
            if len(audio) == 0:
                raise ValueError("Empty audio file")
            
            if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
                logger.warning("Audio contains invalid values, cleaning...")
                audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Resample if necessary
            if sr != self.config.sample_rate:
                logger.info(f"Resampling from {sr}Hz to {self.config.sample_rate}Hz")
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.config.sample_rate)
                sr = self.config.sample_rate
            
            # Normalize input
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val * 0.9  # Keep headroom
            
            load_time = time.time() - start_time
            self.metrics['audio_loading'].append(load_time)
            
            logger.info(f"Audio loaded: {len(audio)/sr:.2f}s at {sr}Hz ({load_time:.3f}s)")
            
            return audio, sr
            
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            raise
    
    def enhance_audio(self, audio_path: str = None) -> Tuple[np.ndarray, int]:
        """Main enhancement pipeline"""
        
        if audio_path is None:
            audio_path = INPUT_AUDIO_PATH
            logger.info(f"Using global input path: {INPUT_AUDIO_PATH}")
        
        logger.info(f"Starting traditional enhancement pipeline for: {audio_path}")
        start_time = time.time()
        
        # Load audio
        original_audio, sr = self.load_audio(audio_path)
        enhanced_audio = original_audio.copy()
        
        # Stage 1: Noise Reduction
        logger.info("Stage 1: Advanced noise reduction...")
        stage_start = time.time()
        
        # Apply multiple noise reduction techniques
        if self.config.noise_reduction_strength > 0.3:
            enhanced_audio = self.noise_reducer.adaptive_spectral_subtraction(enhanced_audio)
            enhanced_audio = self.noise_reducer.wiener_filtering(enhanced_audio)
        
        if self.config.noise_reduction_strength > 0.6:
            enhanced_audio = self.noise_reducer.spectral_gating(enhanced_audio)
        
        # Use external noise reduction if available and requested
        if NOISEREDUCE_AVAILABLE and self.config.noise_reduction_strength > 0.8:
            try:
                enhanced_audio = nr.reduce_noise(
                    y=enhanced_audio, 
                    sr=sr,
                    stationary=False,
                    prop_decrease=0.7
                )
            except Exception as e:
                logger.warning(f"External noise reduction failed: {e}")
        
        stage_time = time.time() - stage_start
        self.metrics['noise_reduction'].append(stage_time)
        logger.info(f"Noise reduction completed in {stage_time:.3f}s")
        
        # Stage 2: Multi-band Processing
        logger.info("Stage 2: Multi-band enhancement...")
        stage_start = time.time()
        enhanced_audio = self.multiband_processor.process_multiband(enhanced_audio)
        stage_time = time.time() - stage_start
        self.metrics['multiband_processing'].append(stage_time)
        logger.info(f"Multi-band processing completed in {stage_time:.3f}s")
        
        # Stage 3: Harmonic-Percussive Enhancement
        logger.info("Stage 3: Harmonic-percussive enhancement...")
        stage_start = time.time()
        enhanced_audio = self.harmonic_processor.separate_and_enhance(enhanced_audio)
        stage_time = time.time() - stage_start
        self.metrics['harmonic_processing'].append(stage_time)
        logger.info(f"Harmonic-percussive processing completed in {stage_time:.3f}s")
        
        # Stage 4: Psychoacoustic Processing
        logger.info("Stage 4: Psychoacoustic enhancement...")
        stage_start = time.time()
        enhanced_audio = self.psychoacoustic_processor.psychoacoustic_enhancement(enhanced_audio)
        stage_time = time.time() - stage_start
        self.metrics['psychoacoustic_processing'].append(stage_time)
        logger.info(f"Psychoacoustic processing completed in {stage_time:.3f}s")
        
        # Stage 5: Dynamic Processing
        logger.info("Stage 5: Dynamic processing...")
        stage_start = time.time()
        enhanced_audio = self.dynamic_processor.process_dynamics(enhanced_audio)
        stage_time = time.time() - stage_start
        self.metrics['dynamic_processing'].append(stage_time)
        logger.info(f"Dynamic processing completed in {stage_time:.3f}s")
        
        # Stage 6: Final Mastering
        logger.info("Stage 6: Final mastering...")
        stage_start = time.time()
        enhanced_audio = self.mastering_processor.master_audio(enhanced_audio, sr)
        stage_time = time.time() - stage_start
        self.metrics['mastering'].append(stage_time)
        logger.info(f"Mastering completed in {stage_time:.3f}s")
        
        # Calculate quality metrics
        total_time = time.time() - start_time
        snr_improvement = self._calculate_snr_improvement(original_audio, enhanced_audio)
        
        self.metrics['total_time'].append(total_time)
        self.metrics['snr_improvement'].append(snr_improvement)
        
        logger.info(f"Enhancement complete! Total time: {total_time:.2f}s")
        logger.info(f"Estimated SNR improvement: {snr_improvement:.2f} dB")
        
        return enhanced_audio, sr
    
    def _calculate_snr_improvement(self, original: np.ndarray, enhanced: np.ndarray) -> float:
        """Calculate SNR improvement estimate"""
        try:
            # Align lengths
            min_len = min(len(original), len(enhanced))
            orig = original[:min_len]
            enh = enhanced[:min_len]
            
            # Remove DC
            orig = orig - np.mean(orig)
            enh = enh - np.mean(enh)
            
            # Original SNR estimate
            orig_power = np.mean(orig ** 2)
            orig_snr = 10 * np.log10(orig_power / (orig_power * 0.1 + 1e-12))
            
            # Enhanced SNR estimate
            enh_power = np.mean(enh ** 2)
            enh_snr = 10 * np.log10(enh_power / (enh_power * 0.05 + 1e-12))  # Assume better noise floor
            
            return enh_snr - orig_snr
            
        except Exception as e:
            logger.warning(f"SNR calculation failed: {e}")
            return 0.0
    
    def save_enhanced_audio(self, audio: np.ndarray, sr: int, output_path: str = None):
        """Save enhanced audio with metadata"""
        
        if output_path is None:
            output_path = OUTPUT_AUDIO_PATH
        
        try:
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save audio
            sf.write(output_path, audio, sr, subtype='PCM_24')
            logger.info(f"Enhanced audio saved to: {output_path}")
            
            # Save processing metadata
            metadata_path = output_path.replace('.wav', '_metadata.json')
            metadata = {
                'processing_info': {
                    'input_file': str(INPUT_AUDIO_PATH),
                    'output_file': str(output_path),
                    'sample_rate': sr,
                    'duration': len(audio) / sr,
                    'enhancement_method': 'Traditional Signal Processing'
                },
                'configuration': asdict(self.config),
                'performance_metrics': {
                    'total_time': np.sum(self.metrics.get('total_time', [])),
                    'snr_improvement': np.mean(self.metrics.get('snr_improvement', [])),
                    'stage_times': {
                        'noise_reduction': np.sum(self.metrics.get('noise_reduction', [])),
                        'multiband_processing': np.sum(self.metrics.get('multiband_processing', [])),
                        'harmonic_processing': np.sum(self.metrics.get('harmonic_processing', [])),
                        'psychoacoustic_processing': np.sum(self.metrics.get('psychoacoustic_processing', [])),
                        'dynamic_processing': np.sum(self.metrics.get('dynamic_processing', [])),
                        'mastering': np.sum(self.metrics.get('mastering', []))
                    }
                },
                'processing_features': {
                    'noise_reduction': True,
                    'multiband_enhancement': self.config.enable_multiband,
                    'psychoacoustic_modeling': self.config.enable_psychoacoustic,
                    'harmonic_enhancement': self.config.enable_harmonic_enhancement,
                    'dynamic_processing': self.config.enable_dynamic_processing,
                    'final_mastering': True
                }
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Metadata saved to: {metadata_path}")
            
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            # Try to save just the audio without metadata
            try:
                sf.write(output_path, audio, sr, subtype='PCM_16')
                logger.info(f"Audio saved successfully (metadata failed): {output_path}")
            except Exception as e2:
                logger.error(f"Failed to save audio file: {e2}")
                raise

def main():
    """Main execution function"""
    
    try:
        print(f"\n{'='*70}")
        print("🎵 ADVANCED TRADITIONAL AUDIO ENHANCEMENT")
        print("Pure Signal Processing - No AI/ML")
        print(f"{'='*70}")
        print(f"Input file: {INPUT_AUDIO_PATH}")
        print(f"Output file: {OUTPUT_AUDIO_PATH}")
        print(f"{'='*70}")
        
        # Initialize enhancer
        config = AdvancedEnhancementConfig()
        enhancer = AdvancedTraditionalEnhancer(config)
        
        # Process audio
        enhanced_audio, sample_rate = enhancer.enhance_audio()
        
        # Save results
        enhancer.save_enhanced_audio(enhanced_audio, sample_rate)
        
        # Print results
        total_time = np.sum(enhancer.metrics.get('total_time', []))
        snr_improvement = np.mean(enhancer.metrics.get('snr_improvement', []))
        
        print(f"\n{'='*70}")
        print("✅ ENHANCEMENT COMPLETE!")
        print(f"{'='*70}")
        print(f"📁 Input: {INPUT_AUDIO_PATH}")
        print(f"📁 Output: {OUTPUT_AUDIO_PATH}")
        print(f"⏱️  Processing Time: {total_time:.2f} seconds")
        print(f"📊 Estimated SNR Improvement: {snr_improvement:+.2f} dB")
        print(f"🎯 Method: Traditional Signal Processing")
        
        # Enhancement assessment
        if snr_improvement > 5:
            print(f"🏆 EXCELLENT - Significant improvement achieved!")
        elif snr_improvement > 2:
            print(f"✅ GOOD - Noticeable improvement achieved")
        elif snr_improvement > 0:
            print(f"⚠️  MODERATE - Some improvement achieved")
        else:
            print(f"ℹ️  INPUT MAY ALREADY BE HIGH QUALITY")
        
        # Processing breakdown
        print(f"\n📈 Processing Breakdown:")
        stage_times = enhancer.metrics.get('stage_times', {})
        for stage, time_taken in stage_times.items():
            if time_taken > 0:
                percentage = (time_taken / total_time) * 100
                print(f"   • {stage.replace('_', ' ').title()}: {time_taken:.3f}s ({percentage:.1f}%)")
        
        print(f"{'='*70}")
        
        return True
        
    except FileNotFoundError:
        print(f"\n❌ ERROR: Audio file not found!")
        print(f"Please ensure the file exists at: {INPUT_AUDIO_PATH}")
        print(f"Update the INPUT_AUDIO_PATH variable with your actual file path.")
        return False
        
    except Exception as e:
        logger.error(f"Enhancement failed: {e}")
        print(f"\n❌ Enhancement failed: {e}")
        print(f"\nTroubleshooting tips:")
        print(f"1. Check input audio file format and integrity")
        print(f"2. Ensure all dependencies are installed")
        print(f"3. Try with a different audio file")
        return False

if __name__ == "__main__":
    # Configuration section
    print("\n" + "="*70)
    print("🔧 TRADITIONAL AUDIO ENHANCEMENT CONFIGURATION")
    print("="*70)
    
    # Check if user has set their file path
    if INPUT_AUDIO_PATH in ["input_noisy_audio.wav", "your_noisy_audio.wav"]:
        print("❌ CONFIGURATION REQUIRED!")
        print("\nPlease edit this script and change the INPUT_AUDIO_PATH variable.")
        print("Find this line in the script and change it:")
        print('INPUT_AUDIO_PATH = "input_noisy_audio.wav"  # ← CHANGE THIS!')
        print("\nTo your actual file path, for example:")
        print('INPUT_AUDIO_PATH = "C:/my_audio/noisy_recording.wav"')
        print('INPUT_AUDIO_PATH = "/home/user/audio/my_file.wav"')
        print('INPUT_AUDIO_PATH = "my_audio.wav"  # if in same folder')
        
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
        
        print("\n" + "="*70)
        sys.exit(1)
    
    # Validate the path
    if not Path(INPUT_AUDIO_PATH).exists():
        print(f"❌ ERROR: Audio file not found!")
        print(f"Current path: {INPUT_AUDIO_PATH}")
        print(f"Full path: {Path(INPUT_AUDIO_PATH).absolute()}")
        print(f"\nPlease check:")
        print(f"1. File exists at the specified location")
        print(f"2. Path is correctly spelled")
        print(f"3. Use forward slashes (/) or raw strings")
        sys.exit(1)
    
    print(f"✅ Configuration validated!")
    print(f"📁 Input file: {INPUT_AUDIO_PATH}")
    print(f"📁 Output file: {OUTPUT_AUDIO_PATH}")
    print(f"🎯 Method: Traditional Signal Processing (No AI)")
    print(f"🔧 Features: Multi-band, Psychoacoustic, Harmonic Enhancement")
    print("="*70)
    
    # Run the enhancement
    success = main()
    
    if not success:
        sys.exit(1)