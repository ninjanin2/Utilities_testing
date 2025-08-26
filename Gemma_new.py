# -*- coding: utf-8 -*-
"""
COMPREHENSIVE ADVANCED SPEECH ENHANCEMENT & TRANSCRIPTION SYSTEM
================================================================

COMPREHENSIVE PREPROCESSING METHODS INCLUDED:
- Spectral Domain Methods (Spectral Subtraction, MBSS, Wiener, MMSE-STSA, MMSE-LSA, OM-LSA)
- Frequency Domain Filtering (Low/High/Band-pass, Adaptive Filtering)
- Time-Frequency Domain Processing (DA-STFT, FFT with Hanning, Frame-Based, TF Masking)
- Advanced Normalization (Z-score Min-Max, Dynamic Range Compression, Noise Gating)
- Statistical Estimators and Signal Subspace Approaches
- Voice Activity Detection and Speech Activity Detection
- Noise Profile Analysis and SNR Enhancement
- Temporal Processing and Frame Averaging
- Real-time Adaptive Algorithms
- Quality Assessment Metrics

Author: Comprehensive AI Audio Processing System
Version: Complete Enhancement 14.0
"""

import os
import gc
import torch
import librosa
import gradio as gr
from transformers import Gemma3nForConditionalGeneration, Gemma3nProcessor, BitsAndBytesConfig
import numpy as np
from typing import Optional, Tuple, Dict, List, Union
import time
import sys
import threading
import queue
import tempfile
import soundfile as sf
from scipy import signal
from scipy.signal import butter, filtfilt, lfilter, wiener, hanning
from scipy.ndimage import median_filter, gaussian_filter1d
from scipy.stats import zscore
from scipy.linalg import svd, pinv
import noisereduce as nr
import datetime
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import psutil
import re
import nltk
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# CRITICAL FIX: Disable torch dynamo
torch._dynamo.config.disable = True
print("🔧 CRITICAL FIX: torch._dynamo compilation disabled")

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
    except:
        pass

# --- COMPREHENSIVE ENHANCEMENT CONFIGURATION ---
MODEL_PATH = "/path/to/your/local/gemma-3n-e4b-it"  # UPDATE THIS PATH

# Enhanced settings for comprehensive processing
CHUNK_SECONDS = 12
OVERLAP_SECONDS = 2
SAMPLE_RATE = 16000
CHUNK_TIMEOUT = 75
MAX_RETRIES = 1
PROCESSING_THREADS = 1

# COMPREHENSIVE: All enhancement methods enabled
SPECTRAL_SUBTRACTION_ENABLED = True
MULTI_BAND_SPECTRAL_SUBTRACTION = True
WIENER_FILTERING_ENABLED = True
MMSE_STSA_ENABLED = True
MMSE_LSA_ENABLED = True
OM_LSA_ENABLED = True
ADAPTIVE_FILTERING_ENABLED = True
DA_STFT_ENABLED = True
TIME_FREQUENCY_MASKING = True
ADVANCED_VAD_ENABLED = True
Z_SCORE_NORMALIZATION = True
DYNAMIC_RANGE_COMPRESSION = True
NOISE_GATING_ENABLED = True
TEMPORAL_SMOOTHING = True
FRAME_AVERAGING = True
SIGNAL_SUBSPACE_APPROACH = True
NOISE_PROFILE_ANALYSIS = True
SNR_ENHANCEMENT = True

# Memory settings
MIN_FREE_MEMORY_GB = 0.3
MEMORY_SAFETY_MARGIN = 0.1
CHECK_MEMORY_FREQUENCY = 5

# Translation settings
MAX_TRANSLATION_CHUNK_SIZE = 1000
SENTENCE_OVERLAP = 1
MIN_CHUNK_SIZE = 100

# Expanded language support
SUPPORTED_LANGUAGES = {
    "🌍 Auto-detect": "auto",
    "🇺🇸 English": "en", "🇪🇸 Spanish": "es", "🇫🇷 French": "fr", "🇩🇪 German": "de",
    "🇮🇹 Italian": "it", "🇵🇹 Portuguese": "pt", "🇷🇺 Russian": "ru", "🇨🇳 Chinese": "zh",
    "🇯🇵 Japanese": "ja", "🇰🇷 Korean": "ko", "🇸🇦 Arabic": "ar", "🇮🇳 Hindi": "hi",
    "🇳🇱 Dutch": "nl", "🇸🇪 Swedish": "sv", "🇳🇴 Norwegian": "no", "🇩🇰 Danish": "da",
    "🇫🇮 Finnish": "fi", "🇵🇱 Polish": "pl", "🇹🇷 Turkish": "tr",
    "🇮🇳 Bengali": "bn", "🇮🇳 Tamil": "ta", "🇮🇳 Telugu": "te", "🇮🇳 Gujarati": "gu",
    "🇮🇳 Marathi": "mr", "🇮🇳 Urdu": "ur", "🇮🇳 Kannada": "kn", "🇮🇳 Malayalam": "ml",
    "🇮🇳 Punjabi": "pa", "🇮🇳 Odia": "or", "🇮🇳 Assamese": "as", "🇮🇳 Sindhi": "sd",
    "🇱🇰 Sinhala": "si", "🇳🇵 Nepali": "ne", "🇵🇰 Pashto": "ps",
    "🇮🇷 Persian/Farsi": "fa", "🇦🇫 Dari": "prs", "🇹🇯 Tajik": "tg", "🇺🇿 Uzbek": "uz",
    "🇰🇿 Kazakh": "kk", "🇰🇬 Kyrgyz": "ky", "🇹🇲 Turkmen": "tk", "🇦🇿 Azerbaijani": "az",
    "🇦🇲 Armenian": "hy", "🇬🇪 Georgian": "ka", "🇮🇱 Hebrew": "he",
    "🇲🇲 Burmese/Myanmar": "my", "🇹🇭 Thai": "th", "🇻🇳 Vietnamese": "vi",
    "🇮🇩 Indonesian": "id", "🇲🇾 Malay": "ms", "🇵🇭 Filipino/Tagalog": "tl",
    "🇰🇭 Khmer/Cambodian": "km", "🇱🇦 Lao": "lo", "🇸🇬 Chinese (Singapore)": "zh-sg",
    "🏔️ Tibetan": "bo", "🇧🇹 Dzongkha": "dz", "🏔️ Sherpa": "xsr", "🏔️ Tamang": "taj",
}

# Environment optimization
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

class TimeoutError(Exception):
    """Custom timeout exception"""
    pass

class ComprehensiveVoiceActivityDetector:
    """COMPREHENSIVE: Advanced multi-feature voice activity detection"""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.frame_length = 1024
        self.hop_length = 256
        
    def detect_voice_activity(self, audio: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Comprehensive VAD using multiple acoustic features and statistical methods"""
        try:
            print("🎤 COMPREHENSIVE voice activity detection...")
            
            # Energy-based features
            frame_energy = librosa.feature.rms(y=audio, frame_length=self.frame_length, hop_length=self.hop_length)[0]
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate, hop_length=self.hop_length)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate, hop_length=self.hop_length)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate, hop_length=self.hop_length)[0]
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sample_rate, hop_length=self.hop_length)
            spectral_flatness = librosa.feature.spectral_flatness(y=audio, hop_length=self.hop_length)[0]
            
            # Temporal features
            zcr = librosa.feature.zero_crossing_rate(audio, frame_length=self.frame_length, hop_length=self.hop_length)[0]
            
            # MFCC features (comprehensive)
            mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13, hop_length=self.hop_length)
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate, hop_length=self.hop_length)
            
            # Tonnetz features
            tonnetz = librosa.feature.tonnetz(y=audio, sr=self.sample_rate)
            
            # Comprehensive statistical thresholding
            energy_threshold = np.percentile(frame_energy, 15)  # Very conservative
            centroid_threshold = np.percentile(spectral_centroids, 10)
            rolloff_threshold = np.percentile(spectral_rolloff, 20)
            bandwidth_threshold = np.percentile(spectral_bandwidth, 25)
            contrast_threshold = np.percentile(np.mean(spectral_contrast, axis=0), 20)
            flatness_threshold = np.percentile(spectral_flatness, 70)
            zcr_threshold = np.percentile(zcr, 85)
            mfcc_threshold = np.percentile(np.mean(mfcc, axis=0), 20)
            
            # Multi-criteria decision with comprehensive weighting
            voice_criteria = [
                frame_energy > energy_threshold,  # Weight: 0.25
                spectral_centroids > centroid_threshold,  # Weight: 0.20
                spectral_rolloff > rolloff_threshold,  # Weight: 0.15
                spectral_bandwidth > bandwidth_threshold,  # Weight: 0.10
                np.mean(spectral_contrast, axis=0) > contrast_threshold,  # Weight: 0.10
                spectral_flatness < flatness_threshold,  # Weight: 0.05
                zcr < zcr_threshold,  # Weight: 0.05
                np.mean(mfcc, axis=0) > mfcc_threshold,  # Weight: 0.10
            ]
            
            # Comprehensive weighted voting
            weights = [0.25, 0.20, 0.15, 0.10, 0.10, 0.05, 0.05, 0.10]
            voice_scores = np.zeros(len(frame_energy))
            
            for criterion, weight in zip(voice_criteria, weights):
                voice_scores += criterion.astype(float) * weight
            
            # Adaptive threshold based on signal statistics
            voice_activity = voice_scores > 0.35  # Conservative threshold
            
            # Advanced smoothing with morphological operations
            voice_activity = median_filter(voice_activity.astype(float), size=7) > 0.3
            
            # Comprehensive statistics
            voice_percentage = np.mean(voice_activity) * 100
            stats = {
                'voice_percentage': voice_percentage,
                'avg_energy': np.mean(frame_energy),
                'avg_spectral_centroid': np.mean(spectral_centroids),
                'avg_spectral_rolloff': np.mean(spectral_rolloff),
                'avg_spectral_bandwidth': np.mean(spectral_bandwidth),
                'avg_spectral_contrast': np.mean(spectral_contrast),
                'avg_spectral_flatness': np.mean(spectral_flatness),
                'avg_zcr': np.mean(zcr),
                'avg_mfcc': np.mean(mfcc),
                'voice_score': np.mean(voice_scores),
                'snr_estimate': self.estimate_snr(audio, voice_activity)
            }
            
            return voice_activity, stats
            
        except Exception as e:
            print(f"❌ Comprehensive VAD failed: {e}")
            return np.ones(len(audio) // self.hop_length, dtype=bool), {}
    
    def estimate_snr(self, audio: np.ndarray, voice_activity: np.ndarray) -> float:
        """Estimate SNR using voice activity detection"""
        try:
            hop_length = 256
            vad_expanded = np.repeat(voice_activity, hop_length)
            
            if len(vad_expanded) > len(audio):
                vad_expanded = vad_expanded[:len(audio)]
            elif len(vad_expanded) < len(audio):
                vad_expanded = np.pad(vad_expanded, (0, len(audio) - len(vad_expanded)), mode='edge')
            
            voice_regions = vad_expanded.astype(bool)
            noise_regions = ~voice_regions
            
            if np.any(voice_regions) and np.any(noise_regions):
                signal_power = np.mean(audio[voice_regions]**2)
                noise_power = np.mean(audio[noise_regions]**2)
                
                if noise_power > 0:
                    snr = 10 * np.log10(signal_power / noise_power)
                    return snr
            
            return 20.0  # Default moderate SNR
        except:
            return 20.0

class ComprehensiveSpeechEnhancer:
    """COMPREHENSIVE: All-in-one speech enhancement with every available technique"""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.vad = ComprehensiveVoiceActivityDetector(sample_rate)
        self.frame_size = 1024
        self.hop_size = 256
        self.scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        print(f"🚀 COMPREHENSIVE Speech Enhancer initialized for {sample_rate}Hz")
        print("✅ ALL enhancement methods loaded and ready")
    
    # SPECTRAL DOMAIN METHODS
    
    def spectral_subtraction(self, audio: np.ndarray, alpha: float = 2.0, beta: float = 0.01) -> np.ndarray:
        """Classical spectral subtraction"""
        try:
            print("🔬 Applying spectral subtraction...")
            
            # Compute STFT
            stft = librosa.stft(audio, n_fft=2048, hop_length=512)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Estimate noise from first few frames
            noise_frames = magnitude[:, :10]
            noise_estimate = np.mean(noise_frames, axis=1, keepdims=True)
            
            # Spectral subtraction
            enhanced_magnitude = magnitude - alpha * noise_estimate
            
            # Apply spectral floor
            enhanced_magnitude = np.maximum(enhanced_magnitude, beta * magnitude)
            
            # Reconstruct
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft, hop_length=512)
            
            return enhanced_audio.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Spectral subtraction failed: {e}")
            return audio
    
    def multi_band_spectral_subtraction(self, audio: np.ndarray) -> np.ndarray:
        """Multi-Band Spectral Subtraction (MBSS)"""
        try:
            print("🔬 Applying Multi-Band Spectral Subtraction...")
            
            # Define frequency bands
            bands = [
                (0, 500, 2.5, 0.02),     # Low frequencies
                (500, 1500, 2.0, 0.01),  # Mid-low frequencies
                (1500, 4000, 1.8, 0.005),# Mid frequencies (speech)
                (4000, 8000, 2.2, 0.015) # High frequencies
            ]
            
            # Compute STFT
            stft = librosa.stft(audio, n_fft=2048, hop_length=512)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Get frequency bins
            freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=2048)
            
            enhanced_magnitude = magnitude.copy()
            
            for low_freq, high_freq, alpha, beta in bands:
                # Find frequency bin indices
                low_bin = np.argmin(np.abs(freqs - low_freq))
                high_bin = np.argmin(np.abs(freqs - high_freq))
                
                # Extract band
                band_magnitude = magnitude[low_bin:high_bin, :]
                
                # Estimate noise for this band
                band_noise = np.mean(band_magnitude[:, :10], axis=1, keepdims=True)
                
                # Apply spectral subtraction for this band
                enhanced_band = band_magnitude - alpha * band_noise
                enhanced_band = np.maximum(enhanced_band, beta * band_magnitude)
                
                # Update enhanced magnitude
                enhanced_magnitude[low_bin:high_bin, :] = enhanced_band
            
            # Reconstruct
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft, hop_length=512)
            
            return enhanced_audio.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Multi-Band Spectral Subtraction failed: {e}")
            return audio
    
    def wiener_filtering(self, audio: np.ndarray) -> np.ndarray:
        """Advanced Wiener filtering with optimal parameters"""
        try:
            print("🔧 Applying advanced Wiener filtering...")
            
            # Estimate noise power using VAD
            vad_result, _ = self.vad.detect_voice_activity(audio)
            hop_length = 256
            vad_expanded = np.repeat(vad_result, hop_length)
            
            if len(vad_expanded) > len(audio):
                vad_expanded = vad_expanded[:len(audio)]
            elif len(vad_expanded) < len(audio):
                vad_expanded = np.pad(vad_expanded, (0, len(audio) - len(vad_expanded)), mode='edge')
            
            noise_regions = ~vad_expanded.astype(bool)
            
            if np.any(noise_regions):
                noise_power = np.var(audio[noise_regions])
            else:
                noise_power = np.var(audio) * 0.1
            
            # Apply Wiener filter in segments
            segment_length = 4096
            overlap = 1024
            enhanced_audio = np.zeros_like(audio)
            
            for i in range(0, len(audio) - segment_length + 1, segment_length - overlap):
                segment = audio[i:i + segment_length]
                
                # Apply Wiener filter
                enhanced_segment = wiener(segment, noise=noise_power)
                
                # Overlap-add
                if i == 0:
                    enhanced_audio[i:i + segment_length] = enhanced_segment
                else:
                    # Blend overlapping regions
                    blend_start = i
                    blend_end = i + overlap
                    
                    # Linear blending
                    alpha = np.linspace(0, 1, overlap)
                    enhanced_audio[blend_start:blend_end] = (
                        (1 - alpha) * enhanced_audio[blend_start:blend_end] +
                        alpha * enhanced_segment[:overlap]
                    )
                    enhanced_audio[blend_end:i + segment_length] = enhanced_segment[overlap:]
            
            return enhanced_audio.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Wiener filtering failed: {e}")
            return audio
    
    def mmse_stsa_estimator(self, audio: np.ndarray) -> np.ndarray:
        """MMSE Short-Time Spectral Amplitude Estimator"""
        try:
            print("🔬 Applying MMSE-STSA estimator...")
            
            # Compute STFT
            stft = librosa.stft(audio, n_fft=1024, hop_length=256)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Estimate noise power spectrum
            noise_magnitude = np.mean(magnitude[:, :10], axis=1, keepdims=True)
            noise_power = noise_magnitude ** 2
            
            # Compute a priori SNR (using decision-directed approach)
            alpha = 0.98  # Smoothing factor
            signal_power = magnitude ** 2
            
            # Initialize a priori SNR
            gamma_k = np.maximum(signal_power / noise_power - 1, 0.1)
            
            # MMSE-STSA gain function
            nu_k = gamma_k / (1 + gamma_k)
            
            # Bessel function approximation for computational efficiency
            def modified_bessel_i0(x):
                """Modified Bessel function of the first kind, order 0"""
                return np.exp(x) / np.sqrt(2 * np.pi * x) * (1 + 1/(8*x))
            
            def modified_bessel_i1(x):
                """Modified Bessel function of the first kind, order 1"""
                return modified_bessel_i0(x) * (1 - 1/(2*x))
            
            # MMSE-STSA gain
            v_k = nu_k * gamma_k / (1 + gamma_k)
            
            # Avoid numerical issues
            v_k = np.clip(v_k, 0.001, 10)
            
            # Simplified gain function (avoiding complex Bessel functions)
            G_k = nu_k * np.exp(-v_k/2) * ((1 + v_k) * modified_bessel_i0(v_k/2) + v_k * modified_bessel_i1(v_k/2))
            
            # Apply gain
            enhanced_magnitude = G_k * magnitude
            
            # Reconstruct
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft, hop_length=256)
            
            return enhanced_audio.astype(np.float32)
            
        except Exception as e:
            print(f"❌ MMSE-STSA failed: {e}")
            return self.spectral_subtraction(audio)  # Fallback
    
    def mmse_lsa_estimator(self, audio: np.ndarray) -> np.ndarray:
        """MMSE Log-Spectral Amplitude Estimator"""
        try:
            print("🔬 Applying MMSE-LSA estimator...")
            
            # Compute STFT
            stft = librosa.stft(audio, n_fft=1024, hop_length=256)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Convert to log domain
            log_magnitude = np.log(magnitude + 1e-10)
            
            # Estimate noise power spectrum
            noise_log_magnitude = np.mean(log_magnitude[:, :10], axis=1, keepdims=True)
            
            # Compute SNR in log domain
            snr = log_magnitude - noise_log_magnitude
            
            # MMSE-LSA gain function (simplified)
            alpha = 2.0
            beta = 0.1
            
            # Gain function
            G = np.exp(alpha * snr / (1 + np.exp(alpha * snr)))
            G = np.maximum(G, beta)
            
            # Apply gain in log domain
            enhanced_log_magnitude = log_magnitude + np.log(G + 1e-10)
            
            # Convert back to linear domain
            enhanced_magnitude = np.exp(enhanced_log_magnitude)
            
            # Reconstruct
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft, hop_length=256)
            
            return enhanced_audio.astype(np.float32)
            
        except Exception as e:
            print(f"❌ MMSE-LSA failed: {e}")
            return self.spectral_subtraction(audio)  # Fallback
    
    def om_lsa_estimator(self, audio: np.ndarray) -> np.ndarray:
        """Optimally-Modified Log-Spectral Amplitude (OM-LSA) Estimator"""
        try:
            print("🔬 Applying OM-LSA estimator...")
            
            # Compute STFT
            stft = librosa.stft(audio, n_fft=1024, hop_length=256)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Parameters
            alpha_d = 0.95  # Decision-directed parameter
            alpha_s = 0.9   # SPP smoothing parameter
            beta = 0.005    # Minimum gain
            
            # Initialize
            num_frames = magnitude.shape[1]
            num_bins = magnitude.shape[0]
            
            # Estimate noise power spectrum
            noise_power = np.mean(magnitude[:, :10] ** 2, axis=1, keepdims=True)
            
            # Initialize variables
            gamma_k = np.zeros_like(magnitude)  # A posteriori SNR
            xi_k = np.ones_like(magnitude)      # A priori SNR
            p_k = np.zeros_like(magnitude)      # Speech presence probability
            
            enhanced_magnitude = np.zeros_like(magnitude)
            
            for frame in range(num_frames):
                # A posteriori SNR
                gamma_k[:, frame:frame+1] = magnitude[:, frame:frame+1] ** 2 / noise_power
                
                if frame > 0:
                    # Decision-directed a priori SNR estimation
                    xi_k_dd = alpha_d * (enhanced_magnitude[:, frame-1:frame] ** 2 / noise_power) + \
                             (1 - alpha_d) * np.maximum(gamma_k[:, frame:frame+1] - 1, 0.1)
                    xi_k[:, frame:frame+1] = xi_k_dd
                
                # Speech presence probability (simplified)
                v_k = gamma_k[:, frame:frame+1] * xi_k[:, frame:frame+1] / (1 + xi_k[:, frame:frame+1])
                
                # Simplified SPP calculation
                p_k[:, frame:frame+1] = 1 / (1 + np.exp(-2 * (v_k - 1)))
                
                # OM-LSA gain function
                G_k = xi_k[:, frame:frame+1] / (1 + xi_k[:, frame:frame+1])
                
                # Apply speech presence probability
                G_k = p_k[:, frame:frame+1] * G_k + (1 - p_k[:, frame:frame+1]) * beta
                
                # Ensure minimum gain
                G_k = np.maximum(G_k, beta)
                
                # Apply gain
                enhanced_magnitude[:, frame:frame+1] = G_k * magnitude[:, frame:frame+1]
            
            # Reconstruct
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft, hop_length=256)
            
            return enhanced_audio.astype(np.float32)
            
        except Exception as e:
            print(f"❌ OM-LSA failed: {e}")
            return self.mmse_stsa_estimator(audio)  # Fallback
    
    # FREQUENCY DOMAIN FILTERING
    
    def comprehensive_frequency_filtering(self, audio: np.ndarray) -> np.ndarray:
        """Comprehensive frequency domain filtering with proper normalization"""
        try:
            print("🎵 Applying comprehensive frequency filtering...")
            
            # FIXED: Ensure cutoff frequencies are within valid range
            nyquist = self.sample_rate / 2.0
            
            # High-pass filter (remove low-frequency noise)
            high_cutoff = 85  # Hz
            if high_cutoff >= nyquist:
                high_cutoff = nyquist * 0.05  # 5% of Nyquist
            
            high_b, high_a = butter(4, high_cutoff, btype='high', fs=self.sample_rate)
            audio = filtfilt(high_b, high_a, audio)
            
            # Low-pass filter (remove high-frequency noise)
            low_cutoff = 7900  # Hz (FIXED: Less than fs/2 = 8000)
            if low_cutoff >= nyquist:
                low_cutoff = nyquist * 0.95  # 95% of Nyquist
            
            low_b, low_a = butter(4, low_cutoff, btype='low', fs=self.sample_rate)
            audio = filtfilt(low_b, low_a, audio)
            
            return audio.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Frequency filtering failed: {e}")
            return audio
    
    def adaptive_filtering(self, audio: np.ndarray) -> np.ndarray:
        """Adaptive filtering with LMS algorithm"""
        try:
            print("🔧 Applying adaptive filtering...")
            
            # Parameters
            filter_length = 32
            mu = 0.01  # Step size
            
            # Initialize adaptive filter
            w = np.zeros(filter_length)
            filtered_audio = np.zeros_like(audio)
            
            # Apply adaptive filter
            for n in range(filter_length, len(audio)):
                # Input vector
                x = audio[n-filter_length:n][::-1]  # Reversed for convolution
                
                # Filter output
                y = np.dot(w, x)
                filtered_audio[n] = y
                
                # Error signal (use delayed input as reference)
                d = audio[n-1] if n > 0 else 0
                e = d - y
                
                # Update filter weights (LMS)
                w = w + mu * e * x
            
            # Copy initial samples
            filtered_audio[:filter_length] = audio[:filter_length]
            
            return filtered_audio.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Adaptive filtering failed: {e}")
            return audio
    
    # TIME-FREQUENCY DOMAIN PROCESSING
    
    def da_stft_processing(self, audio: np.ndarray) -> np.ndarray:
        """Differentiable Adaptive Short-Time Fourier Transform"""
        try:
            print("🔬 Applying DA-STFT processing...")
            
            # Adaptive window selection based on signal characteristics
            frame_energy = librosa.feature.rms(y=audio, frame_length=1024, hop_length=512)[0]
            avg_energy = np.mean(frame_energy)
            
            # Select window size adaptively
            if avg_energy > 0.1:
                n_fft = 2048  # High resolution for strong signals
            elif avg_energy > 0.05:
                n_fft = 1024  # Medium resolution
            else:
                n_fft = 512   # Low resolution for weak signals
            
            hop_length = n_fft // 4
            
            # Apply adaptive STFT with Hanning window
            window = hanning(n_fft)
            stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, window=window)
            
            # Adaptive spectral processing
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Spectral enhancement based on energy distribution
            enhanced_magnitude = magnitude.copy()
            
            # Frequency-dependent processing
            freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=n_fft)
            
            for i, freq in enumerate(freqs):
                if 85 <= freq <= 4000:  # Speech frequency range
                    enhanced_magnitude[i, :] *= 1.1  # Slight enhancement
                elif freq > 6000:  # High frequency noise
                    enhanced_magnitude[i, :] *= 0.8  # Slight suppression
            
            # Reconstruct
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft, hop_length=hop_length, window=window)
            
            return enhanced_audio.astype(np.float32)
            
        except Exception as e:
            print(f"❌ DA-STFT processing failed: {e}")
            return audio
    
    def time_frequency_masking(self, audio: np.ndarray) -> np.ndarray:
        """Time-Frequency masking for noise suppression"""
        try:
            print("🎭 Applying time-frequency masking...")
            
            # Compute STFT
            stft = librosa.stft(audio, n_fft=1024, hop_length=256)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Create ideal binary mask based on SNR
            noise_estimate = np.mean(magnitude[:, :10], axis=1, keepdims=True)
            snr = magnitude / (noise_estimate + 1e-10)
            
            # Binary mask (1 for speech, 0 for noise)
            threshold = 2.0  # SNR threshold
            binary_mask = (snr > threshold).astype(float)
            
            # Smooth the mask to avoid artifacts
            from scipy.ndimage import gaussian_filter
            smooth_mask = gaussian_filter(binary_mask, sigma=1.0)
            
            # Apply soft masking
            soft_mask = smooth_mask * 0.9 + 0.1  # Ensure minimum gain
            
            # Apply mask
            enhanced_magnitude = magnitude * soft_mask
            
            # Reconstruct
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft, hop_length=256)
            
            return enhanced_audio.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Time-frequency masking failed: {e}")
            return audio
    
    def frame_based_processing(self, audio: np.ndarray) -> np.ndarray:
        """Frame-based processing with overlap-add"""
        try:
            print("📊 Applying frame-based processing...")
            
            frame_length = 1024
            hop_length = 256
            overlap = frame_length - hop_length
            
            # Window function
            window = hanning(frame_length)
            
            # Initialize output
            enhanced_audio = np.zeros_like(audio)
            
            # Process frames
            for i in range(0, len(audio) - frame_length + 1, hop_length):
                # Extract frame
                frame = audio[i:i + frame_length] * window
                
                # Process frame (spectral enhancement)
                frame_fft = np.fft.fft(frame)
                magnitude = np.abs(frame_fft)
                phase = np.angle(frame_fft)
                
                # Simple spectral enhancement
                enhanced_magnitude = magnitude ** 0.9  # Slight compression
                
                # Reconstruct frame
                enhanced_fft = enhanced_magnitude * np.exp(1j * phase)
                enhanced_frame = np.real(np.fft.ifft(enhanced_fft))
                
                # Apply window and overlap-add
                enhanced_frame *= window
                enhanced_audio[i:i + frame_length] += enhanced_frame
            
            return enhanced_audio.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Frame-based processing failed: {e}")
            return audio
    
    # PREPROCESSING AND NORMALIZATION TECHNIQUES
    
    def z_score_min_max_normalization(self, audio: np.ndarray) -> np.ndarray:
        """Z-score Min-Max normalization for enhanced feature extraction"""
        try:
            print("📊 Applying Z-score Min-Max normalization...")
            
            # Z-score normalization (standardization)
            audio_zscore = zscore(audio)
            
            # Handle NaN values
            audio_zscore = np.nan_to_num(audio_zscore, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Min-Max normalization
            audio_normalized = self.minmax_scaler.fit_transform(audio_zscore.reshape(-1, 1)).flatten()
            
            # Scale to desired range [-0.8, 0.8] for ASR optimization
            audio_normalized = audio_normalized * 1.6 - 0.8
            
            return audio_normalized.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Z-score Min-Max normalization failed: {e}")
            return librosa.util.normalize(audio).astype(np.float32)
    
    def dynamic_range_compression(self, audio: np.ndarray) -> np.ndarray:
        """Advanced dynamic range compression"""
        try:
            print("📊 Applying dynamic range compression...")
            
            # Parameters
            threshold = 0.3
            ratio = 4.0
            attack_time = 0.003  # 3ms
            release_time = 0.100  # 100ms
            
            # Convert time to samples
            attack_samples = int(attack_time * self.sample_rate)
            release_samples = int(release_time * self.sample_rate)
            
            # Initialize variables
            envelope = 0.0
            compressed_audio = np.zeros_like(audio)
            
            for i, sample in enumerate(audio):
                # Envelope following
                input_level = abs(sample)
                
                if input_level > envelope:
                    # Attack
                    envelope += (input_level - envelope) / attack_samples
                else:
                    # Release
                    envelope += (input_level - envelope) / release_samples
                
                # Compression
                if envelope > threshold:
                    # Calculate compression gain
                    excess = envelope - threshold
                    compressed_excess = excess / ratio
                    gain = (threshold + compressed_excess) / envelope if envelope > 0 else 1.0
                else:
                    gain = 1.0
                
                # Apply gain
                compressed_audio[i] = sample * gain
            
            return compressed_audio.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Dynamic range compression failed: {e}")
            return audio
    
    def noise_gating(self, audio: np.ndarray, threshold_db: float = -40.0) -> np.ndarray:
        """Noise gating to suppress low-level background noise"""
        try:
            print("🚪 Applying noise gating...")
            
            # Convert threshold to linear scale
            threshold_linear = 10 ** (threshold_db / 20.0)
            
            # Calculate envelope
            frame_length = 1024
            hop_length = 256
            
            # RMS energy calculation
            rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
            
            # Expand to audio length
            rms_expanded = np.repeat(rms, hop_length)
            if len(rms_expanded) > len(audio):
                rms_expanded = rms_expanded[:len(audio)]
            elif len(rms_expanded) < len(audio):
                rms_expanded = np.pad(rms_expanded, (0, len(audio) - len(rms_expanded)), mode='edge')
            
            # Create gate
            gate = (rms_expanded > threshold_linear).astype(float)
            
            # Smooth the gate to avoid clicks
            gate_smooth = gaussian_filter1d(gate, sigma=2.0)
            
            # Apply gate with minimum attenuation
            min_gain = 0.1
            gate_gain = gate_smooth * (1 - min_gain) + min_gain
            
            # Apply gating
            gated_audio = audio * gate_gain
            
            return gated_audio.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Noise gating failed: {e}")
            return audio
    
    def temporal_smoothing(self, audio: np.ndarray, window_size: int = 5) -> np.ndarray:
        """Temporal smoothing to reduce transient noise"""
        try:
            print("📈 Applying temporal smoothing...")
            
            # Apply moving average filter
            if window_size > len(audio):
                window_size = len(audio) // 10
            
            # Create smoothing kernel
            kernel = np.ones(window_size) / window_size
            
            # Apply convolution with padding
            smoothed_audio = np.convolve(audio, kernel, mode='same')
            
            return smoothed_audio.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Temporal smoothing failed: {e}")
            return audio
    
    def frame_averaging(self, audio: np.ndarray, num_frames: int = 3) -> np.ndarray:
        """Frame averaging to improve SNR"""
        try:
            print("📊 Applying frame averaging...")
            
            frame_length = 1024
            hop_length = 512
            
            # Extract frames
            frames = []
            for i in range(0, len(audio) - frame_length + 1, hop_length):
                frame = audio[i:i + frame_length]
                frames.append(frame)
            
            if len(frames) < num_frames:
                return audio
            
            # Average frames
            averaged_audio = np.zeros_like(audio)
            
            for i in range(len(frames)):
                start_idx = i * hop_length
                end_idx = start_idx + frame_length
                
                # Select frames to average
                start_frame = max(0, i - num_frames // 2)
                end_frame = min(len(frames), start_frame + num_frames)
                
                # Average selected frames
                avg_frame = np.mean(frames[start_frame:end_frame], axis=0)
                
                # Add to output with overlap handling
                if end_idx <= len(averaged_audio):
                    averaged_audio[start_idx:end_idx] += avg_frame
                else:
                    remaining = len(averaged_audio) - start_idx
                    averaged_audio[start_idx:] += avg_frame[:remaining]
            
            return averaged_audio.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Frame averaging failed: {e}")
            return audio
    
    # ADVANCED METHODS
    
    def signal_subspace_approach(self, audio: np.ndarray) -> np.ndarray:
        """Signal Subspace Approach (SSA) for speech enhancement"""
        try:
            print("🔬 Applying Signal Subspace Approach...")
            
            # Frame the signal
            frame_length = 256
            hop_length = 128
            
            # Create Hankel matrix
            frames = []
            for i in range(0, len(audio) - frame_length + 1, hop_length):
                frames.append(audio[i:i + frame_length])
            
            if len(frames) < 2:
                return audio
            
            # Convert to matrix
            X = np.array(frames).T
            
            # SVD decomposition
            U, s, Vt = svd(X, full_matrices=False)
            
            # Determine signal subspace dimension (keep top components)
            total_energy = np.sum(s**2)
            cumulative_energy = np.cumsum(s**2)
            
            # Keep components that contain 95% of energy
            signal_dim = np.argmax(cumulative_energy / total_energy > 0.95) + 1
            signal_dim = max(1, min(signal_dim, len(s) // 2))  # Ensure reasonable dimension
            
            # Reconstruct using signal subspace
            X_clean = U[:, :signal_dim] @ np.diag(s[:signal_dim]) @ Vt[:signal_dim, :]
            
            # Reconstruct audio
            enhanced_audio = np.zeros_like(audio)
            for i, frame in enumerate(X_clean.T):
                start_idx = i * hop_length
                end_idx = start_idx + frame_length
                if end_idx <= len(enhanced_audio):
                    enhanced_audio[start_idx:end_idx] += frame
                else:
                    remaining = len(enhanced_audio) - start_idx
                    enhanced_audio[start_idx:] += frame[:remaining]
            
            return enhanced_audio.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Signal Subspace Approach failed: {e}")
            return audio
    
    def noise_profile_analysis(self, audio: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Comprehensive noise profile analysis"""
        try:
            print("🔍 Performing noise profile analysis...")
            
            # Detect noise-only regions using VAD
            vad_result, vad_stats = self.vad.detect_voice_activity(audio)
            
            hop_length = 256
            vad_expanded = np.repeat(vad_result, hop_length)
            
            if len(vad_expanded) > len(audio):
                vad_expanded = vad_expanded[:len(audio)]
            elif len(vad_expanded) < len(audio):
                vad_expanded = np.pad(vad_expanded, (0, len(audio) - len(vad_expanded)), mode='edge')
            
            noise_regions = ~vad_expanded.astype(bool)
            
            # Analyze noise characteristics
            noise_profile = {}
            
            if np.any(noise_regions):
                noise_samples = audio[noise_regions]
                
                # Statistical properties
                noise_profile['mean'] = np.mean(noise_samples)
                noise_profile['std'] = np.std(noise_samples)
                noise_profile['rms'] = np.sqrt(np.mean(noise_samples**2))
                noise_profile['peak'] = np.max(np.abs(noise_samples))
                
                # Spectral properties
                noise_stft = librosa.stft(noise_samples[:min(len(noise_samples), 8192)])
                noise_magnitude = np.abs(noise_stft)
                noise_profile['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(S=noise_magnitude))
                noise_profile['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(S=noise_magnitude))
                
                # Noise type classification (simplified)
                if noise_profile['spectral_centroid'] > 3000:
                    noise_profile['type'] = 'high_frequency'
                elif noise_profile['spectral_centroid'] < 1000:
                    noise_profile['type'] = 'low_frequency'
                else:
                    noise_profile['type'] = 'broadband'
                
                # Apply targeted noise reduction
                enhanced_audio = self.targeted_noise_reduction(audio, noise_profile)
                
            else:
                # No clear noise regions detected
                noise_profile['type'] = 'unknown'
                enhanced_audio = audio
            
            return enhanced_audio.astype(np.float32), noise_profile
            
        except Exception as e:
            print(f"❌ Noise profile analysis failed: {e}")
            return audio, {}
    
    def targeted_noise_reduction(self, audio: np.ndarray, noise_profile: Dict) -> np.ndarray:
        """Apply targeted noise reduction based on noise profile"""
        try:
            noise_type = noise_profile.get('type', 'broadband')
            
            if noise_type == 'high_frequency':
                # Apply stronger low-pass filtering
                cutoff = 6000
                b, a = butter(6, cutoff, btype='low', fs=self.sample_rate)
                audio = filtfilt(b, a, audio)
                
            elif noise_type == 'low_frequency':
                # Apply stronger high-pass filtering
                cutoff = 120
                b, a = butter(6, cutoff, btype='high', fs=self.sample_rate)
                audio = filtfilt(b, a, audio)
                
            else:  # broadband
                # Apply moderate filtering
                audio = self.comprehensive_frequency_filtering(audio)
            
            return audio
            
        except Exception as e:
            print(f"❌ Targeted noise reduction failed: {e}")
            return audio
    
    def snr_enhancement(self, audio: np.ndarray) -> Tuple[np.ndarray, float]:
        """SNR enhancement with measurement"""
        try:
            print("📊 Applying SNR enhancement...")
            
            # Measure initial SNR
            initial_snr = self.measure_snr(audio)
            
            # Apply multi-stage enhancement
            enhanced_audio = audio.copy()
            
            # Stage 1: Spectral subtraction
            enhanced_audio = self.spectral_subtraction(enhanced_audio, alpha=1.5, beta=0.05)
            
            # Stage 2: Wiener filtering
            enhanced_audio = self.wiener_filtering(enhanced_audio)
            
            # Stage 3: Dynamic range compression
            enhanced_audio = self.dynamic_range_compression(enhanced_audio)
            
            # Measure final SNR
            final_snr = self.measure_snr(enhanced_audio)
            snr_improvement = final_snr - initial_snr
            
            print(f"📊 SNR improved from {initial_snr:.2f} dB to {final_snr:.2f} dB (+{snr_improvement:.2f} dB)")
            
            return enhanced_audio.astype(np.float32), snr_improvement
            
        except Exception as e:
            print(f"❌ SNR enhancement failed: {e}")
            return audio, 0.0
    
    def measure_snr(self, audio: np.ndarray) -> float:
        """Measure Signal-to-Noise Ratio"""
        try:
            # Use VAD to separate signal and noise
            vad_result, _ = self.vad.detect_voice_activity(audio)
            
            hop_length = 256
            vad_expanded = np.repeat(vad_result, hop_length)
            
            if len(vad_expanded) > len(audio):
                vad_expanded = vad_expanded[:len(audio)]
            elif len(vad_expanded) < len(audio):
                vad_expanded = np.pad(vad_expanded, (0, len(audio) - len(vad_expanded)), mode='edge')
            
            voice_regions = vad_expanded.astype(bool)
            noise_regions = ~voice_regions
            
            if np.any(voice_regions) and np.any(noise_regions):
                signal_power = np.mean(audio[voice_regions]**2)
                noise_power = np.mean(audio[noise_regions]**2)
                
                if noise_power > 0:
                    snr = 10 * np.log10(signal_power / noise_power)
                    return snr
            
            # Fallback: estimate SNR using energy distribution
            frame_energy = librosa.feature.rms(y=audio, frame_length=1024, hop_length=512)[0]
            signal_power = np.percentile(frame_energy, 90)**2
            noise_power = np.percentile(frame_energy, 10)**2
            
            if noise_power > 0:
                snr = 10 * np.log10(signal_power / noise_power)
                return snr
            
            return 20.0  # Default SNR
            
        except Exception as e:
            print(f"❌ SNR measurement failed: {e}")
            return 20.0
    
    # COMPREHENSIVE PIPELINE
    
    def comprehensive_speech_enhancement(self, audio: np.ndarray, enhancement_level: str = "moderate") -> Tuple[np.ndarray, Dict]:
        """COMPREHENSIVE: Complete speech enhancement pipeline with ALL methods"""
        original_audio = audio.copy()
        stats = {'enhancement_level': enhancement_level}
        
        try:
            print(f"🚀 Starting COMPREHENSIVE speech enhancement pipeline ({enhancement_level})...")
            print("✅ ALL preprocessing methods will be applied in optimal order")
            
            # Initial quality assessment
            initial_snr = self.measure_snr(audio)
            stats['initial_snr'] = initial_snr
            stats['original_length'] = len(audio) / self.sample_rate
            
            print(f"📊 Initial SNR: {initial_snr:.2f} dB")
            
            # STAGE 1: PREPROCESSING AND NORMALIZATION
            print("🔧 STAGE 1: Preprocessing and Normalization")
            
            # Z-score Min-Max normalization
            if Z_SCORE_NORMALIZATION:
                audio = self.z_score_min_max_normalization(audio)
            
            # Noise gating
            if NOISE_GATING_ENABLED:
                audio = self.noise_gating(audio)
            
            # STAGE 2: FREQUENCY DOMAIN FILTERING
            print("🎵 STAGE 2: Frequency Domain Filtering")
            
            # Comprehensive frequency filtering (FIXED)
            audio = self.comprehensive_frequency_filtering(audio)
            
            # Adaptive filtering
            if ADAPTIVE_FILTERING_ENABLED:
                audio = self.adaptive_filtering(audio)
            
            # STAGE 3: SPECTRAL DOMAIN METHODS
            print("🔬 STAGE 3: Spectral Domain Methods")
            
            # Classical spectral subtraction
            if SPECTRAL_SUBTRACTION_ENABLED:
                audio = self.spectral_subtraction(audio)
            
            # Multi-Band Spectral Subtraction
            if MULTI_BAND_SPECTRAL_SUBTRACTION and enhancement_level in ["moderate", "aggressive"]:
                audio = self.multi_band_spectral_subtraction(audio)
            
            # Advanced estimators based on enhancement level
            if enhancement_level == "aggressive":
                # Apply all advanced methods
                if MMSE_STSA_ENABLED:
                    audio = self.mmse_stsa_estimator(audio)
                
                if MMSE_LSA_ENABLED:
                    audio = self.mmse_lsa_estimator(audio)
                
                if OM_LSA_ENABLED:
                    audio = self.om_lsa_estimator(audio)
                
            elif enhancement_level == "moderate":
                # Apply selective advanced methods
                if MMSE_STSA_ENABLED:
                    audio = self.mmse_stsa_estimator(audio)
            
            # Wiener filtering
            if WIENER_FILTERING_ENABLED and enhancement_level in ["moderate", "aggressive"]:
                audio = self.wiener_filtering(audio)
            
            # STAGE 4: TIME-FREQUENCY DOMAIN PROCESSING
            print("🔬 STAGE 4: Time-Frequency Domain Processing")
            
            # DA-STFT processing
            if DA_STFT_ENABLED:
                audio = self.da_stft_processing(audio)
            
            # Time-frequency masking
            if TIME_FREQUENCY_MASKING:
                audio = self.time_frequency_masking(audio)
            
            # Frame-based processing
            audio = self.frame_based_processing(audio)
            
            # STAGE 5: ADVANCED METHODS
            print("🔬 STAGE 5: Advanced Methods")
            
            # Signal Subspace Approach
            if SIGNAL_SUBSPACE_APPROACH and enhancement_level == "aggressive":
                audio = self.signal_subspace_approach(audio)
            
            # Noise profile analysis and targeted reduction
            if NOISE_PROFILE_ANALYSIS:
                audio, noise_profile = self.noise_profile_analysis(audio)
                stats.update({'noise_profile': noise_profile})
            
            # STAGE 6: TEMPORAL PROCESSING
            print("📊 STAGE 6: Temporal Processing")
            
            # Temporal smoothing
            if TEMPORAL_SMOOTHING:
                audio = self.temporal_smoothing(audio)
            
            # Frame averaging
            if FRAME_AVERAGING and enhancement_level in ["moderate", "aggressive"]:
                audio = self.frame_averaging(audio)
            
            # Dynamic range compression
            if DYNAMIC_RANGE_COMPRESSION:
                audio = self.dynamic_range_compression(audio)
            
            # STAGE 7: VOICE ACTIVITY ENHANCEMENT
            print("🎤 STAGE 7: Voice Activity Enhancement")
            
            if ADVANCED_VAD_ENABLED:
                vad_result, vad_stats = self.vad.detect_voice_activity(audio)
                stats.update(vad_stats)
                
                # Apply voice activity-based enhancement
                hop_length = 256
                vad_expanded = np.repeat(vad_result, hop_length)
                
                if len(vad_expanded) > len(audio):
                    vad_expanded = vad_expanded[:len(audio)]
                elif len(vad_expanded) < len(audio):
                    vad_expanded = np.pad(vad_expanded, (0, len(audio) - len(vad_expanded)), mode='edge')
                
                voice_regions = vad_expanded.astype(bool)
                
                # Conservative enhancement
                if np.any(voice_regions):
                    audio[voice_regions] *= 1.05  # Light boost
                
                noise_regions = ~voice_regions
                if np.any(noise_regions):
                    audio[noise_regions] *= 0.95  # Light suppression
            
            # STAGE 8: SNR ENHANCEMENT AND FINAL PROCESSING
            print("📊 STAGE 8: SNR Enhancement and Final Processing")
            
            if SNR_ENHANCEMENT:
                audio, snr_improvement = self.snr_enhancement(audio)
                stats['snr_improvement'] = snr_improvement
            
            # Final normalization for ASR
            audio = librosa.util.normalize(audio)
            audio = np.clip(audio, -0.99, 0.99)
            
            # Final quality assessment
            final_snr = self.measure_snr(audio)
            stats['final_snr'] = final_snr
            stats['total_snr_improvement'] = final_snr - initial_snr
            stats['final_rms'] = np.sqrt(np.mean(audio**2))
            
            print(f"✅ COMPREHENSIVE enhancement completed")
            print(f"📊 SNR improvement: {stats['total_snr_improvement']:.2f} dB")
            print(f"📊 Final RMS level: {stats['final_rms']:.4f} (ASR-optimized)")
            
            return audio.astype(np.float32), stats
            
        except Exception as e:
            print(f"❌ Comprehensive enhancement failed: {e}")
            return original_audio.astype(np.float32), {}

class AudioHandler:
    """Audio handling for all Gradio input types"""
    
    @staticmethod
    def convert_to_file(audio_input, target_sr=SAMPLE_RATE):
        if audio_input is None:
            raise ValueError("No audio input provided")
        
        try:
            if isinstance(audio_input, tuple):
                sample_rate, audio_data = audio_input
                print(f"🎙️ Converting live recording: {sample_rate}Hz, {len(audio_data)} samples")
                
                if not isinstance(audio_data, np.ndarray):
                    raise ValueError("Audio data must be numpy array")
                
                if audio_data.dtype != np.float32:
                    if np.issubdtype(audio_data.dtype, np.integer):
                        if audio_data.dtype == np.int16:
                            audio_data = audio_data.astype(np.float32) / 32768.0
                        elif audio_data.dtype == np.int32:
                            audio_data = audio_data.astype(np.float32) / 2147483648.0
                        else:
                            audio_data = audio_data.astype(np.float32)
                    else:
                        audio_data = audio_data.astype(np.float32)
                
                max_val = np.max(np.abs(audio_data))
                if max_val > 1.0:
                    audio_data = audio_data / max_val
                
                if sample_rate != target_sr:
                    print(f"🔄 Resampling from {sample_rate}Hz to {target_sr}Hz")
                    audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sr)
                
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                sf.write(temp_file.name, audio_data, target_sr)
                temp_file.close()
                
                print(f"✅ Live recording converted to: {temp_file.name}")
                return temp_file.name
                
            elif isinstance(audio_input, str):
                if not os.path.exists(audio_input):
                    raise ValueError(f"Audio file not found: {audio_input}")
                
                print(f"📁 Using file path: {audio_input}")
                return audio_input
                
            else:
                raise ValueError(f"Unsupported audio input type: {type(audio_input)}")
                
        except Exception as e:
            print(f"❌ Audio conversion failed: {e}")
            raise
    
    @staticmethod
    def numpy_to_temp_file(audio_array, sample_rate=SAMPLE_RATE):
        try:
            if not isinstance(audio_array, np.ndarray):
                raise ValueError("Input must be numpy array")
            
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            
            max_val = np.max(np.abs(audio_array))
            if max_val > 1.0:
                audio_array = audio_array / max_val
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            sf.write(temp_file.name, audio_array, sample_rate)
            temp_file.close()
            
            return temp_file.name
            
        except Exception as e:
            print(f"❌ Numpy to temp file conversion failed: {e}")
            raise
    
    @staticmethod
    def cleanup_temp_file(file_path):
        try:
            if file_path and os.path.exists(file_path):
                if file_path.startswith('/tmp') or 'tmp' in file_path:
                    os.unlink(file_path)
                    print(f"🗑️ Cleaned up temp file: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"⚠️ Temp file cleanup warning: {e}")

class OptimizedMemoryManager:
    @staticmethod
    def quick_memory_check():
        if not torch.cuda.is_available():
            return True
        
        try:
            allocated = torch.cuda.memory_allocated() / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            available = total - allocated
            return available >= MIN_FREE_MEMORY_GB
        except:
            return True
    
    @staticmethod
    def fast_cleanup():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    @staticmethod
    def log_memory_status(context="", force_log=False):
        if force_log and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"📊 {context} - GPU: {allocated:.1f}GB/{total:.1f}GB")

class SmartTextChunker:
    def __init__(self, max_chunk_size=MAX_TRANSLATION_CHUNK_SIZE, min_chunk_size=MIN_CHUNK_SIZE):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.sentence_overlap = SENTENCE_OVERLAP
    
    def split_into_sentences(self, text: str) -> List[str]:
        try:
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(text)
            if sentences and len(sentences) > 1:
                return sentences
        except:
            pass
        
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        if len(sentences) <= 1:
            sentences = re.split(r'\.\s+', text)
            sentences = [s + '.' if i < len(sentences) - 1 and not s.endswith('.') else s 
                        for i, s in enumerate(sentences)]
        
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences if sentences else [text]
    
    def create_smart_chunks(self, text: str) -> List[str]:
        if not text or len(text) <= self.max_chunk_size:
            return [text] if text else []
        
        print(f"📝 Creating smart chunks for {len(text)} characters...")
        sentences = self.split_into_sentences(text)
        
        if len(sentences) <= 1:
            return self.fallback_chunking(text)
        
        chunks = []
        current_chunk = ""
        sentence_buffer = []
        
        for i, sentence in enumerate(sentences):
            sentence_with_space = sentence if not current_chunk else " " + sentence
            
            if current_chunk and len(current_chunk + sentence_with_space) > self.max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                overlap_sentences = sentence_buffer[-self.sentence_overlap:] if len(sentence_buffer) >= self.sentence_overlap else sentence_buffer
                current_chunk = " ".join(overlap_sentences)
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                
                sentence_buffer = overlap_sentences + [sentence]
            else:
                current_chunk += sentence_with_space
                sentence_buffer.append(sentence)
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        if len(chunks) > 1:
            chunks = [chunk for chunk in chunks if len(chunk) >= self.min_chunk_size]
        
        print(f"✅ Created {len(chunks)} smart chunks")
        return chunks
    
    def fallback_chunking(self, text: str) -> List[str]:
        paragraphs = text.split('\n\n')
        if len(paragraphs) > 1:
            chunks = []
            current_chunk = ""
            
            for para in paragraphs:
                if current_chunk and len(current_chunk + "\n\n" + para) > self.max_chunk_size:
                    chunks.append(current_chunk.strip())
                    current_chunk = para
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + para
                    else:
                        current_chunk = para
            
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            return chunks
        
        words = text.split()
        chunks = []
        current_chunk = ""
        
        for word in words:
            if current_chunk and len(current_chunk + " " + word) > self.max_chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = word
            else:
                if current_chunk:
                    current_chunk += " " + word
                else:
                    current_chunk = word
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks

class ComprehensiveSpeechTranscriber:
    """COMPREHENSIVE: Audio transcriber with complete preprocessing pipeline"""
    
    def __init__(self, model_path: str, use_quantization: bool = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = None
        self.processor = None
        self.audio_enhancer = ComprehensiveSpeechEnhancer(SAMPLE_RATE)
        self.text_chunker = SmartTextChunker()
        self.chunk_count = 0
        self.temp_files = []
        
        print(f"🖥️ Using device: {self.device}")
        print(f"🚀 COMPREHENSIVE speech enhancement enabled with ALL methods:")
        print(f"   🔬 Spectral Domain: Subtraction, MBSS, Wiener, MMSE-STSA, MMSE-LSA, OM-LSA")
        print(f"   🎵 Frequency Domain: Low/High/Band-pass, Adaptive Filtering")
        print(f"   🔬 Time-Frequency: DA-STFT, FFT+Hanning, Frame-Based, TF Masking")
        print(f"   📊 Normalization: Z-score Min-Max, Dynamic Range, Noise Gating")
        print(f"   🔬 Advanced: Signal Subspace, Noise Profile Analysis, SNR Enhancement")
        print(f"   📊 Temporal: Smoothing, Frame Averaging")
        print(f"   🎤 VAD: Comprehensive multi-feature detection")
        print(f"⏱️ Chunk timeout: {CHUNK_TIMEOUT} seconds")
        
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"Model directory not found at '{model_path}'")

        OptimizedMemoryManager.fast_cleanup()
        self.load_model_without_compilation(model_path, use_quantization)
    
    def load_model_without_compilation(self, model_path: str, use_quantization: bool):
        try:
            print("🚀 Loading model without torch.compile()...")
            start_time = time.time()
            
            self.processor = Gemma3nProcessor.from_pretrained(model_path)
            
            if use_quantization and self.device.type == "cuda":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_skip_modules=["lm_head"],
                )
                print("🔧 Using 8-bit quantization...")
            else:
                quantization_config = None
                print("🔧 Using bfloat16 precision...")

            self.model = Gemma3nForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                device_map="auto",
                quantization_config=quantization_config,
                trust_remote_code=True,
                use_safetensors=True,
            )
            
            self.model.eval()
            
            loading_time = time.time() - start_time
            OptimizedMemoryManager.log_memory_status("After model loading", force_log=True)
            print(f"✅ COMPREHENSIVE model loaded in {loading_time:.1f} seconds")
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            raise
    
    def create_speech_chunks(self, audio_array: np.ndarray) -> List[Tuple[np.ndarray, float, float]]:
        chunk_samples = int(CHUNK_SECONDS * SAMPLE_RATE)
        overlap_samples = int(OVERLAP_SECONDS * SAMPLE_RATE)
        stride = chunk_samples - overlap_samples
        
        chunks = []
        start = 0
        
        while start < len(audio_array):
            end = min(start + chunk_samples, len(audio_array))
            
            if end - start < SAMPLE_RATE:
                if chunks:
                    last_chunk, last_start, _ = chunks.pop()
                    extended_chunk = audio_array[int(last_start * SAMPLE_RATE):end]
                    chunks.append((extended_chunk, last_start, end / SAMPLE_RATE))
                break
            
            chunk = audio_array[start:end]
            start_time = start / SAMPLE_RATE
            end_time = end / SAMPLE_RATE
            
            chunks.append((chunk, start_time, end_time))
            start += stride
            
            if len(chunks) >= 100:
                print("⚠️ Reached chunk limit for processing speed")
                break
        
        print(f"✅ Created {len(chunks)} comprehensive processing chunks")
        return chunks
    
    def transcribe_chunk_with_timeout(self, audio_chunk: np.ndarray, language: str = "auto") -> str:
        if self.model is None or self.processor is None:
            return "[MODEL_NOT_LOADED]"
        
        temp_audio_file = None
        
        try:
            self.chunk_count += 1
            if self.chunk_count % CHECK_MEMORY_FREQUENCY == 0:
                if not OptimizedMemoryManager.quick_memory_check():
                    OptimizedMemoryManager.fast_cleanup()
            
            snr = self.audio_enhancer.measure_snr(audio_chunk)
            print(f"🔍 Chunk SNR: {snr:.1f} dB")
            
            temp_audio_file = AudioHandler.numpy_to_temp_file(audio_chunk, SAMPLE_RATE)
            self.temp_files.append(temp_audio_file)
            
            if language == "auto":
                system_message = "Transcribe this audio accurately with proper punctuation."
            else:
                lang_name = [k for k, v in SUPPORTED_LANGUAGES.items() if v == language]
                lang_display = lang_name[0] if lang_name else language
                system_message = f"Transcribe this audio in {lang_display} with proper punctuation."
            
            message = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_message}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": temp_audio_file},
                        {"type": "text", "text": "Transcribe this audio."},
                    ],
                },
            ]

            def transcribe_worker():
                with torch.inference_mode():
                    inputs = self.processor.apply_chat_template(
                        message,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                    ).to(self.device)

                    input_len = inputs["input_ids"].shape[-1]

                    generation = self.model.generate(
                        **inputs, 
                        max_new_tokens=200,
                        do_sample=False,
                        temperature=0.1,
                        pad_token_id=self.processor.tokenizer.eos_token_id,
                        use_cache=True,
                        early_stopping=True
                    )
                    
                    generation = generation[0][input_len:]
                    transcription = self.processor.decode(generation, skip_special_tokens=True)
                    
                    del inputs, generation
                    return transcription.strip()
            
            result_queue = queue.Queue()
            
            def timeout_transcribe():
                try:
                    result = transcribe_worker()
                    result_queue.put(("success", result))
                except Exception as e:
                    result_queue.put(("error", str(e)))
            
            transcribe_thread = threading.Thread(target=timeout_transcribe)
            transcribe_thread.daemon = True
            transcribe_thread.start()
            
            transcribe_thread.join(timeout=CHUNK_TIMEOUT)
            
            if transcribe_thread.is_alive():
                print(f"⏱️ Chunk processing timed out after {CHUNK_TIMEOUT} seconds")
                return "Input Audio Very noisy. Unable to extract details."
            
            try:
                status, result = result_queue.get_nowait()
                if status == "success":
                    if not result or len(result) < 2:
                        return "[AUDIO_UNCLEAR]"
                    return result
                else:
                    return f"[ERROR: {result[:30]}]"
            except queue.Empty:
                return "Input Audio Very noisy. Unable to extract details."
                
        except torch.cuda.OutOfMemoryError as e:
            print(f"❌ CUDA OOM: {e}")
            OptimizedMemoryManager.fast_cleanup()
            return "[CUDA_OUT_OF_MEMORY]"
        except Exception as e:
            print(f"❌ Comprehensive transcription error: {str(e)}")
            return f"[ERROR: {str(e)[:30]}]"
        finally:
            if temp_audio_file:
                AudioHandler.cleanup_temp_file(temp_audio_file)
    
    def translate_text_chunks(self, text: str) -> str:
        if self.model is None or self.processor is None:
            return "[MODEL_NOT_LOADED]"
        
        if not text or text.startswith('['):
            return "[NO_TRANSLATION_NEEDED]"
        
        try:
            print("🌐 Starting comprehensive text translation...")
            
            english_indicators = [
                "the", "and", "is", "in", "to", "of", "a", "that", "it", "with", "for", "as", "was", "on", "are", "you",
                "have", "be", "this", "from", "they", "will", "been", "has", "were", "said", "each", "which", "can",
                "there", "use", "an", "she", "how", "its", "our", "out", "many", "time", "very", "when", "much", "would"
            ]
            
            text_words = re.findall(r'\b\w+\b', text.lower())
            if len(text_words) >= 5:
                english_word_count = sum(1 for word in text_words[:30] if word in english_indicators)
                english_ratio = english_word_count / min(len(text_words), 30)
                
                if english_ratio >= 0.4:
                    print(f"✅ Text appears to be already in English (ratio: {english_ratio:.2f})")
                    return f"[ALREADY_IN_ENGLISH] {text}"
            
            text_chunks = self.text_chunker.create_smart_chunks(text)
            
            if len(text_chunks) == 1:
                return self.translate_single_chunk(text_chunks[0])
            
            translated_chunks = []
            
            for i, chunk in enumerate(text_chunks, 1):
                try:
                    translated_chunk = self.translate_single_chunk(chunk)
                    
                    if translated_chunk.startswith('['):
                        translated_chunks.append(chunk)
                    else:
                        translated_chunks.append(translated_chunk)
                    
                except Exception as e:
                    print(f"❌ Chunk {i} translation failed: {e}")
                    translated_chunks.append(chunk)
                
                OptimizedMemoryManager.fast_cleanup()
                time.sleep(0.2)
            
            merged_translation = self.merge_translated_chunks(translated_chunks)
            return merged_translation
            
        except Exception as e:
            print(f"❌ Comprehensive translation error: {str(e)}")
            OptimizedMemoryManager.fast_cleanup()
            return f"[TRANSLATION_ERROR: {str(e)[:50]}]"
    
    def translate_single_chunk(self, chunk: str) -> str:
        try:
            message = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a professional translator. Translate the given text to English accurately while preserving the meaning, context, and style. Maintain proper punctuation and formatting."}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Translate the following text to English:\n\n{chunk}"},
                    ],
                },
            ]

            with torch.inference_mode():
                inputs = self.processor.apply_chat_template(
                    message,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(self.device)

                input_len = inputs["input_ids"].shape[-1]

                generation = self.model.generate(
                    **inputs, 
                    max_new_tokens=300,
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True,
                    early_stopping=True
                )
                
                generation = generation[0][input_len:]
                translation = self.processor.decode(generation, skip_special_tokens=True)
                
                del inputs, generation
                
                result = translation.strip()
                if not result or len(result) < 2:
                    return "[CHUNK_TRANSLATION_UNCLEAR]"
                
                return result
                
        except Exception as e:
            print(f"❌ Single chunk translation error: {e}")
            return f"[CHUNK_ERROR: {str(e)[:30]}]"
    
    def merge_translated_chunks(self, translated_chunks: List[str]) -> str:
        if not translated_chunks:
            return "[NO_TRANSLATED_CHUNKS]"
        
        valid_chunks = [chunk for chunk in translated_chunks if not chunk.startswith('[')]
        
        if not valid_chunks:
            return "[ALL_CHUNKS_FAILED]"
        
        merged_text = ""
        for i, chunk in enumerate(valid_chunks):
            if i == 0:
                merged_text = chunk
            else:
                if not merged_text.endswith((' ', '\n')) and not chunk.startswith((' ', '\n')):
                    if not merged_text.endswith(('.', '!', '?', ':', ';')):
                        merged_text += " "
                merged_text += chunk
        
        failed_chunks = len(translated_chunks) - len(valid_chunks)
        if failed_chunks > 0:
            success_rate = (len(valid_chunks) / len(translated_chunks)) * 100
            merged_text += f"\n\n[Translation Summary: {len(valid_chunks)}/{len(translated_chunks)} chunks successful ({success_rate:.1f}% success rate)]"
        
        return merged_text.strip()
    
    def transcribe_with_comprehensive_enhancement(self, audio_path: str, language: str = "auto", 
                                                enhancement_level: str = "moderate") -> Tuple[str, str, str, Dict]:
        try:
            print(f"🚀 Starting COMPREHENSIVE transcription with ALL preprocessing methods...")
            print(f"🔧 Enhancement level: {enhancement_level}")
            print(f"🌍 Language: {language}")
            print(f"⏱️
