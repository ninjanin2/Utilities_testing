# -*- coding: utf-8 -*-
"""
COMPREHENSIVE SPEECH ENHANCEMENT WITH USER-SELECTABLE PREPROCESSING
==================================================================

USER-SELECTABLE PREPROCESSING METHODS:
- Spectral Domain Methods (6 techniques)
- Frequency Domain Filtering (2 techniques)
- Time-Frequency Processing (3 techniques)
- Preprocessing & Normalization (5 techniques)
- Advanced Methods (4 techniques)
- All methods available as checkboxes in UI
- 75-second timeout with noise detection messages

Author: User-Controlled AI Audio Processing System
Version: Checkbox-Controlled Enhancement 15.0
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
from scipy.signal import butter, filtfilt, lfilter, wiener
from scipy.signal.windows import hann  # FIXED: Use hann instead of hanning
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

# --- USER-SELECTABLE ENHANCEMENT CONFIGURATION ---
MODEL_PATH = "/path/to/your/local/gemma-3n-e4b-it"  # UPDATE THIS PATH

# Enhanced settings
CHUNK_SECONDS = 12
OVERLAP_SECONDS = 2
SAMPLE_RATE = 16000
CHUNK_TIMEOUT = 75
MAX_RETRIES = 1
PROCESSING_THREADS = 1

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

class UserSelectableVoiceActivityDetector:
    """USER-SELECTABLE: Voice activity detection"""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.frame_length = 1024
        self.hop_length = 256
        
    def detect_voice_activity(self, audio: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Multi-feature VAD"""
        try:
            print("🎤 USER-SELECTED voice activity detection...")
            
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
            
            # MFCC features
            mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13, hop_length=self.hop_length)
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate, hop_length=self.hop_length)
            
            # Statistical thresholding
            energy_threshold = np.percentile(frame_energy, 15)
            centroid_threshold = np.percentile(spectral_centroids, 10)
            rolloff_threshold = np.percentile(spectral_rolloff, 20)
            bandwidth_threshold = np.percentile(spectral_bandwidth, 25)
            contrast_threshold = np.percentile(np.mean(spectral_contrast, axis=0), 20)
            flatness_threshold = np.percentile(spectral_flatness, 70)
            zcr_threshold = np.percentile(zcr, 85)
            mfcc_threshold = np.percentile(np.mean(mfcc, axis=0), 20)
            
            # Multi-criteria decision
            voice_criteria = [
                frame_energy > energy_threshold,
                spectral_centroids > centroid_threshold,
                spectral_rolloff > rolloff_threshold,
                spectral_bandwidth > bandwidth_threshold,
                np.mean(spectral_contrast, axis=0) > contrast_threshold,
                spectral_flatness < flatness_threshold,
                zcr < zcr_threshold,
                np.mean(mfcc, axis=0) > mfcc_threshold,
            ]
            
            # Weighted voting
            weights = [0.25, 0.20, 0.15, 0.10, 0.10, 0.05, 0.05, 0.10]
            voice_scores = np.zeros(len(frame_energy))
            
            for criterion, weight in zip(voice_criteria, weights):
                voice_scores += criterion.astype(float) * weight
            
            # Threshold-based decision
            voice_activity = voice_scores > 0.35
            
            # Smoothing
            voice_activity = median_filter(voice_activity.astype(float), size=7) > 0.3
            
            # Statistics
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
            print(f"❌ VAD failed: {e}")
            return np.ones(len(audio) // self.hop_length, dtype=bool), {}
    
    def estimate_snr(self, audio: np.ndarray, voice_activity: np.ndarray) -> float:
        """Estimate SNR using VAD"""
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
            
            return 20.0
        except:
            return 20.0

class UserSelectableSpeechEnhancer:
    """USER-SELECTABLE: Speech enhancement with checkbox controls"""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.vad = UserSelectableVoiceActivityDetector(sample_rate)
        self.frame_size = 1024
        self.hop_size = 256
        self.scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        print(f"🚀 USER-SELECTABLE Speech Enhancer initialized for {sample_rate}Hz")
    
    # SPECTRAL DOMAIN METHODS
    
    def spectral_subtraction(self, audio: np.ndarray, alpha: float = 2.0, beta: float = 0.01) -> np.ndarray:
        """Classical spectral subtraction"""
        try:
            print("🔬 Applying spectral subtraction...")
            
            stft = librosa.stft(audio, n_fft=2048, hop_length=512)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            noise_frames = magnitude[:, :10]
            noise_estimate = np.mean(noise_frames, axis=1, keepdims=True)
            
            enhanced_magnitude = magnitude - alpha * noise_estimate
            enhanced_magnitude = np.maximum(enhanced_magnitude, beta * magnitude)
            
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft, hop_length=512)
            
            return enhanced_audio.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Spectral subtraction failed: {e}")
            return audio
    
    def multi_band_spectral_subtraction(self, audio: np.ndarray) -> np.ndarray:
        """Multi-Band Spectral Subtraction"""
        try:
            print("🔬 Applying Multi-Band Spectral Subtraction...")
            
            bands = [
                (0, 500, 2.5, 0.02),
                (500, 1500, 2.0, 0.01),
                (1500, 4000, 1.8, 0.005),
                (4000, 8000, 2.2, 0.015)
            ]
            
            stft = librosa.stft(audio, n_fft=2048, hop_length=512)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=2048)
            enhanced_magnitude = magnitude.copy()
            
            for low_freq, high_freq, alpha, beta in bands:
                low_bin = np.argmin(np.abs(freqs - low_freq))
                high_bin = np.argmin(np.abs(freqs - high_freq))
                
                band_magnitude = magnitude[low_bin:high_bin, :]
                band_noise = np.mean(band_magnitude[:, :10], axis=1, keepdims=True)
                
                enhanced_band = band_magnitude - alpha * band_noise
                enhanced_band = np.maximum(enhanced_band, beta * band_magnitude)
                
                enhanced_magnitude[low_bin:high_bin, :] = enhanced_band
            
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft, hop_length=512)
            
            return enhanced_audio.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Multi-Band Spectral Subtraction failed: {e}")
            return audio
    
    def wiener_filtering(self, audio: np.ndarray) -> np.ndarray:
        """Wiener filtering"""
        try:
            print("🔧 Applying Wiener filtering...")
            
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
            
            segment_length = 4096
            overlap = 1024
            enhanced_audio = np.zeros_like(audio)
            
            for i in range(0, len(audio) - segment_length + 1, segment_length - overlap):
                segment = audio[i:i + segment_length]
                enhanced_segment = wiener(segment, noise=noise_power)
                
                if i == 0:
                    enhanced_audio[i:i + segment_length] = enhanced_segment
                else:
                    blend_start = i
                    blend_end = i + overlap
                    
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
        """MMSE-STSA Estimator"""
        try:
            print("🔬 Applying MMSE-STSA estimator...")
            
            stft = librosa.stft(audio, n_fft=1024, hop_length=256)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            noise_magnitude = np.mean(magnitude[:, :10], axis=1, keepdims=True)
            noise_power = noise_magnitude ** 2
            
            alpha = 0.98
            signal_power = magnitude ** 2
            
            gamma_k = np.maximum(signal_power / noise_power - 1, 0.1)
            nu_k = gamma_k / (1 + gamma_k)
            
            def modified_bessel_i0(x):
                return np.exp(x) / np.sqrt(2 * np.pi * x) * (1 + 1/(8*x))
            
            def modified_bessel_i1(x):
                return modified_bessel_i0(x) * (1 - 1/(2*x))
            
            v_k = nu_k * gamma_k / (1 + gamma_k)
            v_k = np.clip(v_k, 0.001, 10)
            
            G_k = nu_k * np.exp(-v_k/2) * ((1 + v_k) * modified_bessel_i0(v_k/2) + v_k * modified_bessel_i1(v_k/2))
            
            enhanced_magnitude = G_k * magnitude
            
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft, hop_length=256)
            
            return enhanced_audio.astype(np.float32)
            
        except Exception as e:
            print(f"❌ MMSE-STSA failed: {e}")
            return self.spectral_subtraction(audio)
    
    def mmse_lsa_estimator(self, audio: np.ndarray) -> np.ndarray:
        """MMSE-LSA Estimator"""
        try:
            print("🔬 Applying MMSE-LSA estimator...")
            
            stft = librosa.stft(audio, n_fft=1024, hop_length=256)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            log_magnitude = np.log(magnitude + 1e-10)
            noise_log_magnitude = np.mean(log_magnitude[:, :10], axis=1, keepdims=True)
            
            snr = log_magnitude - noise_log_magnitude
            
            alpha = 2.0
            beta = 0.1
            
            G = np.exp(alpha * snr / (1 + np.exp(alpha * snr)))
            G = np.maximum(G, beta)
            
            enhanced_log_magnitude = log_magnitude + np.log(G + 1e-10)
            enhanced_magnitude = np.exp(enhanced_log_magnitude)
            
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft, hop_length=256)
            
            return enhanced_audio.astype(np.float32)
            
        except Exception as e:
            print(f"❌ MMSE-LSA failed: {e}")
            return self.spectral_subtraction(audio)
    
    def om_lsa_estimator(self, audio: np.ndarray) -> np.ndarray:
        """OM-LSA Estimator"""
        try:
            print("🔬 Applying OM-LSA estimator...")
            
            stft = librosa.stft(audio, n_fft=1024, hop_length=256)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            alpha_d = 0.95
            alpha_s = 0.9
            beta = 0.005
            
            num_frames = magnitude.shape[1]
            num_bins = magnitude.shape[0]
            
            noise_power = np.mean(magnitude[:, :10] ** 2, axis=1, keepdims=True)
            
            gamma_k = np.zeros_like(magnitude)
            xi_k = np.ones_like(magnitude)
            p_k = np.zeros_like(magnitude)
            
            enhanced_magnitude = np.zeros_like(magnitude)
            
            for frame in range(num_frames):
                gamma_k[:, frame:frame+1] = magnitude[:, frame:frame+1] ** 2 / noise_power
                
                if frame > 0:
                    xi_k_dd = alpha_d * (enhanced_magnitude[:, frame-1:frame] ** 2 / noise_power) + \
                             (1 - alpha_d) * np.maximum(gamma_k[:, frame:frame+1] - 1, 0.1)
                    xi_k[:, frame:frame+1] = xi_k_dd
                
                v_k = gamma_k[:, frame:frame+1] * xi_k[:, frame:frame+1] / (1 + xi_k[:, frame:frame+1])
                
                p_k[:, frame:frame+1] = 1 / (1 + np.exp(-2 * (v_k - 1)))
                
                G_k = xi_k[:, frame:frame+1] / (1 + xi_k[:, frame:frame+1])
                
                G_k = p_k[:, frame:frame+1] * G_k + (1 - p_k[:, frame:frame+1]) * beta
                
                G_k = np.maximum(G_k, beta)
                
                enhanced_magnitude[:, frame:frame+1] = G_k * magnitude[:, frame:frame+1]
            
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft, hop_length=256)
            
            return enhanced_audio.astype(np.float32)
            
        except Exception as e:
            print(f"❌ OM-LSA failed: {e}")
            return self.mmse_stsa_estimator(audio)
    
    # FREQUENCY DOMAIN FILTERING
    
    def comprehensive_frequency_filtering(self, audio: np.ndarray) -> np.ndarray:
        """Frequency filtering with FIXED parameters"""
        try:
            print("🎵 Applying frequency filtering...")
            
            # High-pass filter
            high_cutoff = 85
            high_b, high_a = butter(4, high_cutoff, btype='high', fs=self.sample_rate)
            audio = filtfilt(high_b, high_a, audio)
            
            # Low-pass filter (FIXED: 7900 Hz instead of 8000 Hz)
            low_cutoff = 7900
            low_b, low_a = butter(4, low_cutoff, btype='low', fs=self.sample_rate)
            audio = filtfilt(low_b, low_a, audio)
            
            return audio.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Frequency filtering failed: {e}")
            return audio
    
    def adaptive_filtering(self, audio: np.ndarray) -> np.ndarray:
        """Adaptive filtering with LMS"""
        try:
            print("🔧 Applying adaptive filtering...")
            
            filter_length = 32
            mu = 0.01
            
            w = np.zeros(filter_length)
            filtered_audio = np.zeros_like(audio)
            
            for n in range(filter_length, len(audio)):
                x = audio[n-filter_length:n][::-1]
                y = np.dot(w, x)
                filtered_audio[n] = y
                
                d = audio[n-1] if n > 0 else 0
                e = d - y
                
                w = w + mu * e * x
            
            filtered_audio[:filter_length] = audio[:filter_length]
            
            return filtered_audio.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Adaptive filtering failed: {e}")
            return audio
    
    # TIME-FREQUENCY DOMAIN PROCESSING
    
    def da_stft_processing(self, audio: np.ndarray) -> np.ndarray:
        """DA-STFT processing with FIXED hann window"""
        try:
            print("🔬 Applying DA-STFT processing...")
            
            frame_energy = librosa.feature.rms(y=audio, frame_length=1024, hop_length=512)[0]
            avg_energy = np.mean(frame_energy)
            
            if avg_energy > 0.1:
                n_fft = 2048
            elif avg_energy > 0.05:
                n_fft = 1024
            else:
                n_fft = 512
            
            hop_length = n_fft // 4
            
            # FIXED: Use hann instead of hanning
            window = hann(n_fft)
            stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, window=window)
            
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            enhanced_magnitude = magnitude.copy()
            
            freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=n_fft)
            
            for i, freq in enumerate(freqs):
                if 85 <= freq <= 4000:
                    enhanced_magnitude[i, :] *= 1.1
                elif freq > 6000:
                    enhanced_magnitude[i, :] *= 0.8
            
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft, hop_length=hop_length, window=window)
            
            return enhanced_audio.astype(np.float32)
            
        except Exception as e:
            print(f"❌ DA-STFT processing failed: {e}")
            return audio
    
    def time_frequency_masking(self, audio: np.ndarray) -> np.ndarray:
        """Time-frequency masking"""
        try:
            print("🎭 Applying time-frequency masking...")
            
            stft = librosa.stft(audio, n_fft=1024, hop_length=256)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            noise_estimate = np.mean(magnitude[:, :10], axis=1, keepdims=True)
            snr = magnitude / (noise_estimate + 1e-10)
            
            threshold = 2.0
            binary_mask = (snr > threshold).astype(float)
            
            from scipy.ndimage import gaussian_filter
            smooth_mask = gaussian_filter(binary_mask, sigma=1.0)
            
            soft_mask = smooth_mask * 0.9 + 0.1
            
            enhanced_magnitude = magnitude * soft_mask
            
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft, hop_length=256)
            
            return enhanced_audio.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Time-frequency masking failed: {e}")
            return audio
    
    def frame_based_processing(self, audio: np.ndarray) -> np.ndarray:
        """Frame-based processing"""
        try:
            print("📊 Applying frame-based processing...")
            
            frame_length = 1024
            hop_length = 256
            overlap = frame_length - hop_length
            
            # FIXED: Use hann instead of hanning
            window = hann(frame_length)
            
            enhanced_audio = np.zeros_like(audio)
            
            for i in range(0, len(audio) - frame_length + 1, hop_length):
                frame = audio[i:i + frame_length] * window
                
                frame_fft = np.fft.fft(frame)
                magnitude = np.abs(frame_fft)
                phase = np.angle(frame_fft)
                
                enhanced_magnitude = magnitude ** 0.9
                
                enhanced_fft = enhanced_magnitude * np.exp(1j * phase)
                enhanced_frame = np.real(np.fft.ifft(enhanced_fft))
                
                enhanced_frame *= window
                enhanced_audio[i:i + frame_length] += enhanced_frame
            
            return enhanced_audio.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Frame-based processing failed: {e}")
            return audio
    
    # PREPROCESSING AND NORMALIZATION
    
    def z_score_min_max_normalization(self, audio: np.ndarray) -> np.ndarray:
        """Z-score Min-Max normalization"""
        try:
            print("📊 Applying Z-score Min-Max normalization...")
            
            audio_zscore = zscore(audio)
            audio_zscore = np.nan_to_num(audio_zscore, nan=0.0, posinf=1.0, neginf=-1.0)
            
            audio_normalized = self.minmax_scaler.fit_transform(audio_zscore.reshape(-1, 1)).flatten()
            audio_normalized = audio_normalized * 1.6 - 0.8
            
            return audio_normalized.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Z-score Min-Max normalization failed: {e}")
            return librosa.util.normalize(audio).astype(np.float32)
    
    def dynamic_range_compression(self, audio: np.ndarray) -> np.ndarray:
        """Dynamic range compression"""
        try:
            print("📊 Applying dynamic range compression...")
            
            threshold = 0.3
            ratio = 4.0
            attack_time = 0.003
            release_time = 0.100
            
            attack_samples = int(attack_time * self.sample_rate)
            release_samples = int(release_time * self.sample_rate)
            
            envelope = 0.0
            compressed_audio = np.zeros_like(audio)
            
            for i, sample in enumerate(audio):
                input_level = abs(sample)
                
                if input_level > envelope:
                    envelope += (input_level - envelope) / attack_samples
                else:
                    envelope += (input_level - envelope) / release_samples
                
                if envelope > threshold:
                    excess = envelope - threshold
                    compressed_excess = excess / ratio
                    gain = (threshold + compressed_excess) / envelope if envelope > 0 else 1.0
                else:
                    gain = 1.0
                
                compressed_audio[i] = sample * gain
            
            return compressed_audio.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Dynamic range compression failed: {e}")
            return audio
    
    def noise_gating(self, audio: np.ndarray, threshold_db: float = -40.0) -> np.ndarray:
        """Noise gating"""
        try:
            print("🚪 Applying noise gating...")
            
            threshold_linear = 10 ** (threshold_db / 20.0)
            
            frame_length = 1024
            hop_length = 256
            
            rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
            
            rms_expanded = np.repeat(rms, hop_length)
            if len(rms_expanded) > len(audio):
                rms_expanded = rms_expanded[:len(audio)]
            elif len(rms_expanded) < len(audio):
                rms_expanded = np.pad(rms_expanded, (0, len(audio) - len(rms_expanded)), mode='edge')
            
            gate = (rms_expanded > threshold_linear).astype(float)
            
            gate_smooth = gaussian_filter1d(gate, sigma=2.0)
            
            min_gain = 0.1
            gate_gain = gate_smooth * (1 - min_gain) + min_gain
            
            gated_audio = audio * gate_gain
            
            return gated_audio.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Noise gating failed: {e}")
            return audio
    
    def temporal_smoothing(self, audio: np.ndarray, window_size: int = 5) -> np.ndarray:
        """Temporal smoothing"""
        try:
            print("📈 Applying temporal smoothing...")
            
            if window_size > len(audio):
                window_size = len(audio) // 10
            
            kernel = np.ones(window_size) / window_size
            
            smoothed_audio = np.convolve(audio, kernel, mode='same')
            
            return smoothed_audio.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Temporal smoothing failed: {e}")
            return audio
    
    def frame_averaging(self, audio: np.ndarray, num_frames: int = 3) -> np.ndarray:
        """Frame averaging"""
        try:
            print("📊 Applying frame averaging...")
            
            frame_length = 1024
            hop_length = 512
            
            frames = []
            for i in range(0, len(audio) - frame_length + 1, hop_length):
                frame = audio[i:i + frame_length]
                frames.append(frame)
            
            if len(frames) < num_frames:
                return audio
            
            averaged_audio = np.zeros_like(audio)
            
            for i in range(len(frames)):
                start_idx = i * hop_length
                end_idx = start_idx + frame_length
                
                start_frame = max(0, i - num_frames // 2)
                end_frame = min(len(frames), start_frame + num_frames)
                
                avg_frame = np.mean(frames[start_frame:end_frame], axis=0)
                
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
        """Signal Subspace Approach"""
        try:
            print("🔬 Applying Signal Subspace Approach...")
            
            frame_length = 256
            hop_length = 128
            
            frames = []
            for i in range(0, len(audio) - frame_length + 1, hop_length):
                frames.append(audio[i:i + frame_length])
            
            if len(frames) < 2:
                return audio
            
            X = np.array(frames).T
            
            U, s, Vt = svd(X, full_matrices=False)
            
            total_energy = np.sum(s**2)
            cumulative_energy = np.cumsum(s**2)
            
            signal_dim = np.argmax(cumulative_energy / total_energy > 0.95) + 1
            signal_dim = max(1, min(signal_dim, len(s) // 2))
            
            X_clean = U[:, :signal_dim] @ np.diag(s[:signal_dim]) @ Vt[:signal_dim, :]
            
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
        """Noise profile analysis"""
        try:
            print("🔍 Performing noise profile analysis...")
            
            vad_result, vad_stats = self.vad.detect_voice_activity(audio)
            
            hop_length = 256
            vad_expanded = np.repeat(vad_result, hop_length)
            
            if len(vad_expanded) > len(audio):
                vad_expanded = vad_expanded[:len(audio)]
            elif len(vad_expanded) < len(audio):
                vad_expanded = np.pad(vad_expanded, (0, len(audio) - len(vad_expanded)), mode='edge')
            
            noise_regions = ~vad_expanded.astype(bool)
            
            noise_profile = {}
            
            if np.any(noise_regions):
                noise_samples = audio[noise_regions]
                
                noise_profile['mean'] = np.mean(noise_samples)
                noise_profile['std'] = np.std(noise_samples)
                noise_profile['rms'] = np.sqrt(np.mean(noise_samples**2))
                noise_profile['peak'] = np.max(np.abs(noise_samples))
                
                noise_stft = librosa.stft(noise_samples[:min(len(noise_samples), 8192)])
                noise_magnitude = np.abs(noise_stft)
                noise_profile['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(S=noise_magnitude))
                noise_profile['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(S=noise_magnitude))
                
                if noise_profile['spectral_centroid'] > 3000:
                    noise_profile['type'] = 'high_frequency'
                elif noise_profile['spectral_centroid'] < 1000:
                    noise_profile['type'] = 'low_frequency'
                else:
                    noise_profile['type'] = 'broadband'
                
                enhanced_audio = self.targeted_noise_reduction(audio, noise_profile)
                
            else:
                noise_profile['type'] = 'unknown'
                enhanced_audio = audio
            
            return enhanced_audio.astype(np.float32), noise_profile
            
        except Exception as e:
            print(f"❌ Noise profile analysis failed: {e}")
            return audio, {}
    
    def targeted_noise_reduction(self, audio: np.ndarray, noise_profile: Dict) -> np.ndarray:
        """Targeted noise reduction"""
        try:
            noise_type = noise_profile.get('type', 'broadband')
            
            if noise_type == 'high_frequency':
                cutoff = 6000
                b, a = butter(6, cutoff, btype='low', fs=self.sample_rate)
                audio = filtfilt(b, a, audio)
                
            elif noise_type == 'low_frequency':
                cutoff = 120
                b, a = butter(6, cutoff, btype='high', fs=self.sample_rate)
                audio = filtfilt(b, a, audio)
                
            else:
                audio = self.comprehensive_frequency_filtering(audio)
            
            return audio
            
        except Exception as e:
            print(f"❌ Targeted noise reduction failed: {e}")
            return audio
    
    def snr_enhancement(self, audio: np.ndarray) -> Tuple[np.ndarray, float]:
        """SNR enhancement"""
        try:
            print("📊 Applying SNR enhancement...")
            
            initial_snr = self.measure_snr(audio)
            
            enhanced_audio = audio.copy()
            
            enhanced_audio = self.spectral_subtraction(enhanced_audio, alpha=1.5, beta=0.05)
            enhanced_audio = self.wiener_filtering(enhanced_audio)
            enhanced_audio = self.dynamic_range_compression(enhanced_audio)
            
            final_snr = self.measure_snr(enhanced_audio)
            snr_improvement = final_snr - initial_snr
            
            print(f"📊 SNR improved from {initial_snr:.2f} dB to {final_snr:.2f} dB (+{snr_improvement:.2f} dB)")
            
            return enhanced_audio.astype(np.float32), snr_improvement
            
        except Exception as e:
            print(f"❌ SNR enhancement failed: {e}")
            return audio, 0.0
    
    def advanced_vad_enhancement(self, audio: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Advanced VAD enhancement"""
        try:
            print("🎤 Advanced VAD enhancement...")
            
            vad_result, vad_stats = self.vad.detect_voice_activity(audio)
            
            hop_length = 256
            vad_expanded = np.repeat(vad_result, hop_length)
            
            if len(vad_expanded) > len(audio):
                vad_expanded = vad_expanded[:len(audio)]
            elif len(vad_expanded) < len(audio):
                vad_expanded = np.pad(vad_expanded, (0, len(audio) - len(vad_expanded)), mode='edge')
            
            enhanced_audio = audio.copy()
            voice_regions = vad_expanded.astype(bool)
            
            if np.any(voice_regions):
                enhanced_audio[voice_regions] *= 1.05
            
            noise_regions = ~voice_regions
            if np.any(noise_regions):
                enhanced_audio[noise_regions] *= 0.95
            
            return enhanced_audio.astype(np.float32), vad_stats
            
        except Exception as e:
            print(f"❌ Advanced VAD enhancement failed: {e}")
            return audio, {}
    
    def measure_snr(self, audio: np.ndarray) -> float:
        """Measure SNR"""
        try:
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
            
            frame_energy = librosa.feature.rms(y=audio, frame_length=1024, hop_length=512)[0]
            signal_power = np.percentile(frame_energy, 90)**2
            noise_power = np.percentile(frame_energy, 10)**2
            
            if noise_power > 0:
                snr = 10 * np.log10(signal_power / noise_power)
                return snr
            
            return 20.0
            
        except Exception as e:
            print(f"❌ SNR measurement failed: {e}")
            return 20.0
    
    # USER-SELECTABLE PIPELINE
    
    def user_selectable_enhancement(self, audio: np.ndarray, selected_methods: Dict, enhancement_level: str = "moderate") -> Tuple[np.ndarray, Dict]:
        """USER-SELECTABLE: Enhancement pipeline with checkbox controls"""
        original_audio = audio.copy()
        stats = {'enhancement_level': enhancement_level, 'selected_methods': selected_methods}
        
        try:
            print(f"🚀 Starting USER-SELECTABLE enhancement pipeline ({enhancement_level})...")
            print("✅ Applying only user-selected methods")
            
            initial_snr = self.measure_snr(audio)
            stats['initial_snr'] = initial_snr
            stats['original_length'] = len(audio) / self.sample_rate
            
            print(f"📊 Initial SNR: {initial_snr:.2f} dB")
            
            # STAGE 1: PREPROCESSING AND NORMALIZATION
            if selected_methods.get('preprocessing_methods'):
                print("🔧 STAGE 1: User-selected preprocessing methods")
                
                if "Z-score Min-Max Normalization" in selected_methods['preprocessing_methods']:
                    audio = self.z_score_min_max_normalization(audio)
                
                if "Noise Gating" in selected_methods['preprocessing_methods']:
                    audio = self.noise_gating(audio)
                
                if "Dynamic Range Compression" in selected_methods['preprocessing_methods']:
                    audio = self.dynamic_range_compression(audio)
                
                if "Temporal Smoothing" in selected_methods['preprocessing_methods']:
                    audio = self.temporal_smoothing(audio)
                
                if "Frame Averaging" in selected_methods['preprocessing_methods']:
                    audio = self.frame_averaging(audio)
            
            # STAGE 2: FREQUENCY DOMAIN FILTERING
            if selected_methods.get('frequency_methods'):
                print("🎵 STAGE 2: User-selected frequency domain methods")
                
                if "Comprehensive Frequency Filtering" in selected_methods['frequency_methods']:
                    audio = self.comprehensive_frequency_filtering(audio)
                
                if "Adaptive Filtering" in selected_methods['frequency_methods']:
                    audio = self.adaptive_filtering(audio)
            
            # STAGE 3: SPECTRAL DOMAIN METHODS
            if selected_methods.get('spectral_methods'):
                print("🔬 STAGE 3: User-selected spectral domain methods")
                
                if "Spectral Subtraction" in selected_methods['spectral_methods']:
                    audio = self.spectral_subtraction(audio)
                
                if "Multi-Band Spectral Subtraction" in selected_methods['spectral_methods']:
                    audio = self.multi_band_spectral_subtraction(audio)
                
                if "Wiener Filtering" in selected_methods['spectral_methods']:
                    audio = self.wiener_filtering(audio)
                
                if "MMSE-STSA Estimator" in selected_methods['spectral_methods']:
                    audio = self.mmse_stsa_estimator(audio)
                
                if "MMSE-LSA Estimator" in selected_methods['spectral_methods']:
                    audio = self.mmse_lsa_estimator(audio)
                
                if "OM-LSA Estimator" in selected_methods['spectral_methods']:
                    audio = self.om_lsa_estimator(audio)
            
            # STAGE 4: TIME-FREQUENCY DOMAIN PROCESSING
            if selected_methods.get('time_frequency_methods'):
                print("🔬 STAGE 4: User-selected time-frequency methods")
                
                if "DA-STFT Processing" in selected_methods['time_frequency_methods']:
                    audio = self.da_stft_processing(audio)
                
                if "Time-Frequency Masking" in selected_methods['time_frequency_methods']:
                    audio = self.time_frequency_masking(audio)
                
                if "Frame-Based Processing" in selected_methods['time_frequency_methods']:
                    audio = self.frame_based_processing(audio)
            
            # STAGE 5: ADVANCED METHODS
            if selected_methods.get('advanced_methods'):
                print("🔬 STAGE 5: User-selected advanced methods")
                
                if "Signal Subspace Approach" in selected_methods['advanced_methods']:
                    audio = self.signal_subspace_approach(audio)
                
                if "Noise Profile Analysis" in selected_methods['advanced_methods']:
                    audio, noise_profile = self.noise_profile_analysis(audio)
                    stats.update({'noise_profile': noise_profile})
                
                if "SNR Enhancement" in selected_methods['advanced_methods']:
                    audio, snr_improvement = self.snr_enhancement(audio)
                    stats['snr_improvement'] = snr_improvement
                
                if "Advanced VAD Enhancement" in selected_methods['advanced_methods']:
                    audio, vad_stats = self.advanced_vad_enhancement(audio)
                    stats.update(vad_stats)
            
            # Final normalization for ASR
            audio = librosa.util.normalize(audio)
            audio = np.clip(audio, -0.99, 0.99)
            
            # Final quality assessment
            final_snr = self.measure_snr(audio)
            stats['final_snr'] = final_snr
            stats['total_snr_improvement'] = final_snr - initial_snr
            stats['final_rms'] = np.sqrt(np.mean(audio**2))
            
            print(f"✅ USER-SELECTABLE enhancement completed")
            print(f"📊 SNR improvement: {stats['total_snr_improvement']:.2f} dB")
            print(f"📊 Final RMS level: {stats['final_rms']:.4f} (ASR-optimized)")
            
            return audio.astype(np.float32), stats
            
        except Exception as e:
            print(f"❌ User-selectable enhancement failed: {e}")
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

class UserSelectableSpeechTranscriber:
    """USER-SELECTABLE: Audio transcriber with checkbox-controlled preprocessing"""
    
    def __init__(self, model_path: str, use_quantization: bool = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = None
        self.processor = None
        self.audio_enhancer = UserSelectableSpeechEnhancer(SAMPLE_RATE)
        self.text_chunker = SmartTextChunker()
        self.chunk_count = 0
        self.temp_files = []
        
        print(f"🖥️ Using device: {self.device}")
        print(f"🚀 USER-SELECTABLE speech enhancement enabled")
        print(f"✅ All methods available as checkbox controls")
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
            print(f"✅ USER-SELECTABLE model loaded in {loading_time:.1f} seconds")
            
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
        
        print(f"✅ Created {len(chunks)} user-selectable processing chunks")
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
            print(f"❌ User-selectable transcription error: {str(e)}")
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
            print("🌐 Starting user-selectable text translation...")
            
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
            print(f"❌ User-selectable translation error: {str(e)}")
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
    
    def transcribe_with_user_selected_enhancement(self, audio_path: str, selected_methods: Dict, language: str = "auto", 
                                                enhancement_level: str = "moderate") -> Tuple[str, str, str, Dict]:
        try:
            print(f"🚀 Starting USER-SELECTABLE transcription with selected preprocessing methods...")
            print(f"🔧 Enhancement level: {enhancement_level}")
            print(f"🌍 Language: {language}")
            print(f"✅ Selected methods: {len([item for sublist in selected_methods.values() for item in sublist])} total")
            print(f"⏱️ Chunk timeout: {CHUNK_TIMEOUT} seconds")
            
            OptimizedMemoryManager.log_memory_status("Initial", force_log=True)
            
            try:
                audio_info = sf.info(audio_path)
                duration_seconds = audio_info.frames / audio_info.samplerate
                print(f"⏱️ Audio duration: {duration_seconds:.2f} seconds")
                
                max_duration = 900
                if duration_seconds > max_duration:
                    print(f"⚠️ Processing first {max_duration/60:.1f} minutes")
                    audio_array, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True, duration=max_duration)
                else:
                    audio_array, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
                    
            except Exception as e:
                print(f"❌ Audio loading failed: {e}")
                return f"❌ Audio loading failed: {e}", audio_path, audio_path, {}
            
            # USER-SELECTABLE: Apply only selected methods
            enhanced_audio, stats = self.audio_enhancer.user_selectable_enhancement(
                audio_array, selected_methods, enhancement_level
            )
            
            enhanced_path = tempfile.mktemp(suffix="_user_selected_enhanced.wav")
            original_path = tempfile.mktemp(suffix="_original.wav")
            
            sf.write(enhanced_path, enhanced_audio, SAMPLE_RATE)
            sf.write(original_path, audio_array, SAMPLE_RATE)
            
            print("✂️ Creating user-selectable processing chunks...")
            chunks = self.create_speech_chunks(enhanced_audio)
            
            if not chunks:
                return "❌ No valid chunks created", original_path, enhanced_path, stats
            
            transcriptions = []
            successful = 0
            timeout_count = 0
            
            start_time = time.time()
            
            for i, (chunk, start_time_chunk, end_time_chunk) in enumerate(chunks):
                print(f"🚀 Processing user-selected chunk {i+1}/{len(chunks)} ({start_time_chunk:.1f}s-{end_time_chunk:.1f}s)")
                
                try:
                    transcription = self.transcribe_chunk_with_timeout(chunk, language)
                    transcriptions.append(transcription)
                    
                    if transcription == "Input Audio Very noisy. Unable to extract details.":
                        timeout_count += 1
                        print(f"⏱️ Chunk {i+1}: Timeout due to noisy audio")
                    elif not transcription.startswith('['):
                        successful += 1
                        print(f"✅ Chunk {i+1}: {transcription[:50]}...")
                    else:
                        print(f"⚠️ Chunk {i+1}: {transcription}")
                
                except Exception as e:
                    print(f"❌ Chunk {i+1} failed: {e}")
                    transcriptions.append(f"[CHUNK_{i+1}_ERROR]")
                
                if i % CHECK_MEMORY_FREQUENCY == 0:
                    OptimizedMemoryManager.fast_cleanup()
            
            processing_time = time.time() - start_time
            
            print("🔗 Merging user-selected transcriptions...")
            final_transcription = self.merge_transcriptions_with_timeout_info(
                transcriptions, timeout_count
            )
            
            print(f"✅ USER-SELECTABLE transcription completed in {processing_time:.2f}s")
            print(f"📊 Success rate: {successful}/{len(chunks)} ({successful/len(chunks)*100:.1f}%)")
            if timeout_count > 0:
                print(f"⏱️ Timeout chunks: {timeout_count}/{len(chunks)} (very noisy audio)")
            
            return final_transcription, original_path, enhanced_path, stats
                
        except Exception as e:
            error_msg = f"❌ User-selectable transcription failed: {e}"
            print(error_msg)
            OptimizedMemoryManager.fast_cleanup()
            return error_msg, audio_path, audio_path, {}
        finally:
            for temp_file in self.temp_files:
                AudioHandler.cleanup_temp_file(temp_file)
            self.temp_files.clear()
    
    def merge_transcriptions_with_timeout_info(self, transcriptions: List[str], timeout_count: int) -> str:
        if not transcriptions:
            return "No transcriptions generated"
        
        valid_transcriptions = []
        error_count = 0
        noisy_timeout_count = 0
        
        for i, text in enumerate(transcriptions):
            if text == "Input Audio Very noisy. Unable to extract details.":
                noisy_timeout_count += 1
            elif text.startswith('[') and text.endswith(']'):
                error_count += 1
            else:
                cleaned_text = text.strip()
                if cleaned_text and len(cleaned_text) > 1:
                    valid_transcriptions.append(cleaned_text)
        
        if not valid_transcriptions:
            if noisy_timeout_count > 0:
                return f"❌ All {len(transcriptions)} chunks timed out due to very noisy audio. Unable to extract any details from this audio."
            else:
                return f"❌ No valid transcriptions from {len(transcriptions)} chunks."
        
        merged_text = " ".join(valid_transcriptions)
        
        total_chunks = len(transcriptions)
        success_rate = (len(valid_transcriptions) / total_chunks) * 100
        
        summary_parts = []
        if len(valid_transcriptions) > 0:
            summary_parts.append(f"{len(valid_transcriptions)} chunks successful")
        if error_count > 0:
            summary_parts.append(f"{error_count} chunks had errors")
        if noisy_timeout_count > 0:
            summary_parts.append(f"{noisy_timeout_count} chunks too noisy (timed out)")
        
        if error_count > 0 or noisy_timeout_count > 0:
            merged_text += f"\n\n[User-Selected Processing Summary: {', '.join(summary_parts)} - {success_rate:.1f}% success rate]"
            
            if noisy_timeout_count > 0:
                merged_text += f"\n[Note: {noisy_timeout_count} chunks were too noisy and timed out after {CHUNK_TIMEOUT} seconds each]"
        
        return merged_text.strip()
    
    def __del__(self):
        for temp_file in self.temp_files:
            AudioHandler.cleanup_temp_file(temp_file)

# Global variables
transcriber = None
log_capture = None

class SafeLogCapture:
    def __init__(self):
        self.log_buffer = []
        self.max_lines = 100
        self.lock = threading.Lock()
    
    def write(self, text):
        if text.strip():
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            
            if "🚀" in text or "USER-SELECTABLE" in text:
                emoji = "🚀"
            elif "⏱️" in text or "timeout" in text.lower() or "noisy" in text.lower():
                emoji = "⏱️"
            elif "🌐" in text or "Translation" in text:
                emoji = "🌐"
            elif "❌" in text or "Error" in text or "failed" in text:
                emoji = "🔴"
            elif "✅" in text or "success" in text or "completed" in text:
                emoji = "🟢"
            elif "⚠️" in text or "Warning" in text:
                emoji = "🟡"
            else:
                emoji = "⚪"
            
            log_entry = f"[{timestamp}] {emoji} {text.strip()}"
            
            with self.lock:
                self.log_buffer.append(log_entry)
                if len(self.log_buffer) > self.max_lines:
                    self.log_buffer.pop(0)
        
        sys.__stdout__.write(text)
    
    def flush(self):
        sys.__stdout__.flush()
    
    def isatty(self):
        return False
    
    def get_logs(self):
        with self.lock:
            return "\n".join(self.log_buffer[-50:]) if self.log_buffer else "🚀 User-selectable system ready..."

def setup_user_selectable_logging():
    logging.basicConfig(
        level=logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.__stdout__)],
        force=True
    )
    
    global log_capture
    log_capture = SafeLogCapture()
    sys.stdout = log_capture

def get_current_logs():
    global log_capture
    if log_capture:
        return log_capture.get_logs()
    return "🚀 User-selectable system initializing..."

def initialize_user_selectable_transcriber():
    global transcriber
    if transcriber is None:
        try:
            print("🚀 Initializing USER-SELECTABLE Speech Enhancement & Transcription System...")
            print("✅ ALL PREPROCESSING METHODS AVAILABLE AS CHECKBOX CONTROLS:")
            print("🔬 Spectral Domain: 6 methods (Spectral Subtraction, MBSS, Wiener, MMSE-STSA, MMSE-LSA, OM-LSA)")
            print("🎵 Frequency Domain: 2 methods (Comprehensive Filtering, Adaptive Filtering)")
            print("🔬 Time-Frequency: 3 methods (DA-STFT, TF Masking, Frame-Based)")
            print("📊 Preprocessing: 5 methods (Z-score, Compression, Gating, Smoothing, Averaging)")
            print("🔬 Advanced: 4 methods (Signal Subspace, Noise Profile, SNR Enhancement, VAD)")
            print(f"⏱️ Chunk timeout: {CHUNK_TIMEOUT} seconds")
            
            transcriber = UserSelectableSpeechTranscriber(model_path=MODEL_PATH, use_quantization=True)
            return "✅ USER-SELECTABLE transcription system ready! All methods available as checkboxes."
        except Exception as e:
            try:
                print("🔄 Retrying without quantization...")
                transcriber = UserSelectableSpeechTranscriber(model_path=MODEL_PATH, use_quantization=False)
                return "✅ USER-SELECTABLE system loaded (standard precision)!"
            except Exception as e2:
                error_msg = f"❌ User-selectable system failure: {str(e2)}"
                print(error_msg)
                return error_msg
    return "✅ USER-SELECTABLE system already active!"

def transcribe_audio_user_selectable(audio_input, language_choice, enhancement_level, 
                                   spectral_methods, frequency_methods, time_frequency_methods, 
                                   preprocessing_methods, advanced_methods, progress=gr.Progress()):
    global transcriber
    
    if audio_input is None:
        return "❌ Please upload an audio file or record audio.", None, None, "", ""
    
    if transcriber is None:
        return "❌ System not initialized. Please wait for startup.", None, None, "", ""
    
    # Organize selected methods
    selected_methods = {
        'spectral_methods': spectral_methods or [],
        'frequency_methods': frequency_methods or [],
        'time_frequency_methods': time_frequency_methods or [],
        'preprocessing_methods': preprocessing_methods or [],
        'advanced_methods': advanced_methods or []
    }
    
    total_selected = sum(len(methods) for methods in selected_methods.values())
    
    if total_selected == 0:
        return "❌ Please select at least one preprocessing method.", None, None, "", ""
    
    start_time = time.time()
    print(f"🚀 Starting USER-SELECTABLE transcription with {total_selected} selected methods...")
    print(f"🌍 Language: {language_choice}")
    print(f"🔧 Enhancement: {enhancement_level}")
    print(f"⏱️ Timeout per chunk: {CHUNK_TIMEOUT} seconds")
    
    progress(0.1, desc="Initializing USER-SELECTABLE processing...")
    
    temp_audio_path = None
    
    try:
        temp_audio_path = AudioHandler.convert_to_file(audio_input, SAMPLE_RATE)
        
        progress(0.3, desc="Applying USER-SELECTED speech enhancement methods...")
        
        language_code = SUPPORTED_LANGUAGES.get(language_choice, "auto")
        
        progress(0.5, desc="USER-SELECTABLE transcription with timeout protection...")
        
        transcription, original_path, enhanced_path, enhancement_stats = transcriber.transcribe_with_user_selected_enhancement(
            temp_audio_path, selected_methods, language_code, enhancement_level
        )
        
        progress(0.9, desc="Generating USER-SELECTABLE reports...")
        
        enhancement_report = create_user_selectable_enhancement_report(enhancement_stats, enhancement_level, selected_methods)
        
        processing_time = time.time() - start_time
        processing_report = create_user_selectable_processing_report(
            temp_audio_path, language_choice, enhancement_level, 
            processing_time, len(transcription.split()) if isinstance(transcription, str) else 0,
            enhancement_stats, selected_methods
        )
        
        progress(1.0, desc="USER-SELECTABLE processing complete!")
        
        print(f"✅ USER-SELECTABLE transcription completed in {processing_time:.2f}s")
        print(f"📊 Output: {len(transcription.split()) if isinstance(transcription, str) else 0} words")
        
        return transcription, original_path, enhanced_path, enhancement_report, processing_report
        
    except Exception as e:
        error_msg = f"❌ User-selectable system error: {str(e)}"
        print(error_msg)
        OptimizedMemoryManager.fast_cleanup()
        return error_msg, None, None, "", ""
    finally:
        if temp_audio_path:
            AudioHandler.cleanup_temp_file(temp_audio_path)

def translate_transcription_user_selectable(transcription_text, progress=gr.Progress()):
    global transcriber
    
    if not transcription_text or transcription_text.strip() == "":
        return "❌ No transcription text to translate. Please transcribe audio first."
    
    if transcriber is None:
        return "❌ System not initialized. Please wait for system startup."
    
    if transcription_text.startswith("❌") or transcription_text.startswith("["):
        return "❌ Cannot translate error messages or system messages. Please provide valid transcription text."
    
    progress(0.1, desc="Preparing text for user-selectable translation...")
    
    try:
        text_to_translate = transcription_text
        if "\n\n[User-Selected Processing Summary:" in text_to_translate:
            text_to_translate = text_to_translate.split("\n\n[User-Selected Processing Summary:")[0].strip()
        
        progress(0.3, desc="Creating smart text chunks...")
        
        start_time = time.time()
        translated_text = transcriber.translate_text_chunks(text_to_translate)
        translation_time = time.time() - start_time
        
        progress(0.9, desc="Finalizing user-selectable translation...")
        
        if not translated_text.startswith('['):
            translated_text += f"\n\n[User-Selectable Translation completed in {translation_time:.2f}s using smart chunking]"
        
        progress(1.0, desc="User-selectable translation complete!")
        
        print(f"✅ User-selectable translation completed in {translation_time:.2f}s")
        
        return translated_text
        
    except Exception as e:
        error_msg = f"❌ User-selectable translation failed: {str(e)}"
        print(error_msg)
        OptimizedMemoryManager.fast_cleanup()
        return error_msg

def create_user_selectable_enhancement_report(stats: Dict, level: str, selected_methods: Dict) -> str:
    if not stats:
        return "⚠️ Enhancement statistics not available"
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Count selected methods
    method_counts = {key: len(methods) for key, methods in selected_methods.items()}
    total_methods = sum(method_counts.values())
    
    report = f"""
🚀 USER-SELECTABLE SPEECH ENHANCEMENT REPORT
===========================================
Timestamp: {timestamp}
Enhancement Level: {level.upper()}
Total Selected Methods: {total_methods}

📊 USER-SELECTED QUALITY ANALYSIS:
• Initial SNR: {stats.get('initial_snr', 0):.2f} dB
• Final SNR: {stats.get('final_snr', 0):.2f} dB
• Total SNR Improvement: {stats.get('total_snr_improvement', 0):.2f} dB
• Audio Duration: {stats.get('original_length', 0):.2f} seconds
• Final RMS Energy: {stats.get('final_rms', 0):.4f}

🚀 USER-SELECTED METHODS BY CATEGORY:

🔬 SPECTRAL DOMAIN METHODS ({method_counts['spectral_methods']} selected):
"""
    
    spectral_available = [
        "Spectral Subtraction", "Multi-Band Spectral Subtraction", "Wiener Filtering",
        "MMSE-STSA Estimator", "MMSE-LSA Estimator", "OM-LSA Estimator"
    ]
    
    for method in spectral_available:
        status = "✅ APPLIED" if method in selected_methods.get('spectral_methods', []) else "❌ NOT SELECTED"
        report += f"• {method}: {status}\n"
    
    report += f"""
🎵 FREQUENCY DOMAIN METHODS ({method_counts['frequency_methods']} selected):
"""
    
    frequency_available = ["Comprehensive Frequency Filtering", "Adaptive Filtering"]
    
    for method in frequency_available:
        status = "✅ APPLIED" if method in selected_methods.get('frequency_methods', []) else "❌ NOT SELECTED"
        report += f"• {method}: {status}\n"
    
    report += f"""
🔬 TIME-FREQUENCY DOMAIN METHODS ({method_counts['time_frequency_methods']} selected):
"""
    
    time_freq_available = ["DA-STFT Processing", "Time-Frequency Masking", "Frame-Based Processing"]
    
    for method in time_freq_available:
        status = "✅ APPLIED" if method in selected_methods.get('time_frequency_methods', []) else "❌ NOT SELECTED"
        report += f"• {method}: {status}\n"
    
    report += f"""
📊 PREPROCESSING & NORMALIZATION ({method_counts['preprocessing_methods']} selected):
"""
    
    preprocessing_available = [
        "Z-score Min-Max Normalization", "Dynamic Range Compression", "Noise Gating",
        "Temporal Smoothing", "Frame Averaging"
    ]
    
    for method in preprocessing_available:
        status = "✅ APPLIED" if method in selected_methods.get('preprocessing_methods', []) else "❌ NOT SELECTED"
        report += f"• {method}: {status}\n"
    
    report += f"""
🔬 ADVANCED METHODS ({method_counts['advanced_methods']} selected):
"""
    
    advanced_available = [
        "Signal Subspace Approach", "Noise Profile Analysis", "SNR Enhancement", "Advanced VAD Enhancement"
    ]
    
    for method in advanced_available:
        status = "✅ APPLIED" if method in selected_methods.get('advanced_methods', []) else "❌ NOT SELECTED"
        report += f"• {method}: {status}\n"
    
    report += f"""
🎤 VOICE ACTIVITY ANALYSIS:
• Voice Percentage: {stats.get('voice_percentage', 0):.1f}%
• Voice Score: {stats.get('voice_score', 0):.3f}
• SNR Estimate: {stats.get('snr_estimate', 0):.2f} dB

⏱️ TIMEOUT PROTECTION:
• Chunk Timeout: {CHUNK_TIMEOUT} seconds
• User-Selected Noise Detection: ✅ ACTIVE
• Timeout Messages: ✅ ENABLED

🏆 USER-SELECTABLE ENHANCEMENT SCORE: {min(100, total_methods * 5)}/100

🔧 TECHNICAL SPECIFICATIONS:
• Processing Method: USER-SELECTABLE CHECKBOX-CONTROLLED PIPELINE
• Selected Methods: {total_methods} out of 20 available techniques
• ASR Optimization: Final normalization applied
• Quality Detection: Multi-feature analysis
• Memory Management: GPU-optimized with cleanup
• Error Recovery: Comprehensive fallback systems
"""
    return report

def create_user_selectable_processing_report(audio_path: str, language: str, enhancement: str, 
                                           processing_time: float, word_count: int, stats: Dict, selected_methods: Dict) -> str:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        file_size = os.path.getsize(audio_path) / (1024 * 1024)
        audio_info = f"File size: {file_size:.2f} MB"
    except:
        audio_info = "File info unavailable"
    
    device_info = f"GPU: {torch.cuda.get_device_name()}" if torch.cuda.is_available() else "CPU Processing"
    
    initial_snr = stats.get('initial_snr', 0)
    final_snr = stats.get('final_snr', 0)
    snr_improvement = stats.get('total_snr_improvement', 0)
    voice_percentage = stats.get('voice_percentage', 0)
    final_rms = stats.get('final_rms', 0)
    
    # Count selected methods
    method_counts = {key: len(methods) for key, methods in selected_methods.items()}
    total_methods = sum(method_counts.values())
    
    report = f"""
🚀 USER-SELECTABLE SPEECH TRANSCRIPTION REPORT
=============================================
Generated: {timestamp}

🎵 USER-SELECTABLE AUDIO PROCESSING:
• Source File: {os.path.basename(audio_path)}
• {audio_info}
• Target Language: {language}
• Enhancement Level: {enhancement.upper()}
• Total Selected Methods: {total_methods} out of 20 available

⚡ PERFORMANCE METRICS:
• Processing Time: {processing_time:.2f} seconds
• Words Generated: {word_count}
• Processing Speed: {word_count/processing_time:.1f} words/second
• Processing Device: {device_info}

🚀 USER-SELECTABLE CONFIGURATION:
• Model: Gemma 3N E4B-IT (User-Selectable Enhanced)
• Chunk Size: {CHUNK_SECONDS} seconds (User-Selectable Optimized)
• Chunk Timeout: {CHUNK_TIMEOUT} seconds per chunk
• Overlap: {OVERLAP_SECONDS} seconds (Context Preserving)
• Enhancement Method: USER-SELECTABLE CHECKBOX-CONTROLLED PIPELINE

📊 USER-SELECTED QUALITY TRANSFORMATION:
• Initial SNR: {initial_snr:.2f} dB → {final_snr:.2f} dB
• Total SNR Improvement: {snr_improvement:.2f} dB
• Voice Activity: {voice_percentage:.1f}% of audio
• Final RMS Level: {final_rms:.4f} (ASR-Optimized)
• Enhancement Rating: {'EXCEPTIONAL' if snr_improvement > 10 else 'EXCELLENT' if snr_improvement > 5 else 'VERY GOOD' if snr_improvement > 2 else 'GOOD' if snr_improvement > 0 else 'MAINTAINED'}

🚀 USER-SELECTED METHODS BREAKDOWN:
• Spectral Domain: {method_counts['spectral_methods']}/6 methods selected
• Frequency Domain: {method_counts['frequency_methods']}/2 methods selected
• Time-Frequency: {method_counts['time_frequency_methods']}/3 methods selected
• Preprocessing: {method_counts['preprocessing_methods']}/5 methods selected
• Advanced Methods: {method_counts['advanced_methods']}/4 methods selected

⏱️ TIMEOUT & NOISE HANDLING:
• Timeout Protection: ✅ {CHUNK_TIMEOUT}s per chunk
• User-Selected Quality Detection: ✅ Applied methods analysis
• Timeout Messages: ✅ "Input Audio Very noisy. Unable to extract details."
• Fallback Systems: ✅ User-selected error recovery

🌐 TRANSLATION FEATURES:
• Translation Control: ✅ USER-INITIATED (Optional)
• Smart Text Chunking: ✅ ENABLED
• Context Preservation: ✅ SENTENCE OVERLAP
• Processing Method: ✅ USER-SELECTABLE PIPELINE

📊 USER-SELECTABLE SYSTEM STATUS:
• Enhancement Method: ✅ USER-SELECTABLE CHECKBOX-CONTROLLED PIPELINE
• Selected Methods: ✅ {total_methods} METHODS APPLIED
• ASR Optimization: ✅ FINAL NORMALIZATION
• Timeout Protection: ✅ ACTIVE (75s per chunk)
• Quality Detection: ✅ USER-SELECTED ANALYSIS
• Memory Optimization: ✅ GPU-AWARE CLEANUP
• Error Recovery: ✅ USER-SELECTED FALLBACK SYSTEMS

✅ STATUS: USER-SELECTABLE TRANSCRIPTION COMPLETED
🚀 AUDIO ENHANCEMENT: USER-CONTROLLED CHECKBOX PIPELINE
⏱️ TIMEOUT PROTECTION: 75-SECOND CHUNK SAFETY
🔧 PREPROCESSING: USER-SELECTED METHODS ONLY
📊 ASR OPTIMIZATION: CHECKBOX-CONTROLLED NORMALIZATION
🎯 RELIABILITY: USER-SELECTABLE PROCESSING WITH COMPREHENSIVE FALLBACKS
"""
    return report

def create_user_selectable_interface():
    """Create user-selectable speech enhancement interface with checkbox controls"""
    
    user_selectable_css = """
    :root {
        --primary-color: #0f172a;
        --secondary-color: #1e293b;
        --accent-color: #06b6d4;
        --checkbox-color: #10b981;
        --success-color: #10b981;
        --timeout-color: #f59e0b;
        --translation-color: #3b82f6;
        --bg-primary: #020617;
        --bg-secondary: #0f172a;
        --bg-tertiary: #1e293b;
        --text-primary: #f8fafc;
        --text-secondary: #cbd5e1;
        --border-color: #475569;
    }
    
    .gradio-container {
        background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%) !important;
        font-family: 'Inter', sans-serif !important;
        color: var(--text-primary) !important;
        min-height: 100vh !important;
    }
    
    .user-selectable-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 15%, #06b6d4 30%, #10b981 45%, #f59e0b 60%, #3b82f6 75%, #8b5cf6 90%, #ec4899 100%) !important;
        padding: 60px 40px !important;
        border-radius: 30px !important;
        text-align: center !important;
        margin-bottom: 50px !important;
        box-shadow: 0 30px 60px rgba(6, 182, 212, 0.4) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .user-selectable-title {
        font-size: 4rem !important;
        font-weight: 900 !important;
        color: white !important;
        margin-bottom: 20px !important;
        text-shadow: 0 5px 15px rgba(6, 182, 212, 0.6) !important;
        position: relative !important;
        z-index: 2 !important;
    }
    
    .user-selectable-subtitle {
        font-size: 1.5rem !important;
        color: rgba(255,255,255,0.95) !important;
        font-weight: 600 !important;
        position: relative !important;
        z-index: 2 !important;
    }
    
    .user-selectable-card {
        background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%) !important;
        border: 3px solid var(--accent-color) !important;
        border-radius: 25px !important;
        padding: 35px !important;
        margin: 25px 0 !important;
        box-shadow: 0 20px 40px rgba(6, 182, 212, 0.3) !important;
        transition: all 0.4s ease !important;
    }
    
    .checkbox-group {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(6, 182, 212, 0.1) 100%) !important;
        border: 2px solid var(--checkbox-color) !important;
        border-radius: 20px !important;
        padding: 25px !important;
        margin: 20px 0 !important;
    }
    
    .user-selectable-button {
        background: linear-gradient(135deg, var(--accent-color) 0%, var(--checkbox-color) 100%) !important;
        border: none !important;
        border-radius: 20px !important;
        color: white !important;
        font-weight: 800 !important;
        font-size: 1.3rem !important;
        padding: 20px 40px !important;
        transition: all 0.4s ease !important;
        box-shadow: 0 10px 30px rgba(6, 182, 212, 0.5) !important;
        text-transform: uppercase !important;
        letter-spacing: 1.5px !important;
    }
    
    .translation-button {
        background: linear-gradient(135deg, var(--translation-color) 0%, var(--accent-color) 100%) !important;
        border: none !important;
        border-radius: 18px !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 1.2rem !important;
        padding: 18px 35px !important;
        transition: all 0.4s ease !important;
        box-shadow: 0 10px 30px rgba(59, 130, 246, 0.5) !important;
        text-transform: uppercase !important;
        letter-spacing: 1.2px !important;
    }
    
    .status-user-selectable {
        background: linear-gradient(135deg, var(--success-color), #059669) !important;
        color: white !important;
        padding: 18px 30px !important;
        border-radius: 15px !important;
        font-weight: 700 !important;
        text-align: center !important;
        box-shadow: 0 10px 25px rgba(16, 185, 129, 0.5) !important;
        border: 3px solid rgba(16, 185, 129, 0.4) !important;
    }
    
    .translation-section {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(6, 182, 212, 0.15) 100%) !important;
        border: 3px solid var(--translation-color) !important;
        border-radius: 25px !important;
        padding: 30px !important;
        margin: 25px 0 !important;
        position: relative !important;
    }
    
    .card-header {
        color: var(--accent-color) !important;
        font-size: 1.7rem !important;
        font-weight: 800 !important;
        margin-bottom: 30px !important;
        padding-bottom: 18px !important;
        border-bottom: 4px solid var(--accent-color) !important;
    }
    
    .checkbox-header {
        color: var(--checkbox-color) !important;
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        margin-bottom: 20px !important;
        padding-bottom: 12px !important;
        border-bottom: 3px solid var(--checkbox-color) !important;
    }
    
    .log-user-selectable {
        background: linear-gradient(135deg, rgba(0, 0, 0, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%) !important;
        border: 3px solid var(--accent-color) !important;
        border-radius: 18px !important;
        color: var(--text-secondary) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1rem !important;
        line-height: 1.8 !important;
        padding: 25px !important;
        max-height: 450px !important;
        overflow-y: auto !important;
        white-space: pre-wrap !important;
    }
    """
    
    with gr.Blocks(
        css=user_selectable_css, 
        theme=gr.themes.Base(),
        title="🚀 User-Selectable Speech Enhancement & Transcription"
    ) as interface:
        
        # User-Selectable Header
        gr.HTML("""
        <div class="user-selectable-header">
            <h1 class="user-selectable-title">🚀 USER-SELECTABLE SPEECH ENHANCEMENT</h1>
            <p class="user-selectable-subtitle">Checkbox-Controlled Preprocessing • 20 Available Methods • Custom Pipeline • ASR-Optimized • 75s Timeout</p>
            <div style="margin-top: 25px;">
                <span style="background: rgba(6, 182, 212, 0.25); color: #06b6d4; padding: 12px 24px; border-radius: 30px; margin: 0 10px; font-size: 1.1rem; font-weight: 700;">☑️ CHECKBOX CONTROL</span>
                <span style="background: rgba(16, 185, 129, 0.25); color: #10b981; padding: 12px 24px; border-radius: 30px; margin: 0 10px; font-size: 1.1rem; font-weight: 700;">🔧 20 METHODS</span>
                <span style="background: rgba(59, 130, 246, 0.25); color: #3b82f6; padding: 12px 24px; border-radius: 30px; margin: 0 10px; font-size: 1.1rem; font-weight: 700;">📊 CUSTOM PIPELINE</span>
                <span style="background: rgba(245, 158, 11, 0.25); color: #f59e0b; padding: 12px 24px; border-radius: 30px; margin: 0 10px; font-size: 1.1rem; font-weight: 700;">⏱️ 75s TIMEOUT</span>
            </div>
        </div>
        """)
        
        # System Status
        status_display = gr.Textbox(
            label="🚀 User-Selectable System Status",
            value="Initializing USER-SELECTABLE speech enhancement system with checkbox controls...",
            interactive=False,
            elem_classes="status-user-selectable"
        )
        
        # Main Interface
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML('<div class="user-selectable-card"><div class="card-header">🚀 User-Selectable Control Panel</div>')
                
                audio_input = gr.Audio(
                    label="🎵 Upload Audio File or Record Live",
                    type="filepath"
                )
                
                language_dropdown = gr.Dropdown(
                    choices=list(SUPPORTED_LANGUAGES.keys()),
                    value="🌍 Auto-detect",
                    label="🌍 Language Selection (150+ Supported)",
                    info="All languages with USER-SELECTABLE enhancement"
                )
                
                enhancement_radio = gr.Radio(
                    choices=[
                        ("🟢 Light - USER-SELECTABLE minimal processing", "light"),
                        ("🟡 Moderate - USER-SELECTABLE balanced enhancement", "moderate"), 
                        ("🔴 Aggressive - USER-SELECTABLE maximum processing", "aggressive")
                    ],
                    value="moderate",
                    label="🚀 User-Selectable Enhancement Level",
                    info="Enhancement level affects selected method parameters"
                )
                
                gr.HTML('</div>')
                
                # Checkbox Groups for Method Selection
                gr.HTML('<div class="user-selectable-card"><div class="card-header">☑️ Select Preprocessing Methods</div>')
                
                # Spectral Domain Methods
                gr.HTML('<div class="checkbox-group"><div class="checkbox-header">🔬 Spectral Domain Methods (6 available)</div>')
                spectral_methods = gr.CheckboxGroup(
                    choices=[
                        "Spectral Subtraction",
                        "Multi-Band Spectral Subtraction", 
                        "Wiener Filtering",
                        "MMSE-STSA Estimator",
                        "MMSE-LSA Estimator",
                        "OM-LSA Estimator"
                    ],
                    value=["Spectral Subtraction", "Wiener Filtering"],  # Default selections
                    label="🔬 Spectral Domain Methods",
                    info="Advanced spectral processing techniques"
                )
                gr.HTML('</div>')
                
                # Frequency Domain Methods
                gr.HTML('<div class="checkbox-group"><div class="checkbox-header">🎵 Frequency Domain Methods (2 available)</div>')
                frequency_methods = gr.CheckboxGroup(
                    choices=[
                        "Comprehensive Frequency Filtering",
                        "Adaptive Filtering"
                    ],
                    value=["Comprehensive Frequency Filtering"],  # Default selection
                    label="🎵 Frequency Domain Methods",
                    info="Frequency-based filtering techniques"
                )
                gr.HTML('</div>')
                
                # Time-Frequency Domain Methods
                gr.HTML('<div class="checkbox-group"><div class="checkbox-header">🔬 Time-Frequency Methods (3 available)</div>')
                time_frequency_methods = gr.CheckboxGroup(
                    choices=[
                        "DA-STFT Processing",
                        "Time-Frequency Masking",
                        "Frame-Based Processing"
                    ],
                    value=["Frame-Based Processing"],  # Default selection
                    label="🔬 Time-Frequency Methods",
                    info="Advanced time-frequency processing"
                )
                gr.HTML('</div>')
                
                # Preprocessing & Normalization Methods
                gr.HTML('<div class="checkbox-group"><div class="checkbox-header">📊 Preprocessing & Normalization (5 available)</div>')
                preprocessing_methods = gr.CheckboxGroup(
                    choices=[
                        "Z-score Min-Max Normalization",
                        "Dynamic Range Compression",
                        "Noise Gating",
                        "Temporal Smoothing",
                        "Frame Averaging"
                    ],
                    value=["Dynamic Range Compression", "Noise Gating"],  # Default selections
                    label="📊 Preprocessing Methods",
                    info="Signal conditioning and normalization"
                )
                gr.HTML('</div>')
                
                # Advanced Methods
                gr.HTML('<div class="checkbox-group"><div class="checkbox-header">🔬 Advanced Methods (4 available)</div>')
                advanced_methods = gr.CheckboxGroup(
                    choices=[
                        "Signal Subspace Approach",
                        "Noise Profile Analysis",
                        "SNR Enhancement",
                        "Advanced VAD Enhancement"
                    ],
                    value=["Advanced VAD Enhancement"],  # Default selection
                    label="🔬 Advanced Methods",
                    info="State-of-the-art enhancement techniques"
                )
                gr.HTML('</div>')
                
                transcribe_btn = gr.Button(
                    "🚀 START USER-SELECTABLE TRANSCRIPTION",
                    variant="primary",
                    elem_classes="user-selectable-button",
                    size="lg"
                )
                
                gr.HTML('</div>')
            
            with gr.Column(scale=2):
                gr.HTML('<div class="user-selectable-card"><div class="card-header">📊 User-Selectable Results</div>')
                
                transcription_output = gr.Textbox(
                    label="📝 Original Transcription (USER-SELECTED Enhanced)",
                    placeholder="Your USER-SELECTABLE transcription will appear here...",
                    lines=12,
                    max_lines=18,
                    interactive=False,
                    show_copy_button=True
                )
                
                copy_original_btn = gr.Button("📋 Copy Original Transcription", size="sm")
                
                gr.HTML('</div>')
                
                # Translation Section
                gr.HTML("""
                <div class="translation-section">
                    <div style="color: #3b82f6; font-size: 1.5rem; font-weight: 800; margin-bottom: 25px; margin-top: 18px;">🌐 Optional English Translation</div>
                    <p style="color: #cbd5e1; margin-bottom: 25px; font-size: 1.2rem;">
                        Click the button below to translate your transcription to English using smart text chunking.
                    </p>
                </div>
                """)
                
                with gr.Row():
                    translate_btn = gr.Button(
                        "🌐 TRANSLATE TO ENGLISH (SMART CHUNKING)",
                        variant="secondary",
                        elem_classes="translation-button",
                        size="lg"
                    )
                
                english_translation_output = gr.Textbox(
                    label="🌐 English Translation (Optional)",
                    placeholder="Click the translate button above to generate English translation...",
                    lines=10,
                    max_lines=18,
                    interactive=False,
                    show_copy_button=True
                )
                
                copy_translation_btn = gr.Button("🌐 Copy English Translation", size="sm")
        
        # Audio Comparison
        with gr.Row():
            with gr.Column():
                gr.HTML('<div class="user-selectable-card"><div class="card-header">📥 Original Audio</div>')
                original_audio_player = gr.Audio(
                    label="Original Audio",
                    interactive=False
                )
                gr.HTML('</div>')
            
            with gr.Column():
                gr.HTML('<div class="user-selectable-card"><div class="card-header">🚀 USER-SELECTABLE Enhanced Audio</div>')
                enhanced_audio_player = gr.Audio(
                    label="USER-SELECTABLE Enhanced Audio (Custom Checkbox Pipeline)",
                    interactive=False
                )
                gr.HTML('</div>')
        
        # Reports
        with gr.Row():
            with gr.Column():
                with gr.Accordion("🚀 USER-SELECTABLE Enhancement Report", open=False):
                    enhancement_report = gr.Textbox(
                        label="USER-SELECTABLE Enhancement Report",
                        lines=20,
                        show_copy_button=True,
                        interactive=False
                    )
            
            with gr.Column():
                with gr.Accordion("📋 USER-SELECTABLE Processing Report", open=False):
                    processing_report = gr.Textbox(
                        label="USER-SELECTABLE Processing Report", 
                        lines=20,
                        show_copy_button=True,
                        interactive=False
                    )
        
        # System Monitoring
        gr.HTML('<div class="user-selectable-card"><div class="card-header">🚀 USER-SELECTABLE System Monitoring</div>')
        
        log_display = gr.Textbox(
            label="",
            value="🚀 USER-SELECTABLE system ready - checkbox controls active...",
            interactive=False,
            lines=14,
            max_lines=20,
            elem_classes="log-user-selectable",
            show_label=False
        )
        
        with gr.Row():
            refresh_logs_btn = gr.Button("🔄 Refresh USER-SELECTABLE Logs", size="sm")
            clear_logs_btn = gr.Button("🗑️ Clear Logs", size="sm")
        
        gr.HTML('</div>')
        
        # Event Handlers
        transcribe_btn.click(
            fn=transcribe_audio_user_selectable,
            inputs=[audio_input, language_dropdown, enhancement_radio, 
                   spectral_methods, frequency_methods, time_frequency_methods, 
                   preprocessing_methods, advanced_methods],
            outputs=[transcription_output, original_audio_player, enhanced_audio_player, enhancement_report, processing_report],
            show_progress=True
        )
        
        translate_btn.click(
            fn=translate_transcription_user_selectable,
            inputs=[transcription_output],
            outputs=[english_translation_output],
            show_progress=True
        )
        
        copy_original_btn.click(
            fn=lambda text: text,
            inputs=[transcription_output],
            outputs=[],
            js="(text) => { navigator.clipboard.writeText(text); return text; }"
        )
        
        copy_translation_btn.click(
            fn=lambda text: text,
            inputs=[english_translation_output],
            outputs=[],
            js="(text) => { navigator.clipboard.writeText(text); return text; }"
        )
        
        refresh_logs_btn.click(
            fn=get_current_logs,
            inputs=[],
            outputs=[log_display]
        )
        
        def clear_user_selectable_logs():
            global log_capture
            if log_capture:
                with log_capture.lock:
                    log_capture.log_buffer.clear()
            return "🚀 USER-SELECTABLE logs cleared - system ready"
        
        clear_logs_btn.click(
            fn=clear_user_selectable_logs,
            inputs=[],
            outputs=[log_display]
        )
        
        def auto_refresh_user_selectable_logs():
            return get_current_logs()
        
        timer = gr.Timer(value=3, active=True)
        timer.tick(
            fn=auto_refresh_user_selectable_logs,
            inputs=[],
            outputs=[log_display]
        )
        
        interface.load(
            fn=initialize_user_selectable_transcriber,
            inputs=[],
            outputs=[status_display]
        )
    
    return interface

def main():
    """Launch the complete USER-SELECTABLE speech enhancement transcription system"""
    
    if "/path/to/your/" in MODEL_PATH:
        print("="*80)
        print("🚀 USER-SELECTABLE SPEECH ENHANCEMENT SYSTEM CONFIGURATION REQUIRED")
        print("="*80)
        print("Please update the MODEL_PATH variable with your local Gemma 3N model directory")
        print("Download from: https://huggingface.co/google/gemma-3n-e4b-it")
        print("="*80)
        return
    
    setup_user_selectable_logging()
    
    print("🚀 Launching USER-SELECTABLE SPEECH ENHANCEMENT & TRANSCRIPTION SYSTEM...")
    print("="*80)
    print("☑️ USER-SELECTABLE CHECKBOX FEATURES - ALL METHODS AVAILABLE:")
    print("="*80)
    print("🔬 SPECTRAL DOMAIN METHODS (6 checkbox options):")
    print("   ☑️ Spectral Subtraction")
    print("   ☑️ Multi-Band Spectral Subtraction (MBSS)")
    print("   ☑️ Wiener Filtering with optimal parameters")
    print("   ☑️ MMSE Short-Time Spectral Amplitude (MMSE-STSA) Estimator")
    print("   ☑️ MMSE Log-Spectral Amplitude (MMSE-LSA) Estimator")
    print("   ☑️ Optimally-Modified Log-Spectral Amplitude (OM-LSA) Estimator")
    print("="*80)
    print("🎵 FREQUENCY DOMAIN METHODS (2 checkbox options):")
    print("   ☑️ Comprehensive Frequency Filtering (FIXED 85Hz-7900Hz)")
    print("   ☑️ Adaptive Filtering with LMS algorithm")
    print("="*80)
    print("🔬 TIME-FREQUENCY DOMAIN METHODS (3 checkbox options):")
    print("   ☑️ Differentiable Adaptive Short-Time Fourier Transform (DA-STFT)")
    print("   ☑️ Time-Frequency Masking for noise isolation")
    print("   ☑️ Frame-Based Processing with overlap-add")
    print("="*80)
    print("📊 PREPROCESSING & NORMALIZATION (5 checkbox options):")
    print("   ☑️ Z-score Min-Max Normalization for enhanced feature extraction")
    print("   ☑️ Dynamic Range Compression with attack/release times")
    print("   ☑️ Noise Gating with adaptive thresholds")
    print("   ☑️ Temporal Smoothing to reduce transient noise")
    print("   ☑️ Frame Averaging to improve Signal-to-Noise Ratio")
    print("="*80)
    print("🔬 ADVANCED METHODS (4 checkbox options):")
    print("   ☑️ Signal Subspace Approach (SSA) with SVD decomposition")
    print("   ☑️ Noise Profile Analysis with targeted reduction")
    print("   ☑️ Signal-to-Noise Ratio (SNR) Enhancement")
    print("   ☑️ Advanced VAD Enhancement with multi-feature detection")
    print("="*80)
    print("🚀 USER-SELECTABLE FEATURES:")
    print("   👤 Checkbox Control: Users select which methods to apply")
    print("   🔧 Custom Pipeline: Only selected methods are executed")
    print("   📊 Method Reports: Detailed breakdown of applied techniques")
    print("   ⚡ Efficiency: Skip unused methods for faster processing")
    print("   🎯 Flexibility: Mix and match techniques for optimal results")
    print("="*80)
    print("⏱️ TIMEOUT PROTECTION:")
    print(f"   ⏱️ {CHUNK_TIMEOUT}-second timeout per chunk")
    print("   ⏱️ User-selected quality detection and assessment")
    print("   ⏱️ 'Input Audio Very noisy. Unable to extract details.' messages")
    print("   ⏱️ Graceful degradation for problematic audio")
    print("="*80)
    print("🌐 OPTIONAL TRANSLATION FEATURES:")
    print("   👤 User Control: Translation only when user clicks button")
    print("   📝 Smart Chunking: Preserves meaning with sentence overlap")
    print(f"   📏 Chunk Size: {MAX_TRANSLATION_CHUNK_SIZE} characters with {SENTENCE_OVERLAP} sentence overlap")
    print("   🔗 Context Preservation: Intelligent sentence boundary detection")
    print("   🛡️ Error Recovery: Graceful handling of failed chunks")
    print("="*80)
    print("🌍 LANGUAGE SUPPORT: 150+ languages including:")
    print("   • Burmese, Pashto, Persian, Dzongkha, Tibetan")
    print("   • All major world languages and regional variants")
    print("   • Smart English detection to skip unnecessary translation")
    print("="*80)
    print("🚀 USER-SELECTABLE ADVANTAGES:")
    print("   ☑️ Complete Control: Users decide which methods to use")
    print("   ⚡ Optimized Performance: Only selected methods consume resources")
    print("   🎯 Targeted Processing: Focus on specific enhancement needs")
    print("   📊 Transparent Reports: See exactly which methods were applied")
    print("   🔧 Flexible Combinations: Create custom processing pipelines")
    print("   💡 Educational: Learn which methods work best for your audio")
    print("="*80)
    
    try:
        interface = create_user_selectable_interface()
        
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=False,
            show_error=True,
            quiet=False,
            favicon_path=None,
            auth=None,
            inbrowser=True,
            prevent_thread_lock=False
        )
        
    except Exception as e:
        print(f"❌ USER-SELECTABLE system launch failed: {e}")
        print("🔧 USER-SELECTABLE system troubleshooting:")
        print("   • Verify model path is correct and accessible")
        print("   • Check GPU memory availability and drivers")
        print("   • Ensure all dependencies are installed:")
        print("     pip install --upgrade torch transformers gradio librosa soundfile")
        print("     pip install --upgrade noisereduce scipy nltk scikit-learn")
        print("   • Verify Python environment and version compatibility")
        print("   • Check port 7860 availability")
        print("   • ALL preprocessing methods are available as checkbox controls")
        print("   • Users can select any combination of the 20 available methods")
        print("   • Custom processing pipelines based on user selections")
        print("   • FIXED hann window function (no longer hanning)")
        print("   • Comprehensive fallback systems are active")
        print("="*80)

if __name__ == "__main__":
    main()

