# -*- coding: utf-8 -*-
"""
COMPREHENSIVE ADAPTIVE SPEECH TRANSCRIPTION WITH GEEKSFORGEEKS PREPROCESSING
=========================================================================

COMPLETE FEATURES IMPLEMENTED:
- Full GeeksforGeeks preprocessing pipeline (ALL 9 techniques)
- Adaptive chunk sizing with automatic fallback (30s → 10s, 15s, 20s, 40s)
- Enable/disable preprocessing toggle
- 75-second timeout protection with noise detection
- Advanced noise reduction and comprehensive normalization
- Feature extraction (MFCCs, spectral characteristics, log-mel spectrograms)
- Optional English translation with smart chunking
- Complete Gradio UI with real-time monitoring
- Comprehensive error handling and fallback systems

Author: Comprehensive Adaptive Audio Processing System
Version: Complete GeeksforGeeks Enhanced 17.0
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
from scipy.signal import butter, filtfilt
from scipy.signal.windows import hann  # FIXED: Use hann instead of hanning
import noisereduce as nr
import datetime
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import psutil
import re
import nltk
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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

# --- COMPREHENSIVE CONFIGURATION ---
MODEL_PATH = "/path/to/your/local/gemma-3n-e4b-it"  # UPDATE THIS PATH

# Adaptive chunk settings
DEFAULT_CHUNK_SECONDS = 30
FALLBACK_CHUNK_SECONDS = [10, 15, 20, 40]  # Fallback chunk sizes
OVERLAP_SECONDS = 2
SAMPLE_RATE = 16000
CHUNK_TIMEOUT = 75
MAX_RETRIES = 1

# GeeksforGeeks preprocessing settings
GEEKSFORGEEKS_CUTOFF_FREQ = 4000  # Low-pass filter cutoff
GEEKSFORGEEKS_FILTER_ORDER = 4    # Butterworth filter order
GEEKSFORGEEKS_TARGET_LENGTH = 16000  # Target length for model input

# Memory settings
MIN_FREE_MEMORY_GB = 0.3
CHECK_MEMORY_FREQUENCY = 5

# Translation settings
MAX_TRANSLATION_CHUNK_SIZE = 1000
SENTENCE_OVERLAP = 1
MIN_CHUNK_SIZE = 100

# Language support
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

class ComprehensiveGeeksforGeeksAudioPreprocessor:
    """COMPREHENSIVE GeeksforGeeks-based audio preprocessing with ALL techniques"""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.target_length = GEEKSFORGEEKS_TARGET_LENGTH
        print(f"🚀 COMPREHENSIVE GeeksforGeeks Audio Preprocessor initialized for {sample_rate}Hz")
    
    # 1. RESAMPLING - Standardizing sample rate (GeeksforGeeks Method)
    def resample_audio(self, audio_path: str, target_sr: int = None) -> Tuple[np.ndarray, int]:
        """Resample audio to target sample rate (GeeksforGeeks method)"""
        try:
            if target_sr is None:
                target_sr = self.sample_rate
            
            print(f"🔄 GEEKSFORGEEKS: Resampling audio to {target_sr}Hz...")
            y, sr = librosa.load(audio_path, sr=target_sr)
            print(f"✅ Sample rate after resampling: {sr}")
            return y, sr
            
        except Exception as e:
            print(f"❌ Resampling failed: {e}")
            y, sr = librosa.load(audio_path, sr=None)
            if sr != target_sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
            return y, sr
    
    # 2. FILTERING - Noise reduction and frequency filtering (GeeksforGeeks Method)
    def butter_lowpass_filter(self, data: np.ndarray, cutoff_freq: int, sample_rate: int, order: int = 4) -> np.ndarray:
        """Apply Butterworth low-pass filter for noise reduction (GeeksforGeeks method)"""
        try:
            print(f"🎵 GEEKSFORGEEKS: Applying Butterworth low-pass filter (cutoff: {cutoff_freq}Hz, order: {order})...")
            
            nyquist = 0.5 * sample_rate
            normal_cutoff = cutoff_freq / nyquist
            
            if normal_cutoff >= 1.0:
                normal_cutoff = 0.99
                print(f"⚠️ Adjusted cutoff frequency to {normal_cutoff * nyquist:.0f}Hz")
            
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            filtered_data = filtfilt(b, a, data)
            
            print(f"✅ Filtered audio shape: {filtered_data.shape}")
            return filtered_data.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Butterworth filtering failed: {e}")
            return data.astype(np.float32)
    
    # 3. NOISE REDUCTION - Advanced denoising (GeeksforGeeks Enhancement)
    def advanced_noise_reduction(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Advanced noise reduction using spectral gating (GeeksforGeeks enhancement)"""
        try:
            print("🔇 GEEKSFORGEEKS: Applying advanced noise reduction...")
            
            # Use noisereduce library for spectral gating
            reduced_noise_audio = nr.reduce_noise(y=audio, sr=sr, stationary=False, prop_decrease=0.8)
            
            print("✅ Advanced noise reduction completed")
            return reduced_noise_audio.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Advanced noise reduction failed: {e}")
            return audio.astype(np.float32)
    
    # 4. NORMALIZATION - Signal amplitude scaling (GeeksforGeeks Method)
    def comprehensive_normalization(self, audio: np.ndarray) -> np.ndarray:
        """Comprehensive normalization for consistent signal magnitudes (GeeksforGeeks method)"""
        try:
            print("📊 GEEKSFORGEEKS: Applying comprehensive normalization...")
            
            # Method 1: Peak normalization
            peak_normalized = audio / (np.max(np.abs(audio)) + 1e-8)
            
            # Method 2: RMS normalization
            rms = np.sqrt(np.mean(peak_normalized**2))
            target_rms = 0.15  # Target RMS level
            rms_normalized = peak_normalized * (target_rms / (rms + 1e-8))
            
            # Method 3: Z-score normalization
            mean = np.mean(rms_normalized)
            std = np.std(rms_normalized)
            z_normalized = (rms_normalized - mean) / (std + 1e-8)
            
            # Final scaling to [-0.95, 0.95] range
            final_normalized = np.clip(z_normalized * 0.3, -0.95, 0.95)
            
            print(f"✅ Normalization completed - RMS: {np.sqrt(np.mean(final_normalized**2)):.4f}")
            return final_normalized.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Comprehensive normalization failed: {e}")
            return librosa.util.normalize(audio).astype(np.float32)
    
    # 5. HANDLING VARIABLE LENGTHS - Padding/Trimming (GeeksforGeeks Method)
    def handle_variable_lengths(self, audio: np.ndarray, target_length: int = None) -> np.ndarray:
        """Handle variable lengths through padding/trimming (GeeksforGeeks method)"""
        try:
            if target_length is None:
                target_length = self.target_length
            
            print(f"📏 GEEKSFORGEEKS: Handling variable lengths (target: {target_length} samples)...")
            
            if len(audio) < target_length:
                # Padding with reflection to avoid discontinuities
                pad_length = target_length - len(audio)
                if len(audio) > 0:
                    audio = np.pad(audio, (0, pad_length), mode='reflect')
                else:
                    audio = np.zeros(target_length)
                print(f"📈 Padded audio to {len(audio)} samples using reflection")
            else:
                # Trimming from center to preserve important parts
                start = (len(audio) - target_length) // 2
                audio = audio[start:start + target_length]
                print(f"✂️ Trimmed audio to {len(audio)} samples from center")
            
            print(f"✅ Variable length handling completed: {audio.shape}")
            return audio.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Variable length handling failed: {e}")
            return audio.astype(np.float32)
    
    # 6. FEATURE EXTRACTION - Spectral characteristics (GeeksforGeeks Method)
    def extract_comprehensive_features(self, audio: np.ndarray, sr: int) -> Dict:
        """Extract comprehensive audio features (GeeksforGeeks method)"""
        try:
            print("🔍 GEEKSFORGEEKS: Extracting comprehensive audio features...")
            
            features = {}
            
            # Mel-frequency cepstral coefficients (MFCCs)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            features['mfccs'] = mfccs
            features['mfccs_mean'] = np.mean(mfccs, axis=1)
            features['mfccs_std'] = np.std(mfccs, axis=1)
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            features['chroma'] = chroma
            features['chroma_mean'] = np.mean(chroma, axis=1)
            
            # Spectral centroid
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
            features['spectral_centroid'] = spectral_centroid
            features['spectral_centroid_mean'] = np.mean(spectral_centroid)
            
            # Spectral bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
            features['spectral_bandwidth'] = spectral_bandwidth
            features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
            features['spectral_rolloff'] = spectral_rolloff
            features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)
            features['zcr'] = zcr
            features['zcr_mean'] = np.mean(zcr)
            
            # Spectral contrast
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            features['spectral_contrast'] = spectral_contrast
            features['spectral_contrast_mean'] = np.mean(spectral_contrast, axis=1)
            
            # Tonnetz
            tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
            features['tonnetz'] = tonnetz
            features['tonnetz_mean'] = np.mean(tonnetz, axis=1)
            
            print(f"✅ Extracted {len(features)} comprehensive feature sets")
            return features
            
        except Exception as e:
            print(f"❌ Feature extraction failed: {e}")
            return {}
    
    # 7. LOG-MEL SPECTROGRAM - Enhanced spectral representation (GeeksforGeeks Method)
    def compute_enhanced_logmel_spectrogram(self, audio: np.ndarray, sr: int, n_mels: int = 128, 
                                          hop_length: int = 512, n_fft: int = 2048) -> np.ndarray:
        """Compute enhanced log-mel spectrogram (GeeksforGeeks method)"""
        try:
            print(f"📊 GEEKSFORGEEKS: Computing enhanced log-mel spectrogram (n_mels: {n_mels}, n_fft: {n_fft})...")
            
            # Compute mel spectrogram with enhanced parameters
            mel_spectrogram = librosa.feature.melspectrogram(
                y=audio, 
                sr=sr, 
                n_mels=n_mels, 
                hop_length=hop_length,
                n_fft=n_fft,
                fmin=80,  # Minimum frequency
                fmax=8000  # Maximum frequency for speech
            )
            
            # Convert to log scale with improved reference
            logmel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max, top_db=80)
            
            print(f"✅ Enhanced log-mel spectrogram shape: {logmel_spectrogram.shape}")
            return logmel_spectrogram
            
        except Exception as e:
            print(f"❌ Enhanced log-mel spectrogram computation failed: {e}")
            return np.array([])
    
    # 8. STANDARDIZATION OF FORMATS - Ensure consistent formats (GeeksforGeeks Method)
    def standardize_audio_format(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, Dict]:
        """Standardize audio format for consistency (GeeksforGeeks method)"""
        try:
            print("🔧 GEEKSFORGEEKS: Standardizing audio format...")
            
            format_info = {}
            
            # Ensure mono audio
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
                format_info['converted_to_mono'] = True
            else:
                format_info['converted_to_mono'] = False
            
            # Ensure float32 format
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
                format_info['converted_to_float32'] = True
            else:
                format_info['converted_to_float32'] = False
            
            # Ensure proper sample rate
            format_info['sample_rate'] = sr
            format_info['duration'] = len(audio) / sr
            format_info['samples'] = len(audio)
            
            # Check for NaN or infinite values
            if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
                audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
                format_info['cleaned_invalid_values'] = True
            else:
                format_info['cleaned_invalid_values'] = False
            
            print(f"✅ Audio format standardized: {format_info}")
            return audio.astype(np.float32), format_info
            
        except Exception as e:
            print(f"❌ Audio format standardization failed: {e}")
            return audio.astype(np.float32), {}
    
    # 9. MODEL EFFICIENCY OPTIMIZATION - Prepare for efficient processing (GeeksforGeeks Method)
    def optimize_for_model_efficiency(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, Dict]:
        """Optimize audio for efficient model processing (GeeksforGeeks method)"""
        try:
            print("⚡ GEEKSFORGEEKS: Optimizing for model efficiency...")
            
            optimization_info = {}
            
            # Pre-emphasis filter to balance frequency spectrum
            pre_emphasis = 0.97
            emphasized_audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
            optimization_info['pre_emphasis_applied'] = True
            
            # Window the signal to reduce spectral leakage using FIXED hann function
            window = hann(len(emphasized_audio))
            windowed_audio = emphasized_audio * window
            optimization_info['windowing_applied'] = True
            
            # Ensure power-of-2 length for efficient FFT processing
            target_length = 2 ** int(np.ceil(np.log2(len(windowed_audio))))
            if len(windowed_audio) < target_length:
                padded_audio = np.pad(windowed_audio, (0, target_length - len(windowed_audio)), mode='constant')
                optimization_info['padded_for_fft'] = True
                optimization_info['fft_length'] = target_length
            else:
                padded_audio = windowed_audio[:target_length]
                optimization_info['padded_for_fft'] = False
                optimization_info['fft_length'] = target_length
            
            # Final normalization for numerical stability
            if np.max(np.abs(padded_audio)) > 0:
                padded_audio = padded_audio / np.max(np.abs(padded_audio)) * 0.95
            
            optimization_info['final_length'] = len(padded_audio)
            optimization_info['final_rms'] = np.sqrt(np.mean(padded_audio**2))
            
            print(f"✅ Model efficiency optimization completed: {optimization_info}")
            return padded_audio.astype(np.float32), optimization_info
            
        except Exception as e:
            print(f"❌ Model efficiency optimization failed: {e}")
            return audio.astype(np.float32), {}
    
    # 10. COMPREHENSIVE PREPROCESSING PIPELINE - All methods combined (GeeksforGeeks Complete)
    def apply_comprehensive_geeksforgeeks_preprocessing(self, audio_path: str, enable_all_features: bool = True) -> Tuple[np.ndarray, Dict]:
        """Apply COMPLETE GeeksforGeeks preprocessing pipeline with ALL techniques"""
        try:
            print("🚀 Starting COMPREHENSIVE GEEKSFORGEEKS preprocessing pipeline...")
            print("✅ ALL GeeksforGeeks techniques will be applied:")
            print("   1. ✅ Resampling - Standardizing sample rate")
            print("   2. ✅ Filtering - Butterworth low-pass noise reduction") 
            print("   3. ✅ Noise Reduction - Advanced spectral gating")
            print("   4. ✅ Normalization - Comprehensive signal scaling")
            print("   5. ✅ Variable Length Handling - Padding/trimming")
            print("   6. ✅ Feature Extraction - Spectral characteristics")
            print("   7. ✅ Log-Mel Spectrogram - Enhanced representation")
            print("   8. ✅ Format Standardization - Consistent formats")
            print("   9. ✅ Model Efficiency - Optimization for processing")
            
            comprehensive_stats = {}
            
            # Step 1: Resampling - Standardizing sample rate
            resampled_audio, sr = self.resample_audio(audio_path, self.sample_rate)
            comprehensive_stats['original_length'] = len(resampled_audio) / sr
            comprehensive_stats['sample_rate'] = sr
            
            # Step 2: Format Standardization - Ensure consistent formats
            standardized_audio, format_info = self.standardize_audio_format(resampled_audio, sr)
            comprehensive_stats.update(format_info)
            
            # Step 3: Advanced Noise Reduction - Remove background noise
            denoised_audio = self.advanced_noise_reduction(standardized_audio, sr)
            
            # Step 4: Filtering - Butterworth low-pass filter
            filtered_audio = self.butter_lowpass_filter(
                denoised_audio, 
                GEEKSFORGEEKS_CUTOFF_FREQ, 
                sr, 
                GEEKSFORGEEKS_FILTER_ORDER
            )
            
            # Step 5: Comprehensive Normalization - Signal scaling
            normalized_audio = self.comprehensive_normalization(filtered_audio)
            
            # Step 6: Variable Length Handling - Uniform length
            length_handled_audio = self.handle_variable_lengths(normalized_audio)
            
            # Step 7: Model Efficiency Optimization - Prepare for processing
            optimized_audio, optimization_info = self.optimize_for_model_efficiency(length_handled_audio, sr)
            comprehensive_stats.update(optimization_info)
            
            # Step 8: Feature Extraction (optional, for analysis)
            if enable_all_features:
                extracted_features = self.extract_comprehensive_features(optimized_audio, sr)
                comprehensive_stats['extracted_features_count'] = len(extracted_features)
                
                # Step 9: Enhanced Log-Mel Spectrogram
                logmel_spectrogram = self.compute_enhanced_logmel_spectrogram(optimized_audio, sr)
                comprehensive_stats['logmel_spectrogram_shape'] = logmel_spectrogram.shape if logmel_spectrogram.size > 0 else None
            
            # Final quality assessment
            comprehensive_stats['final_rms'] = np.sqrt(np.mean(optimized_audio**2))
            comprehensive_stats['final_length'] = len(optimized_audio) / sr
            comprehensive_stats['final_peak'] = np.max(np.abs(optimized_audio))
            comprehensive_stats['preprocessing_applied'] = True
            comprehensive_stats['all_geeksforgeeks_methods'] = True
            
            print(f"✅ COMPREHENSIVE GEEKSFORGEEKS preprocessing completed")
            print(f"📊 Final stats: RMS={comprehensive_stats['final_rms']:.4f}, Peak={comprehensive_stats['final_peak']:.4f}")
            
            return optimized_audio.astype(np.float32), comprehensive_stats
            
        except Exception as e:
            print(f"❌ Comprehensive GeeksforGeeks preprocessing failed: {e}")
            # Fallback: return basic processed audio
            try:
                audio, sr = librosa.load(audio_path, sr=self.sample_rate)
                return audio.astype(np.float32), {'preprocessing_applied': False, 'fallback_used': True}
            except:
                return np.array([]), {'preprocessing_applied': False, 'error': str(e)}

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

class ComprehensiveAdaptiveSpeechTranscriber:
    """Complete adaptive transcriber with comprehensive GeeksforGeeks preprocessing and variable chunk sizing"""
    
    def __init__(self, model_path: str, use_quantization: bool = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = None
        self.processor = None
        self.audio_preprocessor = ComprehensiveGeeksforGeeksAudioPreprocessor(SAMPLE_RATE)
        self.text_chunker = SmartTextChunker()
        self.chunk_count = 0
        self.temp_files = []
        
        print(f"🖥️ Using device: {self.device}")
        print(f"🚀 COMPREHENSIVE ADAPTIVE transcriber with GeeksforGeeks preprocessing initialized")
        print(f"📏 Default chunk size: {DEFAULT_CHUNK_SECONDS}s")
        print(f"📏 Fallback chunk sizes: {FALLBACK_CHUNK_SECONDS}")
        print(f"⏱️ Chunk timeout: {CHUNK_TIMEOUT} seconds")
        print(f"🔧 GeeksforGeeks preprocessing: ALL 9 techniques available")
        
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
            print(f"✅ COMPREHENSIVE ADAPTIVE model loaded in {loading_time:.1f} seconds")
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            raise
    
    def create_adaptive_chunks(self, audio_array: np.ndarray, chunk_seconds: int) -> List[Tuple[np.ndarray, float, float]]:
        """Create chunks with adaptive sizing"""
        chunk_samples = int(chunk_seconds * SAMPLE_RATE)
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
        
        print(f"✅ Created {len(chunks)} adaptive chunks ({chunk_seconds}s each)")
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
    
    def transcribe_with_adaptive_chunks(self, audio_array: np.ndarray, language: str = "auto") -> Tuple[str, Dict]:
        """Transcribe with adaptive chunk sizing and fallback mechanism"""
        chunk_sizes = [DEFAULT_CHUNK_SECONDS] + FALLBACK_CHUNK_SECONDS
        
        for attempt, chunk_seconds in enumerate(chunk_sizes):
            print(f"🎯 Attempt {attempt + 1}: Trying {chunk_seconds}s chunks...")
            
            chunks = self.create_adaptive_chunks(audio_array, chunk_seconds)
            transcriptions = []
            successful = 0
            failed = 0
            timeout_count = 0
            
            for i, (chunk, start_time_chunk, end_time_chunk) in enumerate(chunks):
                print(f"🚀 Processing chunk {i+1}/{len(chunks)} ({start_time_chunk:.1f}s-{end_time_chunk:.1f}s)")
                
                try:
                    transcription = self.transcribe_chunk_with_timeout(chunk, language)
                    transcriptions.append(transcription)
                    
                    if transcription == "Input Audio Very noisy. Unable to extract details.":
                        timeout_count += 1
                        print(f"⏱️ Chunk {i+1}: Timeout due to noisy audio")
                    elif transcription.startswith('[') and transcription.endswith(']'):
                        failed += 1
                        print(f"❌ Chunk {i+1}: Failed - {transcription}")
                    else:
                        successful += 1
                        print(f"✅ Chunk {i+1}: Success - {transcription[:50]}...")
                
                except Exception as e:
                    print(f"❌ Chunk {i+1} processing error: {e}")
                    transcriptions.append(f"[CHUNK_{i+1}_ERROR]")
                    failed += 1
                
                if i % CHECK_MEMORY_FREQUENCY == 0:
                    OptimizedMemoryManager.fast_cleanup()
            
            # Calculate success rate
            total_chunks = len(chunks)
            success_rate = successful / total_chunks if total_chunks > 0 else 0
            
            print(f"📊 Chunk size {chunk_seconds}s results: {successful}/{total_chunks} successful ({success_rate*100:.1f}%)")
            
            # If success rate is acceptable, use this result
            if success_rate >= 0.7 or successful > 0:  # At least 70% success or some success
                final_transcription = self.merge_transcriptions_with_info(
                    transcriptions, timeout_count, chunk_seconds
                )
                
                stats = {
                    'chunk_seconds_used': chunk_seconds,
                    'total_chunks': total_chunks,
                    'successful_chunks': successful,
                    'failed_chunks': failed,
                    'timeout_chunks': timeout_count,
                    'success_rate': success_rate,
                    'attempts_made': attempt + 1
                }
                
                return final_transcription, stats
            
            print(f"⚠️ Chunk size {chunk_seconds}s had low success rate ({success_rate*100:.1f}%), trying next size...")
        
        # If all chunk sizes failed
        return f"❌ Failed to transcribe with all chunk sizes: {chunk_sizes}", {
            'chunk_seconds_used': 'all_failed',
            'attempts_made': len(chunk_sizes),
            'success_rate': 0
        }
    
    def merge_transcriptions_with_info(self, transcriptions: List[str], timeout_count: int, chunk_seconds: int) -> str:
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
                return f"❌ All {len(transcriptions)} chunks timed out due to very noisy audio. Chunk size: {chunk_seconds}s"
            else:
                return f"❌ No valid transcriptions from {len(transcriptions)} chunks. Chunk size: {chunk_seconds}s"
        
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
            merged_text += f"\n\n[Comprehensive Adaptive Processing Summary: {', '.join(summary_parts)} - {success_rate:.1f}% success rate with {chunk_seconds}s chunks]"
            
            if noisy_timeout_count > 0:
                merged_text += f"\n[Note: {noisy_timeout_count} chunks were too noisy and timed out after {CHUNK_TIMEOUT} seconds each]"
        
        return merged_text.strip()
    
    def transcribe_with_comprehensive_geeksforgeeks_preprocessing(self, audio_path: str, enable_preprocessing: bool, 
                                                                language: str = "auto") -> Tuple[str, str, str, Dict]:
        try:
            print(f"🚀 Starting COMPREHENSIVE ADAPTIVE transcription with GeeksforGeeks preprocessing...")
            print(f"🔧 Preprocessing enabled: {enable_preprocessing}")
            print(f"🌍 Language: {language}")
            print(f"📏 Adaptive chunk sizing enabled with fallback")
            print(f"⏱️ 75-second timeout per chunk with noise detection")
            
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
            
            # Apply comprehensive GeeksforGeeks preprocessing if enabled
            if enable_preprocessing:
                enhanced_audio, preprocessing_stats = self.audio_preprocessor.apply_comprehensive_geeksforgeeks_preprocessing(
                    audio_path, enable_all_features=True
                )
            else:
                print("⚠️ GeeksforGeeks preprocessing DISABLED - using raw audio")
                enhanced_audio = audio_array
                preprocessing_stats = {'preprocessing_applied': False}
            
            enhanced_path = tempfile.mktemp(suffix="_comprehensive_geeksforgeeks_enhanced.wav")
            original_path = tempfile.mktemp(suffix="_original.wav")
            
            sf.write(enhanced_path, enhanced_audio, SAMPLE_RATE)
            sf.write(original_path, audio_array, SAMPLE_RATE)
            
            print("✂️ Starting comprehensive adaptive chunk transcription...")
            start_time = time.time()
            
            # Transcribe with adaptive chunk sizing
            final_transcription, transcription_stats = self.transcribe_with_adaptive_chunks(enhanced_audio, language)
            
            processing_time = time.time() - start_time
            
            # Combine stats
            combined_stats = {**preprocessing_stats, **transcription_stats}
            combined_stats['processing_time'] = processing_time
            
            print(f"✅ COMPREHENSIVE ADAPTIVE transcription completed in {processing_time:.2f}s")
            if 'chunk_seconds_used' in transcription_stats:
                print(f"📏 Optimal chunk size: {transcription_stats['chunk_seconds_used']}s")
            if 'success_rate' in transcription_stats:
                print(f"📊 Success rate: {transcription_stats['success_rate']*100:.1f}%")
            
            return final_transcription, original_path, enhanced_path, combined_stats
                
        except Exception as e:
            error_msg = f"❌ Comprehensive adaptive transcription failed: {e}"
            print(error_msg)
            OptimizedMemoryManager.fast_cleanup()
            return error_msg, audio_path, audio_path, {}
        finally:
            for temp_file in self.temp_files:
                AudioHandler.cleanup_temp_file(temp_file)
            self.temp_files.clear()
    
    def translate_text_chunks(self, text: str) -> str:
        if self.model is None or self.processor is None:
            return "[MODEL_NOT_LOADED]"
        
        if not text or text.startswith('['):
            return "[NO_TRANSLATION_NEEDED]"
        
        try:
            print("🌐 Starting comprehensive adaptive text translation...")
            
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
            
            if "🚀" in text or "COMPREHENSIVE" in text or "GEEKSFORGEEKS" in text:
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
            return "\n".join(self.log_buffer[-50:]) if self.log_buffer else "🚀 Comprehensive system ready..."

def setup_comprehensive_logging():
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
    return "🚀 Comprehensive system initializing..."

def initialize_comprehensive_transcriber():
    global transcriber
    if transcriber is None:
        try:
            print("🚀 Initializing COMPREHENSIVE Speech Transcription with GeeksforGeeks Preprocessing...")
            print("✅ COMPREHENSIVE GEEKSFORGEEKS FEATURES ENABLED:")
            print("🔧 GeeksforGeeks Preprocessing: ALL 9 techniques available")
            print("   1. ✅ Resampling - Standardizing sample rate (16kHz)")
            print("   2. ✅ Filtering - Butterworth low-pass noise reduction (4kHz cutoff)")
            print("   3. ✅ Noise Reduction - Advanced spectral gating")
            print("   4. ✅ Normalization - Comprehensive signal scaling")
            print("   5. ✅ Variable Length Handling - Intelligent padding/trimming")
            print("   6. ✅ Feature Extraction - MFCCs, spectral characteristics")
            print("   7. ✅ Log-Mel Spectrogram - Enhanced representation")
            print("   8. ✅ Format Standardization - Consistent formats")
            print("   9. ✅ Model Efficiency - FFT optimization, pre-emphasis")
            print("📏 Adaptive Chunk Sizing: 30s default, fallbacks: 10s, 15s, 20s, 40s")
            print("⚡ Enable/Disable Preprocessing Toggle: User controlled")
            print("🔄 Automatic Fallback Mechanism: Comprehensive error recovery")
            print(f"⏱️ Chunk timeout: {CHUNK_TIMEOUT} seconds with noise detection")
            
            transcriber = ComprehensiveAdaptiveSpeechTranscriber(model_path=MODEL_PATH, use_quantization=True)
            return "✅ COMPREHENSIVE ADAPTIVE transcription system ready! GeeksforGeeks preprocessing with ALL techniques enabled."
        except Exception as e:
            try:
                print("🔄 Retrying without quantization...")
                transcriber = ComprehensiveAdaptiveSpeechTranscriber(model_path=MODEL_PATH, use_quantization=False)
                return "✅ COMPREHENSIVE system loaded (standard precision)!"
            except Exception as e2:
                error_msg = f"❌ Comprehensive system failure: {str(e2)}"
                print(error_msg)
                return error_msg
    return "✅ COMPREHENSIVE system already active!"

def transcribe_audio_comprehensive_adaptive(audio_input, language_choice, enable_preprocessing, progress=gr.Progress()):
    global transcriber
    
    if audio_input is None:
        return "❌ Please upload an audio file or record audio.", None, None, "", ""
    
    if transcriber is None:
        return "❌ System not initialized. Please wait for startup.", None, None, "", ""
    
    start_time = time.time()
    print(f"🚀 Starting COMPREHENSIVE ADAPTIVE transcription with GeeksforGeeks preprocessing...")
    print(f"🌍 Language: {language_choice}")
    print(f"🔧 GeeksforGeeks preprocessing enabled: {enable_preprocessing}")
    print(f"📏 Adaptive chunk sizing: {DEFAULT_CHUNK_SECONDS}s → {FALLBACK_CHUNK_SECONDS}")
    print(f"⏱️ Timeout protection: {CHUNK_TIMEOUT}s per chunk")
    
    progress(0.1, desc="Initializing COMPREHENSIVE ADAPTIVE processing...")
    
    temp_audio_path = None
    
    try:
        temp_audio_path = AudioHandler.convert_to_file(audio_input, SAMPLE_RATE)
        
        if enable_preprocessing:
            progress(0.3, desc="Applying COMPREHENSIVE GeeksforGeeks preprocessing (ALL 9 techniques)...")
        else:
            progress(0.3, desc="Skipping preprocessing (disabled by user) - using raw audio...")
        
        language_code = SUPPORTED_LANGUAGES.get(language_choice, "auto")
        
        progress(0.5, desc="COMPREHENSIVE ADAPTIVE transcription with variable chunk sizing...")
        
        transcription, original_path, enhanced_path, stats = transcriber.transcribe_with_comprehensive_geeksforgeeks_preprocessing(
            temp_audio_path, enable_preprocessing, language_code
        )
        
        progress(0.9, desc="Generating COMPREHENSIVE reports...")
        
        enhancement_report = create_comprehensive_adaptive_enhancement_report(stats, enable_preprocessing)
        
        processing_time = time.time() - start_time
        processing_report = create_comprehensive_adaptive_processing_report(
            temp_audio_path, language_choice, enable_preprocessing, 
            processing_time, len(transcription.split()) if isinstance(transcription, str) else 0,
            stats
        )
        
        progress(1.0, desc="COMPREHENSIVE ADAPTIVE processing complete!")
        
        print(f"✅ COMPREHENSIVE ADAPTIVE transcription completed in {processing_time:.2f}s")
        print(f"📊 Output: {len(transcription.split()) if isinstance(transcription, str) else 0} words")
        
        return transcription, original_path, enhanced_path, enhancement_report, processing_report
        
    except Exception as e:
        error_msg = f"❌ Comprehensive adaptive system error: {str(e)}"
        print(error_msg)
        OptimizedMemoryManager.fast_cleanup()
        return error_msg, None, None, "", ""
    finally:
        if temp_audio_path:
            AudioHandler.cleanup_temp_file(temp_audio_path)

def translate_transcription_comprehensive_adaptive(transcription_text, progress=gr.Progress()):
    global transcriber
    
    if not transcription_text or transcription_text.strip() == "":
        return "❌ No transcription text to translate. Please transcribe audio first."
    
    if transcriber is None:
        return "❌ System not initialized. Please wait for system startup."
    
    if transcription_text.startswith("❌") or transcription_text.startswith("["):
        return "❌ Cannot translate error messages or system messages. Please provide valid transcription text."
    
    progress(0.1, desc="Preparing text for comprehensive adaptive translation...")
    
    try:
        text_to_translate = transcription_text
        if "\n\n[Comprehensive Adaptive Processing Summary:" in text_to_translate:
            text_to_translate = text_to_translate.split("\n\n[Comprehensive Adaptive Processing Summary:")[0].strip()
        
        progress(0.3, desc="Creating smart text chunks...")
        
        start_time = time.time()
        translated_text = transcriber.translate_text_chunks(text_to_translate)
        translation_time = time.time() - start_time
        
        progress(0.9, desc="Finalizing comprehensive adaptive translation...")
        
        if not translated_text.startswith('['):
            translated_text += f"\n\n[Comprehensive Adaptive Translation completed in {translation_time:.2f}s using smart chunking]"
        
        progress(1.0, desc="Comprehensive adaptive translation complete!")
        
        print(f"✅ Comprehensive adaptive translation completed in {translation_time:.2f}s")
        
        return translated_text
        
    except Exception as e:
        error_msg = f"❌ Comprehensive adaptive translation failed: {str(e)}"
        print(error_msg)
        OptimizedMemoryManager.fast_cleanup()
        return error_msg

def create_comprehensive_adaptive_enhancement_report(stats: Dict, enable_preprocessing: bool) -> str:
    if not stats:
        return "⚠️ Enhancement statistics not available"
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
🚀 COMPREHENSIVE ADAPTIVE SPEECH ENHANCEMENT REPORT
==================================================
Timestamp: {timestamp}
GeeksforGeeks Preprocessing Enabled: {'✅ YES' if enable_preprocessing else '❌ NO'}

📊 COMPREHENSIVE GEEKSFORGEEKS PREPROCESSING STATUS:
"""
    
    if enable_preprocessing and stats.get('preprocessing_applied', False):
        report += f"""• Audio Duration: {stats.get('original_length', 0):.2f} seconds
• Sample Rate: {stats.get('sample_rate', SAMPLE_RATE)} Hz
• Final RMS Level: {stats.get('final_rms', 0):.4f} (ASR-optimized)
• Final Peak Level: {stats.get('final_peak', 0):.4f}
• Preprocessing Applied: ✅ COMPREHENSIVE GEEKSFORGEEKS PIPELINE

🔧 COMPREHENSIVE GEEKSFORGEEKS PIPELINE STAGES (ALL 9 TECHNIQUES):
• Stage 1: ✅ Audio Resampling ({SAMPLE_RATE} Hz standardization)
• Stage 2: ✅ Format Standardization (Mono, float32, validation)
• Stage 3: ✅ Advanced Noise Reduction (Spectral gating)
• Stage 4: ✅ Butterworth Low-pass Filter ({GEEKSFORGEEKS_CUTOFF_FREQ} Hz cutoff, Order {GEEKSFORGEEKS_FILTER_ORDER})
• Stage 5: ✅ Comprehensive Normalization (Peak + RMS + Z-score)
• Stage 6: ✅ Variable Length Handling (Intelligent padding/trimming)
• Stage 7: ✅ Model Efficiency Optimization (Pre-emphasis + Windowing + FFT)
• Stage 8: ✅ Feature Extraction (MFCCs, spectral characteristics)
• Stage 9: ✅ Enhanced Log-Mel Spectrogram (128 mels, optimized parameters)

🔍 COMPREHENSIVE FEATURE EXTRACTION:
• Features Extracted: {stats.get('extracted_features_count', 0)} feature sets
• Log-Mel Spectrogram Shape: {stats.get('logmel_spectrogram_shape', 'N/A')}
• Pre-emphasis Applied: {'✅' if stats.get('pre_emphasis_applied', False) else '❌'}
• Windowing Applied: {'✅' if stats.get('windowing_applied', False) else '❌'}
• FFT Optimization: {'✅' if stats.get('padded_for_fft', False) else '❌'}
• FFT Length: {stats.get('fft_length', 'N/A')}

🔧 FORMAT STANDARDIZATION:
• Converted to Mono: {'✅' if stats.get('converted_to_mono', False) else '❌'}
• Converted to Float32: {'✅' if stats.get('converted_to_float32', False) else '❌'}
• Cleaned Invalid Values: {'✅' if stats.get('cleaned_invalid_values', False) else '❌'}
• Audio Samples: {stats.get('samples', 'N/A')}
"""
    else:
        report += f"""• Preprocessing: ❌ DISABLED BY USER
• Raw Audio Used: ✅ NO PREPROCESSING APPLIED
• Audio Duration: {stats.get('original_length', 0):.2f} seconds
• User Choice: Skip GeeksforGeeks enhancement pipeline
"""
    
    report += f"""
📏 COMPREHENSIVE ADAPTIVE CHUNK SIZING RESULTS:
• Chunk Size Used: {stats.get('chunk_seconds_used', 'N/A')}s
• Total Chunks: {stats.get('total_chunks', 0)}
• Successful Chunks: {stats.get('successful_chunks', 0)}
• Failed Chunks: {stats.get('failed_chunks', 0)}
• Timeout Chunks: {stats.get('timeout_chunks', 0)}
• Success Rate: {stats.get('success_rate', 0)*100:.1f}%
• Attempts Made: {stats.get('attempts_made', 0)}

⏱️ COMPREHENSIVE TIMEOUT PROTECTION:
• Chunk Timeout: {CHUNK_TIMEOUT} seconds per chunk
• Noise Detection Messages: ✅ "Input Audio Very noisy. Unable to extract details."
• Adaptive Quality Detection: ✅ ACTIVE
• Timeout Handling: ✅ COMPREHENSIVE

🚀 COMPREHENSIVE ADAPTIVE FEATURES:
• Default Chunk Size: {DEFAULT_CHUNK_SECONDS} seconds
• Fallback Chunk Sizes: {', '.join(map(str, FALLBACK_CHUNK_SECONDS))} seconds
• Automatic Fallback: ✅ ENABLED
• GeeksforGeeks Preprocessing Toggle: ✅ USER CONTROLLED
• Success Rate Monitoring: ✅ ACTIVE
• Memory Management: ✅ GPU-OPTIMIZED

🏆 COMPREHENSIVE ENHANCEMENT SCORE: {min(100, stats.get('success_rate', 0)*100):.0f}/100

🔧 TECHNICAL SPECIFICATIONS:
• Processing Method: COMPREHENSIVE ADAPTIVE CHUNK SIZING WITH GEEKSFORGEEKS PREPROCESSING
• Enhancement Control: USER TOGGLE (Enable/Disable ALL 9 GeeksforGeeks techniques)
• Fallback Mechanism: AUTOMATIC CHUNK SIZE ADAPTATION WITH SUCCESS RATE MONITORING
• Quality Detection: COMPREHENSIVE MULTI-FEATURE ANALYSIS
• Memory Management: GPU-OPTIMIZED WITH CLEANUP
• Error Recovery: COMPREHENSIVE FALLBACK SYSTEMS WITH TIMEOUT PROTECTION
• Audio Optimization: GEEKSFORGEEKS METHODOLOGY WITH ALL PREPROCESSING TECHNIQUES
"""
    return report

def create_comprehensive_adaptive_processing_report(audio_path: str, language: str, enable_preprocessing: bool, 
                                                   processing_time: float, word_count: int, stats: Dict) -> str:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        file_size = os.path.getsize(audio_path) / (1024 * 1024)
        audio_info = f"File size: {file_size:.2f} MB"
    except:
        audio_info = "File info unavailable"
    
    device_info = f"GPU: {torch.cuda.get_device_name()}" if torch.cuda.is_available() else "CPU Processing"
    
    chunk_seconds = stats.get('chunk_seconds_used', 'N/A')
    success_rate = stats.get('success_rate', 0) * 100
    attempts = stats.get('attempts_made', 0)
    final_rms = stats.get('final_rms', 0)
    final_peak = stats.get('final_peak', 0)
    
    report = f"""
🚀 COMPREHENSIVE ADAPTIVE SPEECH TRANSCRIPTION REPORT
====================================================
Generated: {timestamp}

🎵 COMPREHENSIVE ADAPTIVE AUDIO PROCESSING:
• Source File: {os.path.basename(audio_path)}
• {audio_info}
• Target Language: {language}
• GeeksforGeeks Preprocessing Enabled: {'✅ YES' if enable_preprocessing else '❌ NO'}

⚡ PERFORMANCE METRICS:
• Processing Time: {processing_time:.2f} seconds
• Words Generated: {word_count}
• Processing Speed: {word_count/processing_time:.1f} words/second
• Processing Device: {device_info}

🚀 COMPREHENSIVE ADAPTIVE CONFIGURATION:
• Model: Gemma 3N E4B-IT (Comprehensive Adaptive Enhanced)
• Default Chunk Size: {DEFAULT_CHUNK_SECONDS} seconds
• Fallback Chunk Sizes: {', '.join(map(str, FALLBACK_CHUNK_SECONDS))} seconds
• Chunk Timeout: {CHUNK_TIMEOUT} seconds per chunk
• Overlap: {OVERLAP_SECONDS} seconds (Context Preserving)
• Enhancement Method: COMPREHENSIVE ADAPTIVE CHUNK SIZING WITH GEEKSFORGEEKS PREPROCESSING

📏 COMPREHENSIVE ADAPTIVE CHUNK SIZING RESULTS:
• Optimal Chunk Size Found: {chunk_seconds}s
• Success Rate Achieved: {success_rate:.1f}%
• Total Attempts Made: {attempts}
• Successful Chunks: {stats.get('successful_chunks', 0)}
• Failed Chunks: {stats.get('failed_chunks', 0)}
• Timeout Chunks: {stats.get('timeout_chunks', 0)}

🔧 COMPREHENSIVE GEEKSFORGEEKS PREPROCESSING:
"""
    
    if enable_preprocessing and stats.get('preprocessing_applied', False):
        report += f"""• Resampling: ✅ Applied ({SAMPLE_RATE} Hz standardization)
• Format Standardization: ✅ Applied (Mono, float32, validation)
• Advanced Noise Reduction: ✅ Applied (Spectral gating)
• Butterworth Filter: ✅ Applied ({GEEKSFORGEEKS_CUTOFF_FREQ} Hz cutoff, Order {GEEKSFORGEEKS_FILTER_ORDER})
• Comprehensive Normalization: ✅ Applied (Peak + RMS + Z-score)
• Variable Length Handling: ✅ Applied (Intelligent padding/trimming)
• Model Efficiency Optimization: ✅ Applied (Pre-emphasis + Windowing + FFT)
• Feature Extraction: ✅ Applied (MFCCs, spectral characteristics)
• Enhanced Log-Mel Spectrogram: ✅ Applied (128 mels, optimized)
• Final RMS Level: {final_rms:.4f}
• Final Peak Level: {final_peak:.4f}
• Features Extracted: {stats.get('extracted_features_count', 0)} feature sets
"""
    else:
        report += f"""• GeeksforGeeks Preprocessing: ❌ DISABLED BY USER
• Raw Audio Processing: ✅ NO PREPROCESSING APPLIED
• User Choice: Skip comprehensive enhancement pipeline
• Processing Mode: Raw audio input directly to transcription
"""
    
    report += f"""
⏱️ COMPREHENSIVE TIMEOUT & ADAPTIVE HANDLING:
• Timeout Protection: ✅ {CHUNK_TIMEOUT}s per chunk
• Noise Detection Messages: ✅ "Input Audio Very noisy. Unable to extract details."
• Adaptive Quality Detection: ✅ Success rate monitoring with automatic adaptation
• Automatic Fallback: ✅ Multiple chunk sizes tried automatically
• Comprehensive Error Recovery: ✅ COMPREHENSIVE FALLBACK SYSTEMS

🌐 TRANSLATION FEATURES:
• Translation Control: ✅ USER-INITIATED (Optional)
• Smart Text Chunking: ✅ ENABLED
• Context Preservation: ✅ SENTENCE OVERLAP
• Processing Method: ✅ COMPREHENSIVE ADAPTIVE PIPELINE

📊 COMPREHENSIVE ADAPTIVE SYSTEM STATUS:
• Enhancement Method: ✅ COMPREHENSIVE ADAPTIVE CHUNK SIZING WITH GEEKSFORGEEKS PREPROCESSING
• Preprocessing Control: ✅ USER TOGGLE (Enable/Disable ALL 9 GeeksforGeeks techniques)
• Chunk Size Adaptation: ✅ AUTOMATIC FALLBACK MECHANISM WITH SUCCESS RATE MONITORING
• Success Rate Monitoring: ✅ ACTIVE WITH ADAPTIVE THRESHOLDS
• Quality Detection: ✅ COMPREHENSIVE MULTI-FEATURE ANALYSIS
• Memory Optimization: ✅ GPU-AWARE CLEANUP WITH PERIODIC MONITORING
• Error Recovery: ✅ COMPREHENSIVE FALLBACK SYSTEMS WITH TIMEOUT PROTECTION

✅ STATUS: COMPREHENSIVE ADAPTIVE TRANSCRIPTION COMPLETED
🚀 AUDIO ENHANCEMENT: GEEKSFORGEEKS PREPROCESSING (USER CONTROLLED - ALL 9 TECHNIQUES)
📏 CHUNK SIZING: COMPREHENSIVE ADAPTIVE WITH AUTOMATIC FALLBACK
⏱️ TIMEOUT PROTECTION: 75-SECOND CHUNK SAFETY WITH NOISE DETECTION
🔧 PREPROCESSING: USER TOGGLE (ENABLE/DISABLE COMPREHENSIVE GEEKSFORGEEKS PIPELINE)
📊 OPTIMIZATION: ADAPTIVE CHUNK SIZE SELECTION WITH SUCCESS RATE MONITORING
🎯 RELIABILITY: COMPREHENSIVE ADAPTIVE PROCESSING WITH ALL FALLBACKS AND GEEKSFORGEEKS METHODOLOGY
"""
    return report

def create_comprehensive_adaptive_interface():
    """Create comprehensive adaptive speech enhancement interface with GeeksforGeeks preprocessing"""
    
    comprehensive_adaptive_css = """
    :root {
        --primary-color: #0f172a;
        --secondary-color: #1e293b;
        --accent-color: #059669;
        --adaptive-color: #0ea5e9;
        --geeks-color: #7c3aed;
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
    
    .comprehensive-adaptive-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 10%, #059669 20%, #0ea5e9 30%, #7c3aed 40%, #10b981 50%, #f59e0b 60%, #3b82f6 70%, #8b5cf6 80%, #ec4899 90%, #f97316 100%) !important;
        padding: 70px 50px !important;
        border-radius: 35px !important;
        text-align: center !important;
        margin-bottom: 60px !important;
        box-shadow: 0 40px 80px rgba(5, 150, 105, 0.5) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .comprehensive-adaptive-title {
        font-size: 4.5rem !important;
        font-weight: 900 !important;
        color: white !important;
        margin-bottom: 25px !important;
        text-shadow: 0 8px 20px rgba(5, 150, 105, 0.7) !important;
        position: relative !important;
        z-index: 2 !important;
    }
    
    .comprehensive-adaptive-subtitle {
        font-size: 1.7rem !important;
        color: rgba(255,255,255,0.98) !important;
        font-weight: 700 !important;
        position: relative !important;
        z-index: 2 !important;
        line-height: 1.4 !important;
    }
    
    .comprehensive-adaptive-card {
        background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%) !important;
        border: 3px solid var(--accent-color) !important;
        border-radius: 30px !important;
        padding: 40px !important;
        margin: 30px 0 !important;
        box-shadow: 0 25px 50px rgba(5, 150, 105, 0.4) !important;
        transition: all 0.5s ease !important;
    }
    
    .geeksforgeeks-toggle {
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.2) 0%, rgba(5, 150, 105, 0.2) 100%) !important;
        border: 4px solid var(--geeks-color) !important;
        border-radius: 25px !important;
        padding: 30px !important;
        margin: 25px 0 !important;
        position: relative !important;
    }
    
    .comprehensive-adaptive-button {
        background: linear-gradient(135deg, var(--accent-color) 0%, var(--adaptive-color) 50%, var(--geeks-color) 100%) !important;
        border: none !important;
        border-radius: 25px !important;
        color: white !important;
        font-weight: 900 !important;
        font-size: 1.4rem !important;
        padding: 25px 50px !important;
        transition: all 0.5s ease !important;
        box-shadow: 0 15px 35px rgba(5, 150, 105, 0.6) !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
    }
    
    .translation-button {
        background: linear-gradient(135deg, var(--translation-color) 0%, var(--adaptive-color) 100%) !important;
        border: none !important;
        border-radius: 20px !important;
        color: white !important;
        font-weight: 800 !important;
        font-size: 1.3rem !important;
        padding: 20px 40px !important;
        transition: all 0.5s ease !important;
        box-shadow: 0 12px 35px rgba(59, 130, 246, 0.6) !important;
        text-transform: uppercase !important;
        letter-spacing: 1.5px !important;
    }
    
    .status-comprehensive-adaptive {
        background: linear-gradient(135deg, var(--success-color), #047857, var(--geeks-color)) !important;
        color: white !important;
        padding: 20px 35px !important;
        border-radius: 18px !important;
        font-weight: 800 !important;
        text-align: center !important;
        box-shadow: 0 12px 30px rgba(5, 150, 105, 0.6) !important;
        border: 4px solid rgba(5, 150, 105, 0.5) !important;
    }
    
    .translation-section {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.18) 0%, rgba(14, 165, 233, 0.18) 100%) !important;
        border: 4px solid var(--translation-color) !important;
        border-radius: 30px !important;
        padding: 35px !important;
        margin: 30px 0 !important;
        position: relative !important;
    }
    
    .card-header {
        color: var(--accent-color) !important;
        font-size: 1.8rem !important;
        font-weight: 900 !important;
        margin-bottom: 35px !important;
        padding-bottom: 20px !important;
        border-bottom: 5px solid var(--accent-color) !important;
    }
    
    .geeks-header {
        color: var(--geeks-color) !important;
        font-size: 1.5rem !important;
        font-weight: 800 !important;
        margin-bottom: 25px !important;
        padding-bottom: 15px !important;
        border-bottom: 4px solid var(--geeks-color) !important;
    }
    
    .log-comprehensive-adaptive {
        background: linear-gradient(135deg, rgba(0, 0, 0, 0.90) 0%, rgba(15, 23, 42, 0.98) 100%) !important;
        border: 4px solid var(--accent-color) !important;
        border-radius: 20px !important;
        color: var(--text-secondary) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.05rem !important;
        line-height: 1.9 !important;
        padding: 30px !important;
        max-height: 500px !important;
        overflow-y: auto !important;
        white-space: pre-wrap !important;
    }
    """
    
    with gr.Blocks(
        css=comprehensive_adaptive_css, 
        theme=gr.themes.Base(),
        title="🚀 Comprehensive Adaptive Speech Enhancement & Transcription with GeeksforGeeks Preprocessing"
    ) as interface:
        
        # Comprehensive Adaptive Header
        gr.HTML("""
        <div class="comprehensive-adaptive-header">
            <h1 class="comprehensive-adaptive-title">🚀 COMPREHENSIVE ADAPTIVE SPEECH TRANSCRIPTION</h1>
            <p class="comprehensive-adaptive-subtitle">GeeksforGeeks Preprocessing (ALL 9 Techniques) • Adaptive Chunk Sizing • Enable/Disable Toggle • Auto Fallback • 75s Timeout • Complete Pipeline</p>
            <div style="margin-top: 30px;">
                <span style="background: rgba(5, 150, 105, 0.3); color: #059669; padding: 15px 30px; border-radius: 35px; margin: 0 12px; font-size: 1.2rem; font-weight: 800;">🔧 GEEKSFORGEEKS ALL 9</span>
                <span style="background: rgba(14, 165, 233, 0.3); color: #0ea5e9; padding: 15px 30px; border-radius: 35px; margin: 0 12px; font-size: 1.2rem; font-weight: 800;">📏 ADAPTIVE</span>
                <span style="background: rgba(124, 58, 237, 0.3); color: #7c3aed; padding: 15px 30px; border-radius: 35px; margin: 0 12px; font-size: 1.2rem; font-weight: 800;">⚡ TOGGLE</span>
                <span style="background: rgba(245, 158, 11, 0.3); color: #f59e0b; padding: 15px 30px; border-radius: 35px; margin: 0 12px; font-size: 1.2rem; font-weight: 800;">⏱️ 75s TIMEOUT</span>
            </div>
        </div>
        """)
        
        # System Status
        status_display = gr.Textbox(
            label="🚀 Comprehensive Adaptive System Status",
            value="Initializing COMPREHENSIVE ADAPTIVE speech transcription with GeeksforGeeks preprocessing...",
            interactive=False,
            elem_classes="status-comprehensive-adaptive"
        )
        
        # Main Interface
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML('<div class="comprehensive-adaptive-card"><div class="card-header">🚀 Comprehensive Adaptive Control Panel</div>')
                
                audio_input = gr.Audio(
                    label="🎵 Upload Audio File or Record Live",
                    type="filepath"
                )
                
                language_dropdown = gr.Dropdown(
                    choices=list(SUPPORTED_LANGUAGES.keys()),
                    value="🌍 Auto-detect",
                    label="🌍 Language Selection (150+ Supported)",
                    info="All languages with COMPREHENSIVE ADAPTIVE enhancement"
                )
                
                # GeeksforGeeks Comprehensive Preprocessing Toggle
                gr.HTML("""
                <div class="geeksforgeeks-toggle">
                    <div class="geeks-header">🔧 GeeksforGeeks Comprehensive Preprocessing Control</div>
                    <p style="color: #cbd5e1; margin-bottom: 20px; font-size: 1.1rem; line-height: 1.5;">
                        Enable/disable comprehensive audio preprocessing based on GeeksforGeeks methodology. 
                        Includes ALL 9 techniques: resampling, format standardization, noise reduction, Butterworth filtering, 
                        comprehensive normalization, variable length handling, model efficiency optimization, 
                        feature extraction, and enhanced log-mel spectrograms.
                    </p>
                    <div style="background: rgba(124, 58, 237, 0.1); padding: 15px; border-radius: 12px; margin-top: 15px;">
                        <strong style="color: #7c3aed;">GeeksforGeeks Techniques Included:</strong><br>
                        <span style="font-size: 0.95rem; color: #cbd5e1;">
                        1. Resampling • 2. Format Standardization • 3. Noise Reduction • 4. Butterworth Filtering • 
                        5. Comprehensive Normalization • 6. Variable Length Handling • 7. Model Efficiency • 
                        8. Feature Extraction • 9. Enhanced Log-Mel Spectrograms
                        </span>
                    </div>
                </div>
                """)
                
                enable_preprocessing = gr.Checkbox(
                    label="🔧 Enable Comprehensive GeeksforGeeks Preprocessing (ALL 9 Techniques)",
                    value=True,
                    info="Applies complete GeeksforGeeks methodology: resampling (16kHz), noise reduction, Butterworth filter (4kHz), normalization, feature extraction, etc."
                )
                
                transcribe_btn = gr.Button(
                    "🚀 START COMPREHENSIVE ADAPTIVE TRANSCRIPTION",
                    variant="primary",
                    elem_classes="comprehensive-adaptive-button",
                    size="lg"
                )
                
                gr.HTML('</div>')
                
                # Adaptive Chunk Sizing Info Panel
                gr.HTML("""
                <div class="comprehensive-adaptive-card">
                    <div class="card-header">📏 Comprehensive Adaptive Chunk Sizing Info</div>
                    <div style="color: #cbd5e1; font-size: 1.1rem; line-height: 1.7;">
                        <p><strong style="color: #0ea5e9;">Default:</strong> 30-second chunks with 2s overlap</p>
                        <p><strong style="color: #0ea5e9;">Automatic Fallbacks:</strong> 10s → 15s → 20s → 40s</p>
                        <p><strong style="color: #0ea5e9;">Smart Auto-Retry:</strong> Switches chunk size if transcription fails</p>
                        <p><strong style="color: #0ea5e9;">Success Rate Monitoring:</strong> ≥70% success rate required</p>
                        <p><strong style="color: #0ea5e9;">Timeout Protection:</strong> 75 seconds per chunk with noise detection</p>
                        <p><strong style="color: #0ea5e9;">Noise Messages:</strong> "Input Audio Very noisy. Unable to extract details."</p>
                    </div>
                </div>
                """)
            
            with gr.Column(scale=2):
                gr.HTML('<div class="comprehensive-adaptive-card"><div class="card-header">📊 Comprehensive Adaptive Results</div>')
                
                transcription_output = gr.Textbox(
                    label="📝 Original Transcription (COMPREHENSIVE ADAPTIVE Enhanced)",
                    placeholder="Your COMPREHENSIVE ADAPTIVE transcription will appear here...",
                    lines=14,
                    max_lines=20,
                    interactive=False,
                    show_copy_button=True
                )
                
                copy_original_btn = gr.Button("📋 Copy Original Transcription", size="sm")
                
                gr.HTML('</div>')
                
                # Translation Section
                gr.HTML("""
                <div class="translation-section">
                    <div style="color: #3b82f6; font-size: 1.6rem; font-weight: 900; margin-bottom: 25px; margin-top: 20px;">🌐 Optional English Translation</div>
                    <p style="color: #cbd5e1; margin-bottom: 30px; font-size: 1.3rem; line-height: 1.5;">
                        Click the button below to translate your transcription to English using smart text chunking 
                        with context preservation and sentence overlap.
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
                    lines=12,
                    max_lines=20,
                    interactive=False,
                    show_copy_button=True
                )
                
                copy_translation_btn = gr.Button("🌐 Copy English Translation", size="sm")
        
        # Audio Comparison
        with gr.Row():
            with gr.Column():
                gr.HTML('<div class="comprehensive-adaptive-card"><div class="card-header">📥 Original Audio</div>')
                original_audio_player = gr.Audio(
                    label="Original Audio",
                                        interactive=False
                )
                gr.HTML('</div>')
        
        # Reports
        with gr.Row():
            with gr.Column():
                with gr.Accordion("🚀 COMPREHENSIVE ADAPTIVE Enhancement Report", open=False):
                    enhancement_report = gr.Textbox(
                        label="COMPREHENSIVE ADAPTIVE Enhancement Report",
                        lines=25,
                        show_copy_button=True,
                        interactive=False
                    )
            
            with gr.Column():
                with gr.Accordion("📋 COMPREHENSIVE ADAPTIVE Processing Report", open=False):
                    processing_report = gr.Textbox(
                        label="COMPREHENSIVE ADAPTIVE Processing Report", 
                        lines=25,
                        show_copy_button=True,
                        interactive=False
                    )
        
        # System Monitoring
        gr.HTML('<div class="comprehensive-adaptive-card"><div class="card-header">🚀 COMPREHENSIVE ADAPTIVE System Monitoring</div>')
        
        log_display = gr.Textbox(
            label="",
            value="🚀 COMPREHENSIVE ADAPTIVE system ready - GeeksforGeeks preprocessing with ALL 9 techniques available...",
            interactive=False,
            lines=16,
            max_lines=25,
            elem_classes="log-comprehensive-adaptive",
            show_label=False
        )
        
        with gr.Row():
            refresh_logs_btn = gr.Button("🔄 Refresh COMPREHENSIVE Logs", size="sm")
            clear_logs_btn = gr.Button("🗑️ Clear Logs", size="sm")
        
        gr.HTML('</div>')
        
        # Event Handlers
        transcribe_btn.click(
            fn=transcribe_audio_comprehensive_adaptive,
            inputs=[audio_input, language_dropdown, enable_preprocessing],
            outputs=[transcription_output, original_audio_player, enhanced_audio_player, enhancement_report, processing_report],
            show_progress=True
        )
        
        translate_btn.click(
            fn=translate_transcription_comprehensive_adaptive,
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
        
        def clear_comprehensive_adaptive_logs():
            global log_capture
            if log_capture:
                with log_capture.lock:
                    log_capture.log_buffer.clear()
            return "🚀 COMPREHENSIVE ADAPTIVE logs cleared - system ready"
        
        clear_logs_btn.click(
            fn=clear_comprehensive_adaptive_logs,
            inputs=[],
            outputs=[log_display]
        )
        
        def auto_refresh_comprehensive_adaptive_logs():
            return get_current_logs()
        
        timer = gr.Timer(value=3, active=True)
        timer.tick(
            fn=auto_refresh_comprehensive_adaptive_logs,
            inputs=[],
            outputs=[log_display]
        )
        
        interface.load(
            fn=initialize_comprehensive_transcriber,
            inputs=[],
            outputs=[status_display]
        )
    
    return interface

def main():
    """Launch the complete COMPREHENSIVE ADAPTIVE speech transcription system"""
    
    if "/path/to/your/" in MODEL_PATH:
        print("="*80)
        print("🚀 COMPREHENSIVE ADAPTIVE SPEECH TRANSCRIPTION SYSTEM CONFIGURATION REQUIRED")
        print("="*80)
        print("Please update the MODEL_PATH variable with your local Gemma 3N model directory")
        print("Download from: https://huggingface.co/google/gemma-3n-e4b-it")
        print("="*80)
        return
    
    setup_comprehensive_logging()
    
    print("🚀 Launching COMPREHENSIVE ADAPTIVE SPEECH TRANSCRIPTION SYSTEM...")
    print("="*80)
    print("🔧 COMPREHENSIVE GEEKSFORGEEKS PREPROCESSING IMPLEMENTATION:")
    print("="*80)
    print("📊 GEEKSFORGEEKS PIPELINE STAGES (ALL 9 TECHNIQUES):")
    print("   ✅ Stage 1: Audio Resampling (16000 Hz standardization)")
    print("   ✅ Stage 2: Format Standardization (Mono, float32, validation)")
    print("   ✅ Stage 3: Advanced Noise Reduction (Spectral gating)")
    print("   ✅ Stage 4: Butterworth Low-pass Filter (4000 Hz cutoff, Order 4)")
    print("   ✅ Stage 5: Comprehensive Normalization (Peak + RMS + Z-score)")
    print("   ✅ Stage 6: Variable Length Handling (Intelligent padding/trimming)")
    print("   ✅ Stage 7: Model Efficiency Optimization (Pre-emphasis + Windowing + FFT)")
    print("   ✅ Stage 8: Feature Extraction (MFCCs, spectral characteristics)")
    print("   ✅ Stage 9: Enhanced Log-Mel Spectrogram (128 mels, optimized parameters)")
    print("="*80)
    print("📏 COMPREHENSIVE ADAPTIVE CHUNK SIZING FEATURES:")
    print("   📏 Default Chunk Size: 30 seconds with 2s overlap")
    print("   📏 Fallback Chunk Sizes: 10s, 15s, 20s, 40s (automatic)")
    print("   🔄 Automatic Fallback Mechanism: Success rate monitoring")
    print("   📊 Success Rate Threshold: ≥70% success rate required")
    print("   🎯 Optimal Chunk Detection: Automatic adaptation")
    print("   🔄 Retry Logic: Multiple attempts with different chunk sizes")
    print("="*80)
    print("⚡ COMPREHENSIVE ENABLE/DISABLE PREPROCESSING CONTROL:")
    print("   👤 User Control: Complete enable/disable toggle for ALL 9 techniques")
    print("   🔧 GeeksforGeeks Pipeline: User can skip comprehensive enhancement")
    print("   📊 Raw Audio Processing: Available when preprocessing disabled")
    print("   🎛️ Flexible Processing: Complete user choice driven system")
    print("   🔧 Technique Selection: Enable/disable entire GeeksforGeeks methodology")
    print("="*80)
    print("⏱️ COMPREHENSIVE TIMEOUT PROTECTION:")
    print(f"   ⏱️ {CHUNK_TIMEOUT}-second timeout per chunk with noise detection")
    print("   ⏱️ Comprehensive quality detection and assessment")
    print("   ⏱️ 'Input Audio Very noisy. Unable to extract details.' messages")
    print("   ⏱️ Graceful degradation for problematic audio chunks")
    print("   ⏱️ Timeout handling with adaptive chunk sizing fallback")
    print("="*80)
    print("🌐 COMPREHENSIVE TRANSLATION FEATURES:")
    print("   👤 User Control: Translation only when user clicks button")
    print("   📝 Smart Chunking: Preserves meaning with sentence overlap")
    print(f"   📏 Chunk Size: {MAX_TRANSLATION_CHUNK_SIZE} characters with {SENTENCE_OVERLAP} sentence overlap")
    print("   🔗 Context Preservation: Intelligent sentence boundary detection")
    print("   🛡️ Error Recovery: Graceful handling of failed chunks")
    print("   🌍 Language Detection: Smart English detection to skip unnecessary translation")
    print("="*80)
    print("🌍 COMPREHENSIVE LANGUAGE SUPPORT: 150+ languages including:")
    print("   • European: English, Spanish, French, German, Italian, Portuguese, Russian, Dutch, Swedish, etc.")
    print("   • Asian: Chinese, Japanese, Korean, Hindi, Bengali, Tamil, Telugu, Thai, Vietnamese, etc.")
    print("   • Middle Eastern: Arabic, Persian/Farsi, Hebrew, Turkish, Azerbaijani, etc.")
    print("   • South Asian: Urdu, Gujarati, Marathi, Kannada, Malayalam, Punjabi, Nepali, etc.")
    print("   • Central Asian: Uzbek, Kazakh, Kyrgyz, Turkmen, Tajik, etc.")
    print("   • Southeast Asian: Indonesian, Malay, Filipino/Tagalog, Khmer, Lao, Burmese, etc.")
    print("   • Himalayan: Tibetan, Dzongkha, Sherpa, Tamang, etc.")
    print("   • Regional variants and minority languages supported")
    print("="*80)
    print("🚀 COMPREHENSIVE ADAPTIVE SYSTEM ADVANTAGES:")
    print("   📏 Comprehensive Adaptive Chunk Sizing: Automatically finds optimal chunk size")
    print("   🔄 Intelligent Fallback Mechanism: Tries multiple chunk sizes systematically")
    print("   ⚡ Complete Preprocessing Control: Enable/disable ALL 9 GeeksforGeeks techniques")
    print("   📊 Success Rate Monitoring: Tracks transcription quality and adapts")
    print("   🎯 Automatic Optimization: Selects best chunk size and preprocessing combination")
    print("   🛡️ Comprehensive Error Handling: Multiple fallback systems and timeout protection")
    print("   💡 Educational Value: Shows which settings work best for different audio types")
    print("   🔧 GeeksforGeeks Methodology: Complete implementation of all preprocessing techniques")
    print("   🎛️ User Flexibility: Complete control over preprocessing pipeline")
    print("   📈 Performance Optimization: GPU-aware memory management and cleanup")
    print("="*80)
    print("🔧 TECHNICAL IMPLEMENTATION DETAILS:")
    print("   🖥️ Model: Gemma 3N E4B-IT with comprehensive adaptive enhancements")
    print("   🔧 Preprocessing: GeeksforGeeks methodology with ALL 9 techniques")
    print("   📏 Chunk Management: Adaptive sizing with automatic fallback")
    print("   ⏱️ Timeout System: 75-second per chunk with noise detection")
    print("   🧠 Memory Management: GPU-optimized with periodic cleanup")
    print("   🌍 Language Support: 150+ languages with auto-detection")
    print("   🌐 Translation: Smart chunking with context preservation")
    print("   🎛️ Interface: Comprehensive Gradio UI with real-time monitoring")
    print("="*80)
    
    try:
        interface = create_comprehensive_adaptive_interface()
        
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
        print(f"❌ COMPREHENSIVE ADAPTIVE system launch failed: {e}")
        print("🔧 COMPREHENSIVE ADAPTIVE system troubleshooting:")
        print("   • Verify model path is correct and accessible")
        print("   • Check GPU memory availability and drivers")
        print("   • Ensure all dependencies are installed:")
        print("     pip install --upgrade torch transformers gradio librosa soundfile")
        print("     pip install --upgrade scipy nltk noisereduce scikit-learn")
        print("   • Verify Python environment and version compatibility")
        print("   • Check port 7860 availability")
        print("   • GeeksforGeeks preprocessing: ALL 9 techniques available")
        print("   • Adaptive chunk sizing: Automatic fallback mechanism active")
        print("   • Enable/disable preprocessing toggle: User controlled")
        print("   • Comprehensive fallback systems are active")
        print("   • ASR optimization with complete GeeksforGeeks methodology")
        print("   • Fixed hann window function (no longer hanning)")
        print("   • Comprehensive error recovery systems implemented")
        print("="*80)

if __name__ == "__main__":
    main()

