"""
Professional CUDA-Optimized Audio Enhancement System for RTX A4000
================================================================
Enterprise-grade audio enhancement with advanced AI models, professional
quality metrics, and comprehensive audio restoration capabilities.

RTX A4000 Optimized: 16GB VRAM, Ampere Architecture, Tensor Cores
Author: Professional Audio Enhancement System
Version: 4.0.0-Professional-RTX-A4000
"""

import os
import sys
import warnings
import gc
import json
import logging
import time
import pickle
import hashlib
import threading
from pathlib import Path
from typing import Tuple, Dict, Optional, List, Union, Any, Callable
from dataclasses import dataclass, asdict, field
from contextmanager import contextmanager
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from queue import Queue
import multiprocessing as mp

import numpy as np
import scipy.signal as signal
from scipy.optimize import differential_evolution, minimize
from scipy.stats import pearsonr
import librosa
import soundfile as sf
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# GPU-accelerated imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms

# Advanced audio processing
try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False
    print("Warning: noisereduce not available.")

try:
    import pyloudnorm as pyln
    PYLOUDNORM_AVAILABLE = True
except ImportError:
    PYLOUDNORM_AVAILABLE = False
    print("Warning: pyloudnorm not available.")

try:
    from pesq import pesq
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False
    print("Warning: PESQ not available.")

try:
    from pystoi import stoi
    STOI_AVAILABLE = True
except ImportError:
    STOI_AVAILABLE = False
    print("Warning: STOI not available.")

# Suppress warnings
warnings.filterwarnings('ignore')

# ==================== GLOBAL CONFIGURATION ====================
# 🔥 IMPORTANT: SET YOUR AUDIO FILE PATH HERE! 🔥
INPUT_AUDIO_PATH = "input_noisy_audio.wav"  # ← CHANGE THIS TO YOUR ACTUAL FILE PATH!
OUTPUT_AUDIO_PATH = "enhanced_professional_audio.wav"

# RTX A4000 Optimization Settings (16GB VRAM)
CUDA_DEVICE = "cuda:0"
MIXED_PRECISION = True
GPU_MEMORY_FRACTION = 0.85  # Use 13.6GB of 16GB
BATCH_SIZE_GPU = 32         # Large batches for RTX A4000
CHUNK_SIZE = 32768          # Large chunks
PREFETCH_BATCHES = 8        # Multi-stream processing
TENSOR_CORE_OPTIMIZATION = True
MULTI_STREAM_PROCESSING = True

# Professional Model Paths (Download offline and place here)
MODEL_BASE_PATH = Path("models")
PRETRAINED_MODELS = {
    "demucs": MODEL_BASE_PATH / "demucs" / "htdemucs_ft.th",
    "dns_model": MODEL_BASE_PATH / "microsoft" / "dns_model.pth",
    "wav2vec2": MODEL_BASE_PATH / "transformers" / "wav2vec2-base",
    "wavlm": MODEL_BASE_PATH / "transformers" / "wavlm-base",
    "whisper_encoder": MODEL_BASE_PATH / "openai" / "whisper-small.pt",
    "nemo_ssl": MODEL_BASE_PATH / "nemo" / "ssl_en_conformer_ctc_large.nemo"
}

# Reference data paths
REFERENCE_DATA_PATH = Path("reference_data")
NOISE_PROFILES_PATH = REFERENCE_DATA_PATH / "noise_profiles"
TEST_SIGNALS_PATH = REFERENCE_DATA_PATH / "test_signals"
# ================================================================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('audio_enhancement.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def setup_rtx_a4000_environment():
    """Setup and optimize RTX A4000 environment"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! This script requires RTX A4000 GPU.")
    
    device = torch.device(CUDA_DEVICE)
    torch.cuda.set_device(device)
    
    # Get RTX A4000 info
    gpu_name = torch.cuda.get_device_name(device)
    gpu_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
    compute_capability = torch.cuda.get_device_properties(device).major
    
    if "A4000" not in gpu_name and compute_capability < 8:
        logger.warning(f"Optimized for RTX A4000, detected: {gpu_name}")
    
    logger.info(f"RTX GPU: {gpu_name}")
    logger.info(f"VRAM: {gpu_memory:.1f} GB")
    logger.info(f"Compute Capability: {compute_capability}.{torch.cuda.get_device_properties(device).minor}")
    logger.info(f"Mixed Precision: {'Enabled' if MIXED_PRECISION else 'Disabled'}")
    logger.info(f"Tensor Cores: {'Enabled' if TENSOR_CORE_OPTIMIZATION else 'Disabled'}")
    
    # RTX A4000 optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True  # Tensor Core optimization
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # Memory optimization
    if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
        torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)
    
    # Multi-stream setup
    if MULTI_STREAM_PROCESSING:
        torch.cuda.set_sync_debug_mode(0)  # Disable sync debugging for performance
    
    return device

@dataclass
class ProfessionalEnhancementConfig:
    """Professional RTX A4000 optimized configuration"""
    
    # Audio parameters
    sample_rate: int = 16000
    frame_length: int = 2048      # Increased for RTX A4000
    hop_length: int = 512         # Increased for RTX A4000
    n_fft: int = 2048            # Increased for RTX A4000
    win_length: int = 2048
    
    # RTX A4000 optimized parameters
    cuda_device: str = CUDA_DEVICE
    mixed_precision: bool = MIXED_PRECISION
    gpu_batch_size: int = BATCH_SIZE_GPU
    chunk_size: int = CHUNK_SIZE
    overlap_size: int = 4096      # Increased overlap
    prefetch_batches: int = PREFETCH_BATCHES
    max_gpu_memory_gb: int = int(GPU_MEMORY_FRACTION * 16)
    
    # Professional enhancement parameters
    wiener_alpha: float = 0.95
    spectral_floor: float = 0.001
    vad_threshold: float = 0.6
    noise_gate_threshold: float = -35
    compression_ratio: float = 2.5
    target_loudness: float = -23.0
    
    # Advanced processing
    multi_band_processing: bool = True
    harmonic_enhancement: bool = True
    phase_recovery: bool = True
    psychoacoustic_modeling: bool = True
    temporal_coherence: bool = True
    
    # Model ensemble settings
    ensemble_voting: bool = True
    ensemble_weights: List[float] = field(default_factory=lambda: [0.3, 0.25, 0.2, 0.15, 0.1])
    
    # Quality control
    quality_threshold: float = 0.1  # Minimum improvement threshold
    max_processing_iterations: int = 3
    adaptive_enhancement: bool = True

class RTXMemoryManager:
    """Advanced RTX A4000 memory management"""
    
    def __init__(self, device):
        self.device = device
        self.peak_memory = 0
        self.memory_pools = {}
        self.stream_pool = []
        self.allocation_history = deque(maxlen=1000)
        
        # Create CUDA streams for multi-stream processing
        if MULTI_STREAM_PROCESSING:
            for i in range(4):  # 4 streams for parallel processing
                self.stream_pool.append(torch.cuda.Stream())
    
    def get_stream(self, stream_id: int = 0):
        """Get CUDA stream for parallel processing"""
        if self.stream_pool and stream_id < len(self.stream_pool):
            return self.stream_pool[stream_id]
        return torch.cuda.current_stream()
    
    def create_memory_pool(self, name: str, size_mb: int):
        """Create dedicated memory pool"""
        try:
            pool_size = size_mb * 1024 * 1024
            self.memory_pools[name] = torch.cuda.memory.MemoryPool(self.device)
            logger.info(f"Created memory pool '{name}': {size_mb}MB")
        except Exception as e:
            logger.warning(f"Failed to create memory pool {name}: {e}")
    
    def clear_cache_aggressive(self):
        """Aggressive cache clearing for RTX A4000"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
    
    def get_detailed_memory_info(self):
        """Get detailed GPU memory information"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device) / 1e9
            reserved = torch.cuda.memory_reserved(self.device) / 1e9
            max_allocated = torch.cuda.max_memory_allocated(self.device) / 1e9
            max_reserved = torch.cuda.max_memory_reserved(self.device) / 1e9
            
            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'max_allocated_gb': max_allocated,
                'max_reserved_gb': max_reserved,
                'free_gb': 16.0 - reserved,  # RTX A4000 has 16GB
                'utilization_pct': (reserved / 16.0) * 100
            }
        return {}
    
    @contextmanager
    def memory_context(self, operation_name="operation", expected_memory_gb=1.0):
        """Advanced memory context manager"""
        self.clear_cache_aggressive()
        start_info = self.get_detailed_memory_info()
        start_time = time.time()
        
        try:
            yield
        finally:
            end_info = self.get_detailed_memory_info()
            end_time = time.time()
            
            memory_used = end_info['allocated_gb'] - start_info['allocated_gb']
            time_taken = end_time - start_time
            
            self.allocation_history.append({
                'operation': operation_name,
                'memory_used_gb': memory_used,
                'time_taken_s': time_taken,
                'timestamp': end_time
            })
            
            logger.debug(f"{operation_name}: {memory_used:.2f}GB memory, {time_taken:.3f}s")
            
            if memory_used > expected_memory_gb * 2:
                logger.warning(f"{operation_name} used {memory_used:.2f}GB, expected {expected_memory_gb:.2f}GB")
            
            self.clear_cache_aggressive()

class ProfessionalQualityMetrics:
    """Professional audio quality assessment"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.reference_signals = {}
        self._load_reference_signals()
    
    def _load_reference_signals(self):
        """Load reference signals for quality assessment"""
        try:
            if TEST_SIGNALS_PATH.exists():
                for ref_file in TEST_SIGNALS_PATH.glob("*.wav"):
                    signal_data, sr = librosa.load(ref_file, sr=self.sample_rate)
                    self.reference_signals[ref_file.stem] = signal_data
                logger.info(f"Loaded {len(self.reference_signals)} reference signals")
        except Exception as e:
            logger.warning(f"Failed to load reference signals: {e}")
    
    def calculate_pesq(self, reference: np.ndarray, enhanced: np.ndarray) -> float:
        """Calculate PESQ score"""
        if not PESQ_AVAILABLE:
            return self._estimate_pesq(reference, enhanced)
        
        try:
            # Ensure same length
            min_len = min(len(reference), len(enhanced))
            ref_seg = reference[:min_len]
            enh_seg = enhanced[:min_len]
            
            # PESQ expects specific sample rates
            if self.sample_rate == 16000:
                pesq_score = pesq(self.sample_rate, ref_seg, enh_seg, 'wb')
            elif self.sample_rate == 8000:
                pesq_score = pesq(self.sample_rate, ref_seg, enh_seg, 'nb')
            else:
                # Resample for PESQ
                ref_16k = librosa.resample(ref_seg, orig_sr=self.sample_rate, target_sr=16000)
                enh_16k = librosa.resample(enh_seg, orig_sr=self.sample_rate, target_sr=16000)
                pesq_score = pesq(16000, ref_16k, enh_16k, 'wb')
            
            return pesq_score
        except Exception as e:
            logger.warning(f"PESQ calculation failed: {e}")
            return self._estimate_pesq(reference, enhanced)
    
    def calculate_stoi(self, reference: np.ndarray, enhanced: np.ndarray) -> float:
        """Calculate STOI score"""
        if not STOI_AVAILABLE:
            return self._estimate_stoi(reference, enhanced)
        
        try:
            min_len = min(len(reference), len(enhanced))
            stoi_score = stoi(reference[:min_len], enhanced[:min_len], 
                            self.sample_rate, extended=True)
            return stoi_score
        except Exception as e:
            logger.warning(f"STOI calculation failed: {e}")
            return self._estimate_stoi(reference, enhanced)
    
    def _estimate_pesq(self, reference: np.ndarray, enhanced: np.ndarray) -> float:
        """Estimate PESQ-like score using correlation and SNR"""
        try:
            min_len = min(len(reference), len(enhanced))
            ref_seg = reference[:min_len]
            enh_seg = enhanced[:min_len]
            
            # Correlation component
            correlation = np.corrcoef(ref_seg, enh_seg)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
            
            # SNR component
            noise = enh_seg - ref_seg
            signal_power = np.mean(ref_seg ** 2)
            noise_power = np.mean(noise ** 2)
            
            if noise_power > 0:
                snr = 10 * np.log10(signal_power / noise_power)
            else:
                snr = 50  # Very high SNR
            
            # Map to PESQ-like scale (1-5)
            pesq_est = 1.0 + 4.0 * max(0, min(1, correlation * 0.5 + np.clip(snr / 40, 0, 0.5)))
            return pesq_est
            
        except Exception as e:
            logger.warning(f"PESQ estimation failed: {e}")
            return 2.5
    
    def _estimate_stoi(self, reference: np.ndarray, enhanced: np.ndarray) -> float:
        """Estimate STOI-like score using spectral correlation"""
        try:
            min_len = min(len(reference), len(enhanced))
            ref_seg = reference[:min_len]
            enh_seg = enhanced[:min_len]
            
            # Short-time spectral analysis
            ref_stft = librosa.stft(ref_seg, n_fft=512, hop_length=128)
            enh_stft = librosa.stft(enh_seg, n_fft=512, hop_length=128)
            
            ref_mag = np.abs(ref_stft)
            enh_mag = np.abs(enh_stft)
            
            # Frame-wise correlation
            correlations = []
            for i in range(min(ref_mag.shape[1], enh_mag.shape[1])):
                corr = np.corrcoef(ref_mag[:, i], enh_mag[:, i])[0, 1]
                if not np.isnan(corr):
                    correlations.append(max(0, corr))
            
            stoi_est = np.mean(correlations) if correlations else 0.5
            return stoi_est
            
        except Exception as e:
            logger.warning(f"STOI estimation failed: {e}")
            return 0.5
    
    def calculate_snr_variants(self, reference: np.ndarray, enhanced: np.ndarray) -> Dict[str, float]:
        """Calculate multiple SNR variants"""
        try:
            min_len = min(len(reference), len(enhanced))
            ref_seg = reference[:min_len]
            enh_seg = enhanced[:min_len]
            noise_seg = enh_seg - ref_seg
            
            # Traditional SNR
            signal_power = np.mean(ref_seg ** 2)
            noise_power = np.mean(noise_seg ** 2)
            snr_traditional = 10 * np.log10(signal_power / (noise_power + 1e-12))
            
            # Segmental SNR
            frame_len = 1024
            seg_snrs = []
            for i in range(0, len(ref_seg) - frame_len, frame_len // 2):
                ref_frame = ref_seg[i:i + frame_len]
                noise_frame = noise_seg[i:i + frame_len]
                
                ref_power = np.mean(ref_frame ** 2)
                noise_power_frame = np.mean(noise_frame ** 2)
                
                if ref_power > 1e-12 and noise_power_frame > 1e-12:
                    seg_snr = 10 * np.log10(ref_power / noise_power_frame)
                    seg_snrs.append(np.clip(seg_snr, -20, 50))
            
            snr_segmental = np.mean(seg_snrs) if seg_snrs else snr_traditional
            
            # Frequency-weighted SNR
            ref_spec = np.abs(librosa.stft(ref_seg))
            noise_spec = np.abs(librosa.stft(noise_seg))
            
            # A-weighting approximation
            freqs = librosa.fft_frequencies(sr=self.sample_rate)
            a_weighting = self._approximate_a_weighting(freqs)
            
            weighted_signal_power = np.sum((ref_spec ** 2) * a_weighting[:, np.newaxis])
            weighted_noise_power = np.sum((noise_spec ** 2) * a_weighting[:, np.newaxis])
            
            snr_weighted = 10 * np.log10(weighted_signal_power / (weighted_noise_power + 1e-12))
            
            return {
                'snr_traditional': snr_traditional,
                'snr_segmental': snr_segmental,
                'snr_weighted': snr_weighted,
                'snr_improvement': snr_traditional - self._calculate_input_snr(ref_seg)
            }
            
        except Exception as e:
            logger.warning(f"SNR calculation failed: {e}")
            return {'snr_traditional': 0, 'snr_segmental': 0, 'snr_weighted': 0, 'snr_improvement': 0}
    
    def _approximate_a_weighting(self, freqs: np.ndarray) -> np.ndarray:
        """Approximate A-weighting curve"""
        f = freqs + 1e-12
        
        # A-weighting formula approximation
        c1 = 12194 ** 2
        c2 = 20.6 ** 2
        c3 = 107.7 ** 2
        c4 = 737.9 ** 2
        
        numerator = c1 * (f ** 4)
        denominator = ((f ** 2) + c2) * np.sqrt(((f ** 2) + c3) * ((f ** 2) + c4)) * ((f ** 2) + c1)
        
        a_weighting_db = 20 * np.log10(numerator / denominator) + 2.0
        return 10 ** (a_weighting_db / 20)
    
    def _calculate_input_snr(self, signal: np.ndarray) -> float:
        """Estimate input SNR by analyzing signal characteristics"""
        try:
            # Use energy-based VAD to estimate noise floor
            frame_len = 1024
            frame_energies = []
            
            for i in range(0, len(signal) - frame_len, frame_len // 2):
                frame = signal[i:i + frame_len]
                energy = np.mean(frame ** 2)
                frame_energies.append(energy)
            
            frame_energies = np.array(frame_energies)
            
            # Assume lowest 10% of frames are noise
            noise_threshold = np.percentile(frame_energies, 10)
            signal_threshold = np.percentile(frame_energies, 90)
            
            input_snr = 10 * np.log10(signal_threshold / (noise_threshold + 1e-12))
            return input_snr
            
        except Exception as e:
            logger.warning(f"Input SNR estimation failed: {e}")
            return 10.0

class ContentAnalyzer:
    """Advanced content analysis and classification"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.noise_classifier = None
        self.speech_detector = None
        self._initialize_classifiers()
    
    def _initialize_classifiers(self):
        """Initialize audio content classifiers"""
        # Simple energy-based classifiers for offline use
        self.energy_threshold = 0.01
        self.spectral_centroid_threshold = 2000
        self.zero_crossing_threshold = 0.1
    
    def classify_audio_content(self, audio: np.ndarray) -> Dict[str, Any]:
        """Classify audio content characteristics"""
        try:
            features = self._extract_content_features(audio)
            
            # Speech vs Music detection
            is_speech = self._detect_speech(features)
            is_music = self._detect_music(features)
            
            # Noise type classification
            noise_type = self._classify_noise_type(features)
            
            # Audio quality assessment
            quality_score = self._assess_audio_quality(features)
            
            return {
                'content_type': 'speech' if is_speech else 'music' if is_music else 'unknown',
                'is_speech': is_speech,
                'is_music': is_music,
                'noise_type': noise_type,
                'quality_score': quality_score,
                'features': features
            }
            
        except Exception as e:
            logger.warning(f"Content classification failed: {e}")
            return {'content_type': 'unknown', 'is_speech': False, 'is_music': False, 
                   'noise_type': 'unknown', 'quality_score': 0.5}
    
    def _extract_content_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract comprehensive audio features"""
        try:
            features = {}
            
            # Temporal features
            features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(audio))
            features['energy'] = np.mean(audio ** 2)
            features['rms_energy'] = np.sqrt(features['energy'])
            
            # Spectral features
            stft = librosa.stft(audio)
            magnitude = np.abs(stft)
            
            features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(
                S=magnitude, sr=self.sample_rate))
            features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(
                S=magnitude, sr=self.sample_rate))
            features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(
                S=magnitude, sr=self.sample_rate))
            
            # MFCC features
            mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
            features['mfcc_mean'] = np.mean(mfcc, axis=1).tolist()
            features['mfcc_std'] = np.std(mfcc, axis=1).tolist()
            
            # Harmonic and percussive components
            harmonic, percussive = librosa.effects.hpss(audio)
            features['harmonic_ratio'] = np.mean(harmonic ** 2) / (np.mean(audio ** 2) + 1e-12)
            features['percussive_ratio'] = np.mean(percussive ** 2) / (np.mean(audio ** 2) + 1e-12)
            
            # Pitch and fundamental frequency
            try:
                pitches, magnitudes = librosa.core.piptrack(y=audio, sr=self.sample_rate)
                pitch_values = pitches[magnitudes > 0.1]
                features['mean_pitch'] = np.mean(pitch_values) if len(pitch_values) > 0 else 0
                features['pitch_variance'] = np.var(pitch_values) if len(pitch_values) > 0 else 0
            except:
                features['mean_pitch'] = 0
                features['pitch_variance'] = 0
            
            return features
            
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            return {}
    
    def _detect_speech(self, features: Dict[str, float]) -> bool:
        """Detect if audio contains speech"""
        try:
            # Speech characteristics
            speech_indicators = 0
            
            # Zero crossing rate (speech typically has moderate ZCR)
            if 0.05 < features.get('zero_crossing_rate', 0) < 0.15:
                speech_indicators += 1
            
            # Spectral centroid (speech typically 500-2000 Hz)
            if 500 < features.get('spectral_centroid', 0) < 2000:
                speech_indicators += 1
            
            # Harmonic content (speech is somewhat harmonic)
            if features.get('harmonic_ratio', 0) > 0.3:
                speech_indicators += 1
            
            # Pitch characteristics (speech has moderate pitch variance)
            if features.get('pitch_variance', 0) > 100:
                speech_indicators += 1
            
            return speech_indicators >= 2
            
        except Exception as e:
            logger.warning(f"Speech detection failed: {e}")
            return False
    
    def _detect_music(self, features: Dict[str, float]) -> bool:
        """Detect if audio contains music"""
        try:
            music_indicators = 0
            
            # Music typically has higher spectral centroid
            if features.get('spectral_centroid', 0) > 1500:
                music_indicators += 1
            
            # Music has strong harmonic content
            if features.get('harmonic_ratio', 0) > 0.6:
                music_indicators += 1
            
            # Music often has percussive elements
            if features.get('percussive_ratio', 0) > 0.2:
                music_indicators += 1
            
            # Music typically has more spectral bandwidth
            if features.get('spectral_bandwidth', 0) > 1000:
                music_indicators += 1
            
            return music_indicators >= 2
            
        except Exception as e:
            logger.warning(f"Music detection failed: {e}")
            return False
    
    def _classify_noise_type(self, features: Dict[str, float]) -> str:
        """Classify the type of noise present"""
        try:
            # Simple noise type classification based on spectral characteristics
            spectral_centroid = features.get('spectral_centroid', 1000)
            spectral_bandwidth = features.get('spectral_bandwidth', 1000)
            energy = features.get('energy', 0.01)
            
            if spectral_centroid < 500 and energy > 0.1:
                return 'low_frequency_rumble'
            elif spectral_centroid > 4000 and spectral_bandwidth > 2000:
                return 'broadband_hiss'
            elif 1000 < spectral_centroid < 3000 and energy > 0.05:
                return 'environmental_noise'
            elif spectral_bandwidth < 500:
                return 'tonal_interference'
            else:
                return 'mixed_noise'
                
        except Exception as e:
            logger.warning(f"Noise classification failed: {e}")
            return 'unknown'
    
    def _assess_audio_quality(self, features: Dict[str, float]) -> float:
        """Assess overall audio quality (0-1 scale)"""
        try:
            quality_score = 0.5  # Base score
            
            # Energy level (too low or too high reduces quality)
            energy = features.get('energy', 0.01)
            if 0.01 < energy < 0.5:
                quality_score += 0.2
            elif energy < 0.001 or energy > 0.8:
                quality_score -= 0.2
            
            # Spectral characteristics
            spectral_centroid = features.get('spectral_centroid', 1000)
            if 500 < spectral_centroid < 4000:  # Good range for speech/music
                quality_score += 0.2
            
            # Harmonic content
            harmonic_ratio = features.get('harmonic_ratio', 0)
            if harmonic_ratio > 0.3:  # Good harmonic structure
                quality_score += 0.1
            
            return np.clip(quality_score, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"Quality assessment failed: {e}")
            return 0.5

class AdvancedUNet(nn.Module):
    """Advanced U-Net architecture for audio enhancement"""
    
    def __init__(self, n_fft=2048, channels=[1, 32, 64, 128, 256], dropout=0.1):
        super().__init__()
        self.n_fft = n_fft
        self.channels = channels
        
        # Calculate input dimension
        input_dim = n_fft // 2 + 1
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        in_channels = input_dim
        
        for out_channels in channels[1:]:
            self.encoder_blocks.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2),
                    nn.BatchNorm1d(out_channels),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2),
                    nn.BatchNorm1d(out_channels),
                    nn.GELU(),
                    nn.Dropout(dropout)
                )
            )
            in_channels = out_channels
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(channels[-1], channels[-1] * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels[-1] * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels[-1] * 2, channels[-1], kernel_size=5, padding=2),
            nn.BatchNorm1d(channels[-1]),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        
        decoder_channels = list(reversed(channels))
        for i, out_channels in enumerate(decoder_channels[1:]):
            in_channels = decoder_channels[i]
            
            # Skip connection processing
            self.skip_connections.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=1)
            )
            
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose1d(in_channels + out_channels, out_channels, 
                                     kernel_size=5, padding=2),
                    nn.BatchNorm1d(out_channels),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2),
                    nn.BatchNorm1d(out_channels),
                    nn.GELU(),
                    nn.Dropout(dropout)
                )
            )
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Conv1d(channels[1], input_dim, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x shape: (batch, freq_bins, time_frames)
        
        # Store skip connections
        skip_connections = []
        
        # Encoder
        current = x
        for encoder_block in self.encoder_blocks:
            current = encoder_block(current)
            skip_connections.append(current)
            # Downsample by factor of 2
            if current.size(2) > 2:
                current = F.max_pool1d(current, kernel_size=2)
        
        # Bottleneck
        current = self.bottleneck(current)
        
        # Decoder
        skip_connections.reverse()
        for i, (decoder_block, skip_conv) in enumerate(zip(self.decoder_blocks, self.skip_connections)):
            # Upsample
            current = F.interpolate(current, size=skip_connections[i].size(2), mode='nearest')
            
            # Process skip connection
            skip = skip_conv(skip_connections[i])
            
            # Concatenate skip connection
            current = torch.cat([current, skip], dim=1)
            
            # Decoder block
            current = decoder_block(current)
        
        # Output
        mask = self.output_layer(current)
        
        # Ensure output matches input size
        if mask.size(2) != x.size(2):
            mask = F.interpolate(mask, size=x.size(2), mode='nearest')
        
        return mask

class ConformerEnhancer(nn.Module):
    """Conformer network for advanced speech enhancement"""
    
    def __init__(self, input_dim=1025, model_dim=256, num_layers=6, 
                 num_heads=8, conv_kernel_size=15, dropout=0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.model_dim = model_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, model_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 2000, model_dim))
        
        # Conformer blocks
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(
                dim=model_dim,
                num_heads=num_heads,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x shape: (batch, time, freq)
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        pos_enc = self.pos_encoding[:, :seq_len, :].expand(batch_size, -1, -1)
        x = x + pos_enc
        
        # Apply Conformer blocks
        for conformer_block in self.conformer_blocks:
            x = conformer_block(x)
        
        # Output projection
        mask = self.output_projection(x)
        
        return mask

class ConformerBlock(nn.Module):
    """Single Conformer block"""
    
    def __init__(self, dim=256, num_heads=8, conv_kernel_size=15, dropout=0.1):
        super().__init__()
        
        # Multi-head self-attention
        self.mhsa = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.mhsa_norm = nn.LayerNorm(dim)
        
        # Convolution module
        self.conv_module = ConvolutionModule(dim, conv_kernel_size, dropout)
        self.conv_norm = nn.LayerNorm(dim)
        
        # Feed-forward modules
        self.ff1 = FeedForwardModule(dim, dim * 4, dropout)
        self.ff1_norm = nn.LayerNorm(dim)
        
        self.ff2 = FeedForwardModule(dim, dim * 4, dropout)
        self.ff2_norm = nn.LayerNorm(dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Feed-forward 1 (half step)
        x = x + 0.5 * self.dropout(self.ff1(self.ff1_norm(x)))
        
        # Multi-head self-attention
        attn_out, _ = self.mhsa(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.mhsa_norm(x)
        
        # Convolution module
        x = x + self.dropout(self.conv_module(self.conv_norm(x)))
        
        # Feed-forward 2 (half step)
        x = x + 0.5 * self.dropout(self.ff2(self.ff2_norm(x)))
        
        return x

class ConvolutionModule(nn.Module):
    """Convolution module for Conformer"""
    
    def __init__(self, dim=256, kernel_size=15, dropout=0.1):
        super().__init__()
        
        self.pointwise_conv1 = nn.Conv1d(dim, dim * 2, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, 
            padding=(kernel_size - 1) // 2, groups=dim
        )
        self.batch_norm = nn.BatchNorm1d(dim)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: (batch, seq_len, dim)
        x = x.transpose(1, 2)  # (batch, dim, seq_len)
        
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        x = x.transpose(1, 2)  # (batch, seq_len, dim)
        return x

class FeedForwardModule(nn.Module):
    """Feed-forward module for Conformer"""
    
    def __init__(self, dim=256, hidden_dim=1024, dropout=0.1):
        super().__init__()
        
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.activation = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x

class ModelEnsemble:
    """Advanced model ensemble for professional enhancement"""
    
    def __init__(self, config: ProfessionalEnhancementConfig, device: torch.device):
        self.config = config
        self.device = device
        self.models = {}
        self.model_weights = config.ensemble_weights
        self.performance_history = defaultdict(list)
        
    def initialize_models(self):
        """Initialize ensemble of models"""
        try:
            # U-Net model
            self.models['unet'] = AdvancedUNet(
                n_fft=self.config.n_fft,
                channels=[1, 64, 128, 256, 512]  # Larger for RTX A4000
            ).to(self.device)
            
            # Conformer model
            input_dim = self.config.n_fft // 2 + 1
            self.models['conformer'] = ConformerEnhancer(
                input_dim=input_dim,
                model_dim=512,  # Larger for RTX A4000
                num_layers=8,   # More layers for RTX A4000
                num_heads=16    # More heads for RTX A4000
            ).to(self.device)
            
            # Enhanced LSTM VAD
            self.models['vad'] = EnhancedLSTMVAD(
                input_dim=40,
                hidden_dim=256,  # Larger for RTX A4000
                num_layers=4     # More layers
            ).to(self.device)
            
            # Enhanced Transformer
            self.models['transformer'] = EnhancedTransformer(
                n_fft=self.config.n_fft,
                d_model=512,     # Larger for RTX A4000
                nhead=16,        # More heads
                num_layers=8,    # More layers
                dim_feedforward=2048  # Larger feedforward
            ).to(self.device)
            
            # Enhanced Autoencoder
            self.models['autoencoder'] = EnhancedAutoencoder().to(self.device)
            
            # Set all models to eval mode
            for model in self.models.values():
                model.eval()
            
            # Try to compile models for optimization
            self._compile_models()
            
            # Load pre-trained weights if available
            self._load_pretrained_ensemble()
            
            logger.info(f"Initialized {len(self.models)} models in ensemble")
            
        except Exception as e:
            logger.error(f"Failed to initialize model ensemble: {e}")
            raise
    
    def _compile_models(self):
        """Compile models for optimization"""
        try:
            if hasattr(torch, 'compile'):
                for name, model in self.models.items():
                    try:
                        self.models[name] = torch.compile(model, mode="max-autotune")
                        logger.info(f"Compiled {name} model")
                    except Exception as e:
                        logger.warning(f"Failed to compile {name}: {e}")
        except Exception as e:
            logger.warning(f"Model compilation not available: {e}")
    
    def _load_pretrained_ensemble(self):
        """Load pre-trained weights for ensemble"""
        models_loaded = 0
        
        for model_name in self.models.keys():
            model_path = MODEL_BASE_PATH / f"{model_name}_professional.pth"
            
            if model_path.exists():
                try:
                    state_dict = torch.load(model_path, map_location=self.device)
                    self.models[model_name].load_state_dict(state_dict)
                    models_loaded += 1
                    logger.info(f"Loaded pre-trained {model_name}")
                except Exception as e:
                    logger.warning(f"Failed to load {model_name}: {e}")
        
        if models_loaded == 0:
            logger.warning("No pre-trained models loaded - using random weights")
        else:
            logger.info(f"Loaded {models_loaded}/{len(self.models)} pre-trained models")
    
    def enhance_with_ensemble(self, audio_tensor: torch.Tensor, 
                            stream_id: int = 0) -> torch.Tensor:
        """Enhance audio using model ensemble"""
        try:
            with torch.no_grad():
                # Compute STFT once for all models
                stft = torch.stft(
                    audio_tensor,
                    n_fft=self.config.n_fft,
                    hop_length=self.config.hop_length,
                    window=torch.hann_window(self.config.n_fft).to(self.device),
                    return_complex=True
                )
                
                magnitude = torch.abs(stft)
                phase = torch.angle(stft)
                
                # Normalize magnitude
                magnitude_norm = magnitude / (torch.max(magnitude) + 1e-8)
                
                # Get enhancements from each model
                enhancements = {}
                
                # U-Net enhancement
                if 'unet' in self.models:
                    unet_input = magnitude_norm.unsqueeze(0)  # Add batch dim
                    unet_mask = self.models['unet'](unet_input).squeeze(0)
                    enhancements['unet'] = unet_mask * magnitude
                
                # Conformer enhancement
                if 'conformer' in self.models:
                    conformer_input = magnitude_norm.permute(2, 0, 1).unsqueeze(0)  # (batch, time, freq)
                    conformer_mask = self.models['conformer'](conformer_input)
                    conformer_mask = conformer_mask.squeeze(0).permute(1, 2, 0)  # Back to (freq, time)
                    enhancements['conformer'] = conformer_mask * magnitude
                
                # Transformer enhancement (using existing logic)
                if 'transformer' in self.models:
                    transformer_input = magnitude_norm.permute(1, 0).unsqueeze(0)
                    try:
                        transformer_mask = self.models['transformer'](transformer_input)
                        transformer_mask = transformer_mask.squeeze(0).permute(1, 0)
                        enhancements['transformer'] = transformer_mask * magnitude
                    except Exception as e:
                        logger.warning(f"Transformer enhancement failed: {e}")
                        enhancements['transformer'] = magnitude
                
                # Weighted ensemble combination
                if enhancements:
                    enhanced_magnitude = self._combine_enhancements(enhancements)
                else:
                    enhanced_magnitude = magnitude
                
                # Reconstruct audio
                enhanced_stft = enhanced_magnitude * torch.exp(1j * phase)
                enhanced_audio = torch.istft(
                    enhanced_stft,
                    n_fft=self.config.n_fft,
                    hop_length=self.config.hop_length,
                    window=torch.hann_window(self.config.n_fft).to(self.device)
                )
                
                return enhanced_audio
                
        except Exception as e:
            logger.error(f"Ensemble enhancement failed: {e}")
            return audio_tensor
    
    def _combine_enhancements(self, enhancements: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Combine multiple enhancements using weighted voting"""
        try:
            if not enhancements:
                raise ValueError("No enhancements to combine")
            
            # Ensure we have weights for all models
            model_names = list(enhancements.keys())
            if len(self.model_weights) < len(model_names):
                # Extend weights with equal distribution
                remaining_weight = 1.0 - sum(self.model_weights)
                additional_weights = [remaining_weight / (len(model_names) - len(self.model_weights))] * \
                                   (len(model_names) - len(self.model_weights))
                weights = self.model_weights + additional_weights
            else:
                weights = self.model_weights[:len(model_names)]
            
            # Normalize weights
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            
            # Weighted combination
            combined = torch.zeros_like(next(iter(enhancements.values())))
            
            for i, (model_name, enhancement) in enumerate(enhancements.items()):
                combined += weights[i] * enhancement
            
            return combined
            
        except Exception as e:
            logger.error(f"Enhancement combination failed: {e}")
            # Return average of all enhancements
            return torch.mean(torch.stack(list(enhancements.values())), dim=0)

class EnhancedLSTMVAD(nn.Module):
    """Enhanced LSTM Voice Activity Detection for RTX A4000"""
    
    def __init__(self, input_dim=40, hidden_dim=256, num_layers=4):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, dropout=0.2
        )
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            hidden_dim * 2, num_heads=16, batch_first=True
        )
        
        # Enhanced classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Classification
        output = self.classifier(attn_out)
        
        return output

class EnhancedTransformer(nn.Module):
    """Enhanced Transformer for RTX A4000"""
    
    def __init__(self, n_fft=2048, d_model=512, nhead=16, num_layers=8, dim_feedforward=2048):
        super().__init__()
        self.d_model = d_model
        
        input_dim = n_fft // 2 + 1
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 2000, d_model))
        
        # Enhanced transformer with pre-normalization
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Enhanced output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 4, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        pos_enc = self.pos_encoding[:, :seq_len, :].expand(batch_size, -1, -1)
        x = x + pos_enc
        
        # Transformer processing
        x = self.transformer(x)
        
        # Output projection
        mask = self.output_projection(x)
        
        return mask

class EnhancedAutoencoder(nn.Module):
    """Enhanced Autoencoder for RTX A4000"""
    
    def __init__(self):
        super().__init__()
        
        # Enhanced encoder with residual connections
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Conv1d(256, 512, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(512),
            nn.GELU(),
        )
        
        # Enhanced decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(512, 256, kernel_size=15, stride=2, padding=7, output_padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.ConvTranspose1d(256, 128, kernel_size=15, stride=2, padding=7, output_padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.ConvTranspose1d(128, 64, kernel_size=15, stride=2, padding=7, output_padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.ConvTranspose1d(64, 1, kernel_size=15, stride=1, padding=7),
            nn.Tanh()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class ProfessionalAudioProcessor:
    """Advanced professional audio processing"""
    
    def __init__(self, config: ProfessionalEnhancementConfig, device: torch.device):
        self.config = config
        self.device = device
        self.noise_profiles = {}
        self._load_noise_profiles()
    
    def _load_noise_profiles(self):
        """Load noise profiles for targeted noise reduction"""
        try:
            if NOISE_PROFILES_PATH.exists():
                for noise_file in NOISE_PROFILES_PATH.glob("*.wav"):
                    noise_audio, sr = librosa.load(noise_file, sr=self.config.sample_rate)
                    noise_spectrum = np.abs(librosa.stft(noise_audio))
                    self.noise_profiles[noise_file.stem] = np.mean(noise_spectrum, axis=1)
                logger.info(f"Loaded {len(self.noise_profiles)} noise profiles")
        except Exception as e:
            logger.warning(f"Failed to load noise profiles: {e}")
    
    def multi_band_enhancement(self, audio: np.ndarray) -> np.ndarray:
        """Multi-band audio enhancement"""
        try:
            # Define frequency bands
            bands = [
                (0, 250),      # Low
                (250, 1000),   # Low-mid
                (1000, 4000),  # Mid (speech)
                (4000, 8000),  # High-mid
                (8000, self.config.sample_rate // 2)  # High
            ]
            
            # Split into bands
            band_signals = []
            for low_freq, high_freq in bands:
                # Design bandpass filter
                nyquist = self.config.sample_rate / 2
                low = low_freq / nyquist
                high = min(high_freq / nyquist, 0.99)
                
                if low >= high:
                    band_signals.append(np.zeros_like(audio))
                    continue
                
                if low <= 0:
                    # Low-pass filter
                    sos = signal.butter(4, high, 'low', output='sos')
                elif high >= 0.99:
                    # High-pass filter
                    sos = signal.butter(4, low, 'high', output='sos')
                else:
                    # Band-pass filter
                    sos = signal.butter(4, [low, high], 'band', output='sos')
                
                band_signal = signal.sosfilt(sos, audio)
                band_signals.append(band_signal)
            
            # Process each band individually
            processed_bands = []
            for i, band_signal in enumerate(band_signals):
                if np.max(np.abs(band_signal)) > 1e-6:
                    # Apply band-specific processing
                    if i == 2:  # Speech band - enhance more aggressively
                        processed_band = self._enhance_speech_band(band_signal)
                    elif i == 0:  # Low band - reduce rumble
                        processed_band = self._reduce_low_frequency_noise(band_signal)
                    elif i == 4:  # High band - reduce hiss
                        processed_band = self._reduce_high_frequency_noise(band_signal)
                    else:
                        processed_band = self._general_band_enhancement(band_signal)
                else:
                    processed_band = band_signal
                
                processed_bands.append(processed_band)
            
            # Recombine bands
            enhanced_audio = np.sum(processed_bands, axis=0)
            return enhanced_audio
            
        except Exception as e:
            logger.error(f"Multi-band enhancement failed: {e}")
            return audio
    
    def _enhance_speech_band(self, band_signal: np.ndarray) -> np.ndarray:
        """Enhance speech-critical frequency band"""
        try:
            # Spectral subtraction with speech-optimized parameters
            stft = librosa.stft(band_signal, n_fft=self.config.n_fft, 
                              hop_length=self.config.hop_length)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Estimate noise floor (first 10% of signal)
            noise_frames = max(1, magnitude.shape[1] // 10)
            noise_spectrum = np.median(magnitude[:, :noise_frames], axis=1, keepdims=True)
            
            # Speech-optimized spectral subtraction
            alpha = 2.5  # Higher over-subtraction for speech clarity
            enhanced_magnitude = magnitude - alpha * noise_spectrum
            
            # Softer spectral floor for speech
            floor_magnitude = 0.05 * magnitude
            enhanced_magnitude = np.maximum(enhanced_magnitude, floor_magnitude)
            
            # Reconstruct
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_band = librosa.istft(enhanced_stft, hop_length=self.config.hop_length)
            
            return enhanced_band
            
        except Exception as e:
            logger.warning(f"Speech band enhancement failed: {e}")
            return band_signal
    
    def _reduce_low_frequency_noise(self, band_signal: np.ndarray) -> np.ndarray:
        """Reduce low-frequency rumble and noise"""
        try:
            # Aggressive high-pass filtering for low band
            sos = signal.butter(4, 60 / (self.config.sample_rate / 2), 'high', output='sos')
            filtered_signal = signal.sosfilt(sos, band_signal)
            
            # Gentle compression to reduce rumble
            threshold = 0.1
            ratio = 4.0
            filtered_signal = self._apply_compression(filtered_signal, threshold, ratio)
            
            return filtered_signal
            
        except Exception as e:
            logger.warning(f"Low frequency noise reduction failed: {e}")
            return band_signal
    
    def _reduce_high_frequency_noise(self, band_signal: np.ndarray) -> np.ndarray:
        """Reduce high-frequency hiss and artifacts"""
        try:
            # Spectral gating for high frequencies
            stft = librosa.stft(band_signal, n_fft=512, hop_length=128)
            magnitude = np.abs(stft)
            
            # Adaptive threshold based on signal energy
            energy_threshold = np.percentile(magnitude, 30)  # 30th percentile
            gate_mask = magnitude > energy_threshold
            
            # Apply gating
            gated_magnitude = magnitude * gate_mask
            
            # Smooth transitions
            gated_magnitude = signal.medfilt(gated_magnitude, kernel_size=(1, 3))
            
            # Reconstruct
            phase = np.angle(stft)
            gated_stft = gated_magnitude * np.exp(1j * phase)
            enhanced_band = librosa.istft(gated_stft, hop_length=128)
            
            return enhanced_band
            
        except Exception as e:
            logger.warning(f"High frequency noise reduction failed: {e}")
            return band_signal
    
    def _general_band_enhancement(self, band_signal: np.ndarray) -> np.ndarray:
        """General enhancement for mid-frequency bands"""
        try:
            # Mild spectral subtraction
            stft = librosa.stft(band_signal, n_fft=self.config.n_fft, 
                              hop_length=self.config.hop_length)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Conservative noise estimation
            noise_frames = max(1, magnitude.shape[1] // 20)
            noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
            
            # Mild over-subtraction
            alpha = 1.5
            enhanced_magnitude = magnitude - alpha * noise_spectrum
            
            # Conservative spectral floor
            floor_magnitude = 0.1 * magnitude
            enhanced_magnitude = np.maximum(enhanced_magnitude, floor_magnitude)
            
            # Reconstruct
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_band = librosa.istft(enhanced_stft, hop_length=self.config.hop_length)
            
            return enhanced_band
            
        except Exception as e:
            logger.warning(f"General band enhancement failed: {e}")
            return band_signal
    
    def _apply_compression(self, signal: np.ndarray, threshold: float, ratio: float) -> np.ndarray:
        """Apply dynamic range compression"""
        try:
            # Convert to dB
            signal_db = 20 * np.log10(np.abs(signal) + 1e-12)
            
            # Apply compression above threshold
            compressed_db = np.where(
                signal_db > 20 * np.log10(threshold),
                20 * np.log10(threshold) + (signal_db - 20 * np.log10(threshold)) / ratio,
                signal_db
            )
            
            # Convert back to linear
            compressed_signal = np.sign(signal) * (10 ** (compressed_db / 20))
            
            return compressed_signal
            
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            return signal
    
    def harmonic_percussive_enhancement(self, audio: np.ndarray) -> np.ndarray:
        """Enhance harmonic and percussive components separately"""
        try:
            # Separate harmonic and percussive components
            harmonic, percussive = librosa.effects.hpss(audio, margin=2.0)
            
            # Enhance harmonic component (speech/tonal content)
            enhanced_harmonic = self._enhance_harmonic_component(harmonic)
            
            # Process percussive component (transients, clicks)
            processed_percussive = self._process_percussive_component(percussive)
            
            # Combine with emphasis on harmonic content
            combined = 0.8 * enhanced_harmonic + 0.2 * processed_percussive
            
            return combined
            
        except Exception as e:
            logger.error(f"Harmonic-percussive enhancement failed: {e}")
            return audio
    
    def _enhance_harmonic_component(self, harmonic: np.ndarray) -> np.ndarray:
        """Enhance harmonic component"""
        try:
            # Spectral enhancement for harmonic content
            stft = librosa.stft(harmonic, n_fft=self.config.n_fft, 
                              hop_length=self.config.hop_length)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Enhance harmonic peaks
            # Find spectral peaks
            peak_mask = magnitude > np.roll(magnitude, 1, axis=0)
            peak_mask &= magnitude > np.roll(magnitude, -1, axis=0)
            
            # Boost harmonic peaks
            enhanced_magnitude = magnitude.copy()
            enhanced_magnitude[peak_mask] *= 1.2
            
            # Reconstruct
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_harmonic = librosa.istft(enhanced_stft, hop_length=self.config.hop_length)
            
            return enhanced_harmonic
            
        except Exception as e:
            logger.warning(f"Harmonic enhancement failed: {e}")
            return harmonic
    
    def _process_percussive_component(self, percussive: np.ndarray) -> np.ndarray:
        """Process percussive component to reduce artifacts"""
        try:
            # Gate percussive content to remove low-level artifacts
            threshold = np.percentile(np.abs(percussive), 95)  # Keep only strongest transients
            gate_mask = np.abs(percussive) > threshold * 0.1
            
            # Apply gating with smooth transitions
            gated_percussive = percussive * gate_mask
            
            # Smooth the gating to avoid artifacts
            gate_smooth = signal.medfilt(gate_mask.astype(float), kernel_size=5)
            processed_percussive = percussive * gate_smooth
            
            return processed_percussive
            
        except Exception as e:
            logger.warning(f"Percussive processing failed: {e}")
            return percussive
    
    def phase_recovery(self, audio: np.ndarray) -> np.ndarray:
        """Advanced phase recovery and coherence enhancement"""
        try:
            # Compute STFT
            stft = librosa.stft(audio, n_fft=self.config.n_fft, 
                              hop_length=self.config.hop_length)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Phase unwrapping and smoothing
            unwrapped_phase = np.unwrap(phase, axis=1)
            
            # Smooth phase across time
            smoothed_phase = signal.savgol_filter(unwrapped_phase, 
                                                window_length=min(5, unwrapped_phase.shape[1]),
                                                polyorder=2, axis=1)
            
            # Reconstruct with improved phase
            improved_stft = magnitude * np.exp(1j * smoothed_phase)
            improved_audio = librosa.istft(improved_stft, hop_length=self.config.hop_length)
            
            return improved_audio
            
        except Exception as e:
            logger.error(f"Phase recovery failed: {e}")
            return audio
    
    def psychoacoustic_enhancement(self, audio: np.ndarray) -> np.ndarray:
        """Apply psychoacoustic masking principles"""
        try:
            # Convert to frequency domain
            stft = librosa.stft(audio, n_fft=self.config.n_fft, 
                              hop_length=self.config.hop_length)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Convert to Bark scale for perceptual processing
            bark_spectrum = self._convert_to_bark_scale(magnitude)
            
            # Apply psychoacoustic masking
            masked_spectrum = self._apply_psychoacoustic_masking(bark_spectrum)
            
            # Convert back to linear frequency scale
            enhanced_magnitude = self._convert_from_bark_scale(masked_spectrum, magnitude.shape)
            
            # Reconstruct audio
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft, hop_length=self.config.hop_length)
            
            return enhanced_audio
            
        except Exception as e:
            logger.error(f"Psychoacoustic enhancement failed: {e}")
            return audio
    
    def _convert_to_bark_scale(self, magnitude: np.ndarray) -> np.ndarray:
        """Convert linear frequency scale to Bark scale"""
        try:
            # Simple Bark scale conversion
            freqs = librosa.fft_frequencies(sr=self.config.sample_rate, n_fft=self.config.n_fft)
            bark_freqs = 13 * np.arctan(0.00076 * freqs) + 3.5 * np.arctan((freqs / 7500) ** 2)
            
            # Group frequencies into Bark bands
            num_bark_bands = 24
            bark_bands = np.linspace(0, np.max(bark_freqs), num_bark_bands)
            
            bark_spectrum = np.zeros((num_bark_bands, magnitude.shape[1]))
            
            for i in range(num_bark_bands - 1):
                band_mask = (bark_freqs >= bark_bands[i]) & (bark_freqs < bark_bands[i + 1])
                if np.any(band_mask):
                    bark_spectrum[i, :] = np.mean(magnitude[band_mask, :], axis=0)
            
            return bark_spectrum
            
        except Exception as e:
            logger.warning(f"Bark scale conversion failed: {e}")
            return magnitude
    
    def _apply_psychoacoustic_masking(self, bark_spectrum: np.ndarray) -> np.ndarray:
        """Apply psychoacoustic masking rules"""
        try:
            masked_spectrum = bark_spectrum.copy()
            
            # Apply frequency masking
            for i in range(bark_spectrum.shape[0]):
                for j in range(bark_spectrum.shape[1]):
                    current_level = bark_spectrum[i, j]
                    
                    # Check neighboring bands for masking
                    masking_threshold = 0
                    for k in range(max(0, i-2), min(bark_spectrum.shape[0], i+3)):
                        if k != i:
                            neighbor_level = bark_spectrum[k, j]
                            # Simple masking curve approximation
                            masking_contribution = neighbor_level * 0.1 * np.exp(-abs(i-k))
                            masking_threshold += masking_contribution
                    
                    # Apply masking
                    if current_level < masking_threshold:
                        masked_spectrum[i, j] *= 0.5  # Reduce masked components
            
            return masked_spectrum
            
        except Exception as e:
            logger.warning(f"Psychoacoustic masking failed: {e}")
            return bark_spectrum
    
    def _convert_from_bark_scale(self, bark_spectrum: np.ndarray, 
                                target_shape: Tuple[int, int]) -> np.ndarray:
        """Convert Bark scale back to linear frequency scale"""
        try:
            # Simple upsampling back to linear scale
            enhanced_magnitude = np.zeros(target_shape)
            
            freqs = librosa.fft_frequencies(sr=self.config.sample_rate, n_fft=self.config.n_fft)
            bark_freqs = 13 * np.arctan(0.00076 * freqs) + 3.5 * np.arctan((freqs / 7500) ** 2)
            
            bark_bands = np.linspace(0, np.max(bark_freqs), bark_spectrum.shape[0])
            
            for i in range(len(freqs)):
                # Find closest Bark band
                bark_idx = np.argmin(np.abs(bark_bands - bark_freqs[i]))
                enhanced_magnitude[i, :] = bark_spectrum[bark_idx, :]
            
            return enhanced_magnitude
            
        except Exception as e:
            logger.warning(f"Bark scale back-conversion failed: {e}")
            return np.ones(target_shape) * np.mean(bark_spectrum)

class ProfessionalCUDAAudioEnhancer:
    """Main professional CUDA audio enhancement system"""
    
    def __init__(self, config: Optional[ProfessionalEnhancementConfig] = None):
        self.config = config or ProfessionalEnhancementConfig()
        
        # Setup RTX A4000 environment
        self.device = setup_rtx_a4000_environment()
        self.memory_manager = RTXMemoryManager(self.device)
        
        # Initialize mixed precision scaler
        self.scaler = GradScaler() if self.config.mixed_precision else None
        
        # Initialize components
        self.quality_metrics = ProfessionalQualityMetrics(self.config.sample_rate)
        self.content_analyzer = ContentAnalyzer(self.config.sample_rate)
        self.audio_processor = ProfessionalAudioProcessor(self.config, self.device)
        self.model_ensemble = ModelEnsemble(self.config, self.device)
        
        # Performance tracking
        self.metrics = defaultdict(list)
        self.processing_history = []
        
        # Initialize models
        self._initialize_professional_system()
        
        logger.info("Professional RTX A4000 Audio Enhancer initialized successfully")
    
    def _initialize_professional_system(self):
        """Initialize all professional components"""
        with self.memory_manager.memory_context("system_initialization", 2.0):
            try:
                # Initialize model ensemble
                self.model_ensemble.initialize_models()
                
                # Create memory pools for different operations
                self.memory_manager.create_memory_pool("enhancement", 4096)  # 4GB pool
                self.memory_manager.create_memory_pool("analysis", 1024)     # 1GB pool
                
                logger.info("Professional system initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize professional system: {e}")
                raise
    
    def enhance_audio_professional(self, audio_path: str = None) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        """Main professional enhancement pipeline"""
        if audio_path is None:
            audio_path = INPUT_AUDIO_PATH
            logger.info(f"Using global input path: {INPUT_AUDIO_PATH}")
        
        logger.info(f"Starting professional enhancement pipeline for: {audio_path}")
        
        # Load and validate audio
        start_time = time.time()
        original_audio, sr = self.load_and_validate_audio(audio_path)
        
        # Analyze content
        content_analysis = self.content_analyzer.classify_audio_content(original_audio)
        logger.info(f"Content analysis: {content_analysis['content_type']}, "
                   f"Quality: {content_analysis['quality_score']:.2f}")
        
        # Adaptive configuration based on content
        self._adapt_config_to_content(content_analysis)
        
        # Multi-stage enhancement pipeline
        enhanced_audio = original_audio.copy()
        enhancement_stages = []
        
        # Stage 1: Professional multi-band processing
        logger.info("Stage 1: Multi-band enhancement...")
        stage_start = time.time()
        enhanced_audio = self.audio_processor.multi_band_enhancement(enhanced_audio)
        stage_time = time.time() - stage_start
        
        stage_quality = self._evaluate_enhancement_quality(original_audio, enhanced_audio)
        enhancement_stages.append({
            'stage': 'multi_band_enhancement',
            'time': stage_time,
            'quality_improvement': stage_quality['improvement'],
            'applied': stage_quality['improvement'] > self.config.quality_threshold
        })
        
        if stage_quality['improvement'] <= self.config.quality_threshold:
            enhanced_audio = original_audio.copy()
            logger.warning("Multi-band enhancement degraded quality, reverting")
        
        # Stage 2: Harmonic-percussive enhancement
        logger.info("Stage 2: Harmonic-percussive enhancement...")
        stage_start = time.time()
        hp_enhanced = self.audio_processor.harmonic_percussive_enhancement(enhanced_audio)
        stage_time = time.time() - stage_start
        
        stage_quality = self._evaluate_enhancement_quality(enhanced_audio, hp_enhanced)
        enhancement_stages.append({
            'stage': 'harmonic_percussive_enhancement',
            'time': stage_time,
            'quality_improvement': stage_quality['improvement'],
            'applied': stage_quality['improvement'] > self.config.quality_threshold
        })
        
        if stage_quality['improvement'] > self.config.quality_threshold:
            enhanced_audio = hp_enhanced
        else:
            logger.info("Harmonic-percussive enhancement provided minimal improvement")
        
        # Stage 3: Phase recovery
        if self.config.phase_recovery:
            logger.info("Stage 3: Phase recovery...")
            stage_start = time.time()
            phase_enhanced = self.audio_processor.phase_recovery(enhanced_audio)
            stage_time = time.time() - stage_start
            
            stage_quality = self._evaluate_enhancement_quality(enhanced_audio, phase_enhanced)
            enhancement_stages.append({
                'stage': 'phase_recovery',
                'time': stage_time,
                'quality_improvement': stage_quality['improvement'],
                'applied': stage_quality['improvement'] > -0.1  # More lenient for phase
            })
            
            if stage_quality['improvement'] > -0.1:
                enhanced_audio = phase_enhanced
        
        # Stage 4: Psychoacoustic enhancement
        if self.config.psychoacoustic_modeling:
            logger.info("Stage 4: Psychoacoustic enhancement...")
            stage_start = time.time()
            psycho_enhanced = self.audio_processor.psychoacoustic_enhancement(enhanced_audio)
            stage_time = time.time() - stage_start
            
            stage_quality = self._evaluate_enhancement_quality(enhanced_audio, psycho_enhanced)
            enhancement_stages.append({
                'stage': 'psychoacoustic_enhancement',
                'time': stage_time,
                'quality_improvement': stage_quality['improvement'],
                'applied': stage_quality['improvement'] > self.config.quality_threshold
            })
            
            if stage_quality['improvement'] > self.config.quality_threshold:
                enhanced_audio = psycho_enhanced
        
        # Stage 5: Model ensemble enhancement
        logger.info("Stage 5: AI model ensemble enhancement...")
        stage_start = time.time()
        
        with self.memory_manager.memory_context("ensemble_enhancement", 6.0):
            # Convert to tensor and enhance
            audio_tensor = torch.FloatTensor(enhanced_audio).to(self.device)
            ensemble_enhanced_tensor = self.model_ensemble.enhance_with_ensemble(audio_tensor)
            ensemble_enhanced = ensemble_enhanced_tensor.cpu().numpy()
        
        stage_time = time.time() - stage_start
        
        stage_quality = self._evaluate_enhancement_quality(enhanced_audio, ensemble_enhanced)
        enhancement_stages.append({
            'stage': 'ensemble_enhancement',
            'time': stage_time,
            'quality_improvement': stage_quality['improvement'],
            'applied': stage_quality['improvement'] > self.config.quality_threshold
        })
        
        if stage_quality['improvement'] > self.config.quality_threshold:
            enhanced_audio = ensemble_enhanced
        else:
            logger.info("Ensemble enhancement provided minimal improvement")
        
        # Stage 6: Advanced noise reduction (if available)
        if NOISEREDUCE_AVAILABLE:
            logger.info("Stage 6: Advanced noise reduction...")
            stage_start = time.time()
            try:
                nr_enhanced = nr.reduce_noise(
                    y=enhanced_audio,
                    sr=sr,
                    stationary=False,
                    prop_decrease=0.8
                )
                stage_time = time.time() - stage_start
                
                stage_quality = self._evaluate_enhancement_quality(enhanced_audio, nr_enhanced)
                enhancement_stages.append({
                    'stage': 'advanced_noise_reduction',
                    'time': stage_time,
                    'quality_improvement': stage_quality['improvement'],
                    'applied': stage_quality['improvement'] > self.config.quality_threshold
                })
                
                if stage_quality['improvement'] > self.config.quality_threshold:
                    enhanced_audio = nr_enhanced
                    
            except Exception as e:
                logger.warning(f"Advanced noise reduction failed: {e}")
                enhancement_stages.append({
                    'stage': 'advanced_noise_reduction',
                    'time': 0,
                    'quality_improvement': 0,
                    'applied': False
                })
        
        # Stage 7: Final professional processing
        logger.info("Stage 7: Final professional processing...")
        stage_start = time.time()
        final_audio = self._apply_final_professional_processing(enhanced_audio, sr)
        stage_time = time.time() - stage_start
        
        enhancement_stages.append({
            'stage': 'final_processing',
            'time': stage_time,
            'quality_improvement': 0,  # Always applied
            'applied': True
        })
        
        # Calculate comprehensive quality metrics
        total_time = time.time() - start_time
        quality_assessment = self._calculate_comprehensive_quality_metrics(
            original_audio, final_audio
        )
        
        # Prepare detailed results
        results = {
            'processing_time': total_time,
            'content_analysis': content_analysis,
            'enhancement_stages': enhancement_stages,
            'quality_metrics': quality_assessment,
            'gpu_memory_info': self.memory_manager.get_detailed_memory_info(),
            'configuration_used': asdict(self.config)
        }
        
        # Store processing history
        self.processing_history.append({
            'timestamp': time.time(),
            'input_file': audio_path,
            'results': results
        })
        
        # Log comprehensive results
        self._log_comprehensive_results(results)
        
        return final_audio, sr, results
    
    def load_and_validate_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load and comprehensively validate audio file"""
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        try:
            start_time = time.time()
            
            # Load audio with librosa
            audio, sr = librosa.load(file_path, sr=None, mono=True)
            
            # Comprehensive validation
            if len(audio) == 0:
                raise ValueError("Empty audio file")
            
            if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
                logger.warning("Audio contains NaN or Inf values, cleaning...")
                audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Check dynamic range
            dynamic_range = np.max(audio) - np.min(audio)
            if dynamic_range < 1e-6:
                logger.warning("Audio has very low dynamic range")
            
            # Resample if necessary
            if sr != self.config.sample_rate:
                logger.info(f"Resampling from {sr}Hz to {self.config.sample_rate}Hz")
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.config.sample_rate)
                sr = self.config.sample_rate
            
            # Normalize with headroom
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                # Keep some headroom to prevent clipping during processing
                audio = audio / max_val * 0.9
            
            load_time = time.time() - start_time
            
            logger.info(f"Audio loaded and validated: {len(audio)/sr:.2f}s at {sr}Hz ({load_time:.3f}s)")
            logger.info(f"Dynamic range: {20*np.log10(dynamic_range):.1f}dB")
            
            return audio, sr
            
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            raise
    
    def _adapt_config_to_content(self, content_analysis: Dict[str, Any]):
        """Adapt configuration based on content analysis"""
        try:
            content_type = content_analysis['content_type']
            noise_type = content_analysis['noise_type']
            quality_score = content_analysis['quality_score']
            
            # Adapt based on content type
            if content_type == 'speech':
                # Optimize for speech
                self.config.wiener_alpha = 0.96
                self.config.vad_threshold = 0.5
                self.config.target_loudness = -20.0  # Higher for speech
                logger.info("Configuration adapted for speech content")
                
            elif content_type == 'music':
                # Optimize for music
                self.config.wiener_alpha = 0.92
                self.config.vad_threshold = 0.7
                self.config.target_loudness = -23.0  # Standard for music
                logger.info("Configuration adapted for music content")
            
            # Adapt based on noise type
            if noise_type == 'broadband_hiss':
                self.config.spectral_floor = 0.002  # Higher floor for hiss
            elif noise_type == 'low_frequency_rumble':
                self.config.noise_gate_threshold = -30  # More aggressive gating
            
            # Adapt based on quality
            if quality_score < 0.3:
                # Very poor quality - more aggressive enhancement
                self.config.quality_threshold = 0.05  # Lower threshold
                self.config.max_processing_iterations = 5
                logger.info("Configuration adapted for low-quality audio")
            elif quality_score > 0.8:
                # High quality - conservative enhancement
                self.config.quality_threshold = 0.2  # Higher threshold
                self.config.max_processing_iterations = 2
                logger.info("Configuration adapted for high-quality audio")
                
        except Exception as e:
            logger.warning(f"Failed to adapt configuration: {e}")
    
    def _evaluate_enhancement_quality(self, original: np.ndarray, enhanced: np.ndarray) -> Dict[str, float]:
        """Evaluate enhancement quality improvement"""
        try:
            # Calculate SNR improvement
            original_snr = self._calculate_advanced_snr(original)
            enhanced_snr = self._calculate_advanced_snr(enhanced)
            snr_improvement = enhanced_snr - original_snr
            
            # Calculate spectral similarity (should be high for good enhancement)
            original_spec = np.abs(librosa.stft(original))
            enhanced_spec = np.abs(librosa.stft(enhanced))
            
            # Downsample spectrograms for comparison
            min_frames = min(original_spec.shape[1], enhanced_spec.shape[1])
            original_spec_crop = original_spec[:, :min_frames]
            enhanced_spec_crop = enhanced_spec[:, :min_frames]
            
            # Calculate correlation
            orig_flat = original_spec_crop.flatten()
            enh_flat = enhanced_spec_crop.flatten()
            
            if len(orig_flat) > 0 and len(enh_flat) > 0:
                correlation = np.corrcoef(orig_flat, enh_flat)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
            else:
                correlation = 0.0
            
            # Overall quality improvement score
            improvement = snr_improvement * 0.7 + correlation * 0.3
            
            return {
                'improvement': improvement,
                'snr_improvement': snr_improvement,
                'spectral_correlation': correlation
            }
            
        except Exception as e:
            logger.warning(f"Quality evaluation failed: {e}")
            return {'improvement': 0.0, 'snr_improvement': 0.0, 'spectral_correlation': 0.0}
    
    def _calculate_advanced_snr(self, audio: np.ndarray) -> float:
        """Calculate advanced SNR estimate"""
        try:
            if len(audio) == 0:
                return 0.0
            
            # Remove DC
            audio = audio - np.mean(audio)
            
            # Frame-based analysis
            frame_size = min(2048, len(audio) // 10)
            if frame_size < 100:
                # Very short audio, use simple RMS-based SNR
                signal_power = np.mean(audio ** 2)
                return 10 * np.log10(signal_power / (signal_power * 0.1 + 1e-12))
            
            # Extract frames
            frames = []
            hop_size = frame_size // 2
            for i in range(0, len(audio) - frame_size, hop_size):
                frame = audio[i:i + frame_size]
                frames.append(frame)
            
            if not frames:
                return 10.0
            
            # Calculate frame energies
            frame_energies = [np.mean(frame ** 2) for frame in frames]
            frame_energies = np.array(frame_energies)
            
            # Statistical SNR estimation
            if len(frame_energies) < 4:
                return 15.0
            
            # Sort energies
            sorted_energies = np.sort(frame_energies)
            
            # Noise floor: average of lowest 20%
            noise_count = max(1, len(sorted_energies) // 5)
            noise_power = np.mean(sorted_energies[:noise_count])
            
            # Signal power: average of highest 20%
            signal_count = max(1, len(sorted_energies) // 5)
            signal_power = np.mean(sorted_energies[-signal_count:])
            
            # Calculate SNR
            if noise_power > 0 and signal_power > noise_power:
                snr = 10 * np.log10(signal_power / noise_power)
            else:
                snr = 15.0
            
            return np.clip(snr, 0, 60)
            
        except Exception as e:
            logger.debug(f"Advanced SNR calculation failed: {e}")
            return 15.0
    
    def _apply_final_professional_processing(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply final professional processing and mastering"""
        try:
            processed_audio = audio.copy()
            
            # 1. High-pass filter to remove DC and low-frequency artifacts
            sos = signal.butter(2, 40 / (sr / 2), 'high', output='sos')
            processed_audio = signal.sosfilt(sos, processed_audio)
            
            # 2. Gentle multiband compression
            processed_audio = self._apply_multiband_compression(processed_audio, sr)
            
            # 3. Subtle harmonic enhancement
            processed_audio = self._apply_harmonic_enhancement(processed_audio)
            
            # 4. Professional loudness normalization
            if PYLOUDNORM_AVAILABLE:
                try:
                    meter = pyln.Meter(sr)
                    loudness = meter.integrated_loudness(processed_audio)
                    if -50 < loudness < 0:  # Valid loudness range
                        processed_audio = pyln.normalize.loudness(
                            processed_audio, loudness, self.config.target_loudness
                        )
                        logger.info(f"Loudness normalized to {self.config.target_loudness} LUFS")
                except Exception as e:
                    logger.warning(f"Loudness normalization failed: {e}")
                    # Fallback to simple peak normalization
                    max_val = np.max(np.abs(processed_audio))
                    if max_val > 0:
                        processed_audio = processed_audio / max_val * 0.8
            else:
                # Simple peak normalization
                max_val = np.max(np.abs(processed_audio))
                if max_val > 0:
                    processed_audio = processed_audio / max_val * 0.8
            
            # 5. Final limiting and safety clipping
            processed_audio = self._apply_soft_limiting(processed_audio)
            processed_audio = np.clip(processed_audio, -0.98, 0.98)
            
            return processed_audio
            
        except Exception as e:
            logger.error(f"Final professional processing failed: {e}")
            return audio
    
    def _apply_multiband_compression(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply professional multiband compression"""
        try:
            # Define frequency bands for multiband compression
            bands = [
                (0, 250, 2.5, -25),      # Low: ratio, threshold
                (250, 1000, 2.0, -20),   # Low-mid
                (1000, 4000, 1.8, -18),  # Mid (speech)
                (4000, 8000, 2.2, -22),  # High-mid
                (8000, sr//2, 3.0, -28)  # High
            ]
            
            compressed_bands = []
            
            for low_freq, high_freq, ratio, threshold_db in bands:
                # Extract band
                nyquist = sr / 2
                low = max(low_freq / nyquist, 0.001)
                high = min(high_freq / nyquist, 0.999)
                
                if low >= high:
                    compressed_bands.append(np.zeros_like(audio))
                    continue
                
                if low <= 0.001:
                    sos = signal.butter(4, high, 'low', output='sos')
                elif high >= 0.999:
                    sos = signal.butter(4, low, 'high', output='sos')
                else:
                    sos = signal.butter(4, [low, high], 'band', output='sos')
                
                band_signal = signal.sosfilt(sos, audio)
                
                # Apply compression to this band
                compressed_band = self._apply_compression_to_band(
                    band_signal, threshold_db, ratio
                )
                compressed_bands.append(compressed_band)
            
            # Sum all bands
            return np.sum(compressed_bands, axis=0)
            
        except Exception as e:
            logger.warning(f"Multiband compression failed: {e}")
            return audio
    
    def _apply_compression_to_band(self, band_signal: np.ndarray, 
                                  threshold_db: float, ratio: float) -> np.ndarray:
        """Apply compression to a frequency band"""
        try:
            # Convert threshold to linear
            threshold_linear = 10 ** (threshold_db / 20)
            
            # Apply compression
            compressed_signal = np.zeros_like(band_signal)
            
            for i, sample in enumerate(band_signal):
                abs_sample = abs(sample)
                
                if abs_sample > threshold_linear:
                    # Above threshold - apply compression
                    excess = abs_sample - threshold_linear
                    compressed_excess = excess / ratio
                    compressed_abs = threshold_linear + compressed_excess
                    
                    # Preserve sign
                    compressed_signal[i] = np.sign(sample) * compressed_abs
                else:
                    # Below threshold - no compression
                    compressed_signal[i] = sample
            
            return compressed_signal
            
        except Exception as e:
            logger.warning(f"Band compression failed: {e}")
            return band_signal
    
    def _apply_harmonic_enhancement(self, audio: np.ndarray) -> np.ndarray:
        """Apply subtle harmonic enhancement"""
        try:
            # Add subtle harmonic distortion for warmth
            enhanced = audio.copy()
            
            # Gentle saturation
            drive = 1.05
            enhanced = np.tanh(enhanced * drive) / drive
            
            # Mix with original (parallel processing)
            mix_ratio = 0.15  # 15% processed, 85% original
            enhanced = (1 - mix_ratio) * audio + mix_ratio * enhanced
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"Harmonic enhancement failed: {e}")
            return audio
    
    def _apply_soft_limiting(self, audio: np.ndarray) -> np.ndarray:
        """Apply soft limiting to prevent clipping"""
        try:
            # Soft knee limiting
            threshold = 0.9
            knee = 0.1
            
            limited_audio = np.zeros_like(audio)
            
            for i, sample in enumerate(audio):
                abs_sample = abs(sample)
                
                if abs_sample <= threshold - knee:
                    # Below knee - no limiting
                    limited_audio[i] = sample
                elif abs_sample <= threshold + knee:
                    # In knee region - soft limiting
                    excess = abs_sample - (threshold - knee)
                    knee_ratio = excess / (2 * knee)
                    knee_gain = 1 - (knee_ratio ** 2) * 0.5
                    limited_audio[i] = np.sign(sample) * (threshold - knee + excess * knee_gain)
                else:
                    # Above knee - hard limiting
                    limited_audio[i] = np.sign(sample) * threshold
            
            return limited_audio
            
        except Exception as e:
            logger.warning(f"Soft limiting failed: {e}")
            return np.clip(audio, -0.98, 0.98)
    
    def _calculate_comprehensive_quality_metrics(self, original: np.ndarray, 
                                               enhanced: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive quality assessment"""
        try:
            metrics = {}
            
            # Ensure same length
            min_len = min(len(original), len(enhanced))
            orig = original[:min_len]
            enh = enhanced[:min_len]
            
            # SNR variants
            snr_metrics = self.quality_metrics.calculate_snr_variants(orig, enh)
            metrics.update(snr_metrics)
            
            # PESQ score
            metrics['pesq_score'] = self.quality_metrics.calculate_pesq(orig, enh)
            
            # STOI score
            metrics['stoi_score'] = self.quality_metrics.calculate_stoi(orig, enh)
            
            # Spectral characteristics
            orig_spec = np.abs(librosa.stft(orig))
            enh_spec = np.abs(librosa.stft(enh))
            
            # Spectral centroid change
            orig_centroid = np.mean(librosa.feature.spectral_centroid(S=orig_spec, sr=self.config.sample_rate))
            enh_centroid = np.mean(librosa.feature.spectral_centroid(S=enh_spec, sr=self.config.sample_rate))
            metrics['spectral_centroid_change'] = enh_centroid - orig_centroid
            
            # Spectral rolloff change
            orig_rolloff = np.mean(librosa.feature.spectral_rolloff(S=orig_spec, sr=self.config.sample_rate))
            enh_rolloff = np.mean(librosa.feature.spectral_rolloff(S=enh_spec, sr=self.config.sample_rate))
            metrics['spectral_rolloff_change'] = enh_rolloff - orig_rolloff
            
            # Energy preservation
            orig_energy = np.sum(orig ** 2)
            enh_energy = np.sum(enh ** 2)
            metrics['energy_ratio'] = enh_energy / (orig_energy + 1e-12)
            
            # Dynamic range
            orig_dynamic_range = np.max(orig) - np.min(orig)
            enh_dynamic_range = np.max(enh) - np.min(enh)
            metrics['dynamic_range_ratio'] = enh_dynamic_range / (orig_dynamic_range + 1e-12)
            
            # Overall quality score (0-1)
            quality_components = [
                np.clip(metrics['snr_improvement'] / 10, 0, 1),  # SNR improvement component
                metrics['pesq_score'] / 5.0,                     # PESQ component
                metrics['stoi_score'],                           # STOI component
                np.clip(metrics['energy_ratio'], 0.5, 1.5) - 0.5, # Energy preservation
            ]
            metrics['overall_quality_score'] = np.mean(quality_components)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Comprehensive quality calculation failed: {e}")
            return {'overall_quality_score': 0.5, 'snr_improvement': 0}
    
    def _log_comprehensive_results(self, results: Dict[str, Any]):
        """Log comprehensive enhancement results"""
        try:
            print(f"\n{'='*80}")
            print("🎵 PROFESSIONAL AUDIO ENHANCEMENT RESULTS")
            print(f"{'='*80}")
            
            # Processing summary
            print(f"⏱️  Total Processing Time: {results['processing_time']:.2f} seconds")
            
            # Content analysis
            content = results['content_analysis']
            print(f"🎯 Content Analysis:")
            print(f"   • Type: {content['content_type'].title()}")
            print(f"   • Noise Type: {content['noise_type'].replace('_', ' ').title()}")
            print(f"   • Input Quality Score: {content['quality_score']:.2f}/1.0")
            
            # Enhancement stages
            print(f"\n🔧 Enhancement Stages:")
            total_stages = len(results['enhancement_stages'])
            applied_stages = sum(1 for stage in results['enhancement_stages'] if stage['applied'])
            print(f"   • Total Stages: {total_stages}")
            print(f"   • Applied Stages: {applied_stages}")
            
            for stage in results['enhancement_stages']:
                status = "✅" if stage['applied'] else "⏭️"
                print(f"   {status} {stage['stage'].replace('_', ' ').title()}: "
                      f"{stage['time']:.3f}s, Quality Δ: {stage['quality_improvement']:+.3f}")
            
            # Quality metrics
            quality = results['quality_metrics']
            print(f"\n📊 Quality Metrics:")
            print(f"   • Overall Quality Score: {quality.get('overall_quality_score', 0):.3f}/1.0")
            print(f"   • SNR Improvement: {quality.get('snr_improvement', 0):+.2f} dB")
            print(f"   • PESQ Score: {quality.get('pesq_score', 0):.2f}/5.0")
            print(f"   • STOI Score: {quality.get('stoi_score', 0):.3f}/1.0")
            print(f"   • Spectral Centroid Change: {quality.get('spectral_centroid_change', 0):+.0f} Hz")
            print(f"   • Energy Ratio: {quality.get('energy_ratio', 1):.3f}")
            
            # GPU utilization
            gpu_info = results['gpu_memory_info']
            print(f"\n🚀 RTX A4000 Utilization:")
            print(f"   • Peak Memory Usage: {gpu_info.get('max_allocated_gb', 0):.2f} GB")
            print(f"   • Memory Utilization: {gpu_info.get('utilization_pct', 0):.1f}%")
            print(f"   • Available Memory: {gpu_info.get('free_gb', 0):.2f} GB")
            
            # Performance analysis
            stage_times = [stage['time'] for stage in results['enhancement_stages']]
            if stage_times:
                print(f"\n⚡ Performance Analysis:")
                print(f"   • Fastest Stage: {min(stage_times):.3f}s")
                print(f"   • Slowest Stage: {max(stage_times):.3f}s")
                print(f"   • Average Stage Time: {np.mean(stage_times):.3f}s")
                
                # Real-time factor
                content_duration = len(results.get('enhanced_audio', [1])) / self.config.sample_rate
                rt_factor = results['processing_time'] / content_duration if content_duration > 0 else 0
                print(f"   • Real-time Factor: {rt_factor:.2f}x")
                
                if rt_factor < 1.0:
                    print(f"   ✅ Real-time Processing Capable!")
                else:
                    print(f"   ⚠️  Slower than Real-time")
            
            # Enhancement quality assessment
            overall_quality = quality.get('overall_quality_score', 0)
            snr_improvement = quality.get('snr_improvement', 0)
            
            print(f"\n🎯 Enhancement Assessment:")
            if overall_quality > 0.8 and snr_improvement > 3:
                print(f"   ✅ EXCELLENT - Significant quality improvement achieved!")
            elif overall_quality > 0.6 and snr_improvement > 1:
                print(f"   ✅ GOOD - Noticeable quality improvement")
            elif overall_quality > 0.4 or snr_improvement > 0:
                print(f"   ⚠️  MODERATE - Some improvement, results may vary")
            else:
                print(f"   ❌ LIMITED - Minimal improvement, input may be high quality already")
            
            print(f"{'='*80}\n")
            
        except Exception as e:
            logger.error(f"Failed to log comprehensive results: {e}")
    
    def save_professional_results(self, audio: np.ndarray, sr: int, 
                                results: Dict[str, Any], output_path: str = None):
        """Save enhanced audio and comprehensive results"""
        if output_path is None:
            output_path = OUTPUT_AUDIO_PATH
        
        try:
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save enhanced audio
            sf.write(output_path, audio, sr, subtype='PCM_24')  # Higher quality
            logger.info(f"Enhanced audio saved to: {output_path}")
            
            # Save comprehensive metadata
            metadata_path = output_path.replace('.wav', '_professional_metadata.json')
            
            # Prepare metadata for JSON serialization
            json_results = self._prepare_json_serializable(results)
            json_results['audio_info'] = {
                'duration_seconds': len(audio) / sr,
                'sample_rate': sr,
                'channels': 1,
                'bit_depth': 24,
                'file_size_mb': Path(output_path).stat().st_size / (1024*1024) if Path(output_path).exists() else 0
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            logger.info(f"Professional metadata saved to: {metadata_path}")
            
            # Save processing history
            history_path = output_path.replace('.wav', '_processing_history.pkl')
            with open(history_path, 'wb') as f:
                pickle.dump(self.processing_history, f)
            
            # Generate professional report
            report_path = output_path.replace('.wav', '_professional_report.txt')
            self._generate_professional_report(results, report_path)
            
        except Exception as e:
            logger.error(f"Failed to save professional results: {e}")
            # Try to save just the audio
            try:
                sf.write(output_path, audio, sr, subtype='PCM_16')
                logger.info(f"Audio saved successfully (metadata failed): {output_path}")
            except Exception as e2:
                logger.error(f"Failed to save audio file: {e2}")
                raise
    
    def _prepare_json_serializable(self, obj: Any) -> Any:
        """Prepare object for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._prepare_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return list(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, '__dict__'):
            return self._prepare_json_serializable(obj.__dict__)
        else:
            return obj
    
    def _generate_professional_report(self, results: Dict[str, Any], report_path: str):
        """Generate comprehensive professional report"""
        try:
            with open(report_path, 'w') as f:
                f.write("PROFESSIONAL AUDIO ENHANCEMENT REPORT\n")
                f.write("=" * 50 + "\n\n")
                
                # Executive Summary
                f.write("EXECUTIVE SUMMARY\n")
                f.write("-" * 20 + "\n")
                overall_quality = results['quality_metrics'].get('overall_quality_score', 0)
                snr_improvement = results['quality_metrics'].get('snr_improvement', 0)
                
                if overall_quality > 0.7:
                    f.write("✅ Enhancement Status: SUCCESS\n")
                    f.write("Significant improvement in audio quality achieved.\n\n")
                elif overall_quality > 0.5:
                    f.write("⚠️ Enhancement Status: PARTIAL SUCCESS\n")
                    f.write("Moderate improvement in audio quality achieved.\n\n")
                else:
                    f.write("❌ Enhancement Status: LIMITED SUCCESS\n")
                    f.write("Minimal improvement achieved. Input may already be high quality.\n\n")
                
                # Technical Details
                f.write("TECHNICAL ANALYSIS\n")
                f.write("-" * 20 + "\n")
                f.write(f"Processing Time: {results['processing_time']:.2f} seconds\n")
                f.write(f"Content Type: {results['content_analysis']['content_type'].title()}\n")
                f.write(f"Noise Type: {results['content_analysis']['noise_type'].replace('_', ' ').title()}\n")
                f.write(f"SNR Improvement: {snr_improvement:+.2f} dB\n")
                f.write(f"PESQ Score: {results['quality_metrics'].get('pesq_score', 0):.2f}/5.0\n")
                f.write(f"STOI Score: {results['quality_metrics'].get('stoi_score', 0):.3f}/1.0\n\n")
                
                # Processing Stages
                f.write("PROCESSING STAGES\n")
                f.write("-" * 20 + "\n")
                for stage in results['enhancement_stages']:
                    status = "APPLIED" if stage['applied'] else "SKIPPED"
                    f.write(f"{stage['stage'].replace('_', ' ').title()}: {status} "
                           f"({stage['time']:.3f}s, Quality Δ: {stage['quality_improvement']:+.3f})\n")
                f.write("\n")
                
                # GPU Utilization
                gpu_info = results['gpu_memory_info']
                f.write("GPU UTILIZATION\n")
                f.write("-" * 20 + "\n")
                f.write(f"Peak Memory Usage: {gpu_info.get('max_allocated_gb', 0):.2f} GB\n")
                f.write(f"Memory Utilization: {gpu_info.get('utilization_pct', 0):.1f}%\n")
                f.write(f"Processing Efficiency: RTX A4000 Optimized\n\n")
                
                # Recommendations
                f.write("RECOMMENDATIONS\n")
                f.write("-" * 20 + "\n")
                if snr_improvement > 5:
                    f.write("• Excellent results achieved. Current settings are optimal.\n")
                elif snr_improvement > 2:
                    f.write("• Good results. Consider fine-tuning for specific content types.\n")
                elif snr_improvement > 0:
                    f.write("• Moderate improvement. Input audio may have limited enhancement potential.\n")
                else:
                    f.write("• Limited improvement. Consider using different enhancement strategies.\n")
                
                f.write(f"\nReport generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            logger.info(f"Professional report saved to: {report_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate professional report: {e}")

def main():
    """Main execution function for professional enhancement"""
    
    try:
        print(f"\n{'='*80}")
        print("🎵 PROFESSIONAL CUDA AUDIO ENHANCEMENT SYSTEM")
        print("RTX A4000 Optimized - Enterprise Grade")
        print(f"{'='*80}")
        print(f"Input file: {INPUT_AUDIO_PATH}")
        print(f"Output file: {OUTPUT_AUDIO_PATH}")
        print(f"GPU Optimization: RTX A4000 (16GB VRAM)")
        print(f"Mixed Precision: {'Enabled' if MIXED_PRECISION else 'Disabled'}")
        print(f"Tensor Cores: {'Enabled' if TENSOR_CORE_OPTIMIZATION else 'Disabled'}")
        print(f"Memory Allocation: {int(GPU_MEMORY_FRACTION * 16)}GB / 16GB")
        print(f"{'='*80}")
        
        # Initialize professional enhancer
        config = ProfessionalEnhancementConfig()
        enhancer = ProfessionalCUDAAudioEnhancer(config)
        
        # Process audio with comprehensive pipeline
        enhanced_audio, sample_rate, results = enhancer.enhance_audio_professional()
        
        # Save professional results
        enhancer.save_professional_results(enhanced_audio, sample_rate, results)
        
        # Memory cleanup
        enhancer.memory_manager.clear_cache_aggressive()
        
        print(f"\n🎯 PROFESSIONAL ENHANCEMENT COMPLETE!")
        print(f"Enhanced audio saved to: {OUTPUT_AUDIO_PATH}")
        
        # Final assessment
        overall_quality = results['quality_metrics'].get('overall_quality_score', 0)
        if overall_quality > 0.7:
            print(f"🏆 EXCELLENT RESULTS - Professional quality enhancement achieved!")
        elif overall_quality > 0.5:
            print(f"✅ GOOD RESULTS - Significant improvement in audio quality")
        else:
            print(f"⚠️  MODERATE RESULTS - Some improvement achieved")
        
        return True
        
    except FileNotFoundError:
        print(f"\n❌ ERROR: Audio file not found!")
        print(f"Please ensure the file exists at: {INPUT_AUDIO_PATH}")
        print(f"Update the INPUT_AUDIO_PATH variable with your actual file path.")
        return False
        
    except RuntimeError as e:
        if "CUDA" in str(e):
            print(f"\n❌ CUDA ERROR: {e}")
            print(f"Please ensure:")
            print(f"1. NVIDIA RTX A4000 GPU is available")
            print(f"2. CUDA drivers are properly installed")
            print(f"3. PyTorch with CUDA support is installed")
        else:
            print(f"\n❌ RUNTIME ERROR: {e}")
        return False
        
    except Exception as e:
        logger.error(f"Professional enhancement failed: {e}")
        print(f"\n❌ Enhancement failed: {e}")
        print(f"\nTroubleshooting tips:")
        print(f"1. Check input audio file format and integrity")
        print(f"2. Ensure sufficient GPU memory (16GB RTX A4000 recommended)")
        print(f"3. Verify all dependencies are installed")
        print(f"4. Check CUDA installation and compatibility")
        return False

if __name__ == "__main__":
    # Configuration validation
    print("\n" + "="*80)
    print("🔧 PROFESSIONAL SYSTEM CONFIGURATION")
    print("="*80)
    
    # Check if user has set their file path
    if INPUT_AUDIO_PATH in ["input_noisy_audio.wav", "your_noisy_audio.wav", "path/to/your/noisy_audio.wav"]:
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
        for ext in ["*.wav", "*.mp3", "*.flac", "*.m4a", "*.aac"]:
            audio_files.extend(list(current_dir.glob(ext)))
        
        if audio_files:
            print(f"\n📁 Found these audio files in current directory:")
            for i, file in enumerate(audio_files[:10], 1):  # Show up to 10 files
                print(f"   {i}. {file.name}")
            print(f'\nYou could use: INPUT_AUDIO_PATH = "{audio_files[0].name}"')
        
        print("\n" + "="*80)
        sys.exit(1)
    
    # Validate the path
    if not Path(INPUT_AUDIO_PATH).exists():
        print(f"❌ ERROR: Audio file not found!")
        print(f"Current path: {INPUT_AUDIO_PATH}")
        print(f"Full path: {Path(INPUT_AUDIO_PATH).absolute()}")
        print(f"\nPlease check:")
        print(f"1. File exists at the specified location")
        print(f"2. Path is correctly spelled") 
        print(f"3. Use forward slashes (/) or raw strings on Windows")
        print(f"\nExample valid paths:")
        print(f'INPUT_AUDIO_PATH = "my_audio.wav"  # Same folder')
        print(f'INPUT_AUDIO_PATH = "C:/Users/YourName/Desktop/audio.wav"  # Windows')
        print(f'INPUT_AUDIO_PATH = "/home/user/Documents/audio.wav"  # Linux/Mac')
        print(f'INPUT_AUDIO_PATH = r"C:\\Users\\YourName\\Desktop\\audio.wav"  # Windows raw string')
        sys.exit(1)
    
    # System requirements check
    print(f"✅ Configuration validated!")
    print(f"📁 Input file: {INPUT_AUDIO_PATH}")
    print(f"📁 Output file: {OUTPUT_AUDIO_PATH}")
    print(f"🚀 Target GPU: RTX A4000 (16GB VRAM)")
    print(f"💾 Memory allocation: {int(GPU_MEMORY_FRACTION * 16)}GB")
    print(f"⚡ Batch size: {BATCH_SIZE_GPU}")
    print(f"🔧 Chunk size: {CHUNK_SIZE}")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print(f"❌ CUDA not available! This system requires NVIDIA GPU with CUDA support.")
        sys.exit(1)
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🎯 Detected GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        
        if gpu_memory < 8:
            print(f"⚠️  WARNING: GPU has {gpu_memory:.1f}GB VRAM. Recommended: 16GB+ for optimal performance")
            print(f"    Consider reducing BATCH_SIZE_GPU and GPU_MEMORY_FRACTION")
        elif "A4000" in gpu_name or gpu_memory >= 16:
            print(f"✅ Excellent! GPU is well-suited for professional enhancement.")
        else:
            print(f"✅ GPU detected. Performance may vary depending on workload.")
    
    print("="*80)
    
    # Run the main enhancement
    success = main()
    
    if success:
        print(f"\n🎉 PROFESSIONAL ENHANCEMENT COMPLETED SUCCESSFULLY!")
    else:
        print(f"\n❌ ENHANCEMENT FAILED - Please check the errors above")
        sys.exit(1)