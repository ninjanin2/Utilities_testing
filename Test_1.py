"""
CUDA-Optimized Audio Enhancement System for ASR
===============================================
Professional-grade audio enhancement optimized for CUDA GPU acceleration
with global input path configuration for streamlined processing.

Author: CUDA-Enhanced ASR Pipeline  
Version: 3.1.0-CUDA
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
from scipy.optimize import differential_evolution
import librosa
import soundfile as sf

# GPU-accelerated imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

# Optional advanced libraries
try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False
    print("Warning: noisereduce not available. Using alternative noise reduction.")

try:
    import pyloudnorm as pyln
    PYLOUDNORM_AVAILABLE = True
except ImportError:
    PYLOUDNORM_AVAILABLE = False
    print("Warning: pyloudnorm not available. Using alternative normalization.")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ==================== GLOBAL CONFIGURATION ====================
# 🔥 IMPORTANT: SET YOUR AUDIO FILE PATH HERE! 🔥
# Replace "your_noisy_audio.wav" with the actual path to your audio file
INPUT_AUDIO_PATH = "your_noisy_audio.wav"  # ← CHANGE THIS TO YOUR ACTUAL FILE PATH!
OUTPUT_AUDIO_PATH = "enhanced_audio_cuda.wav"

# Examples of how to set the path:
# INPUT_AUDIO_PATH = "my_recording.wav"                    # File in same folder
# INPUT_AUDIO_PATH = "C:/Users/YourName/Desktop/audio.wav" # Windows full path  
# INPUT_AUDIO_PATH = "/home/user/Documents/audio.wav"      # Linux/Mac full path

# CUDA Configuration - Advanced users can modify these
CUDA_DEVICE = "cuda:0"  # Use first GPU, change to cuda:1, cuda:2, etc. if you have multiple GPUs
MIXED_PRECISION = True  # Enable for RTX series and newer GPUs for 2x speed boost
GPU_MEMORY_FRACTION = 0.8  # Use 80% of GPU memory, adjust based on your GPU
BATCH_SIZE_GPU = 8  # Parallel processing batch size for GPU
# ================================================================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_cuda_environment():
    """Setup and optimize CUDA environment"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! This script requires a CUDA-capable GPU.")
    
    # Set CUDA device
    device = torch.device(CUDA_DEVICE)
    torch.cuda.set_device(device)
    
    # Get GPU info
    gpu_name = torch.cuda.get_device_name(device)
    gpu_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
    
    logger.info(f"CUDA GPU detected: {gpu_name}")
    logger.info(f"GPU Memory: {gpu_memory:.1f} GB")
    logger.info(f"Mixed Precision: {'Enabled' if MIXED_PRECISION else 'Disabled'}")
    
    # Optimize CUDA settings
    torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
    torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
    
    # Set memory fraction
    if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
        torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)
    
    return device

@dataclass
class CUDAEnhancementConfig:
    """CUDA-optimized configuration for audio enhancement"""
    # Audio parameters
    sample_rate: int = 16000
    frame_length: int = 512   # Reduced for RTX 3050 compatibility
    hop_length: int = 128     # Reduced for RTX 3050 compatibility  
    n_fft: int = 512          # Reduced for RTX 3050 compatibility
    
    # Enhancement parameters
    wiener_alpha: float = 0.98
    spectral_floor: float = 0.002
    vad_threshold: float = 0.5
    noise_gate_threshold: float = -40
    compression_ratio: float = 3.0
    target_loudness: float = -23.0
    
    # GPU-specific parameters (optimized for RTX 3050)
    cuda_device: str = CUDA_DEVICE
    mixed_precision: bool = MIXED_PRECISION
    gpu_batch_size: int = 2   # Further reduced for stability
    chunk_size: int = 8192    # Smaller chunks for better memory usage
    overlap_size: int = 1024  # Smaller overlap
    
    # Memory management (conservative for RTX 3050)
    max_gpu_memory_mb: int = int(GPU_MEMORY_FRACTION * 3200)  # Conservative for 4GB GPU
    prefetch_batches: int = 1  # Reduced for lower memory usage
    
    # Optimization settings
    enable_optimization: bool = True
    optimization_iterations: int = 20  # Reduced for faster processing

class CUDAMemoryManager:
    """Efficient CUDA memory management"""
    
    def __init__(self, device):
        self.device = device
        self.peak_memory = 0
        
    def clear_cache(self):
        """Clear CUDA cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def get_memory_info(self):
        """Get current GPU memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device) / 1e9
            cached = torch.cuda.memory_reserved(self.device) / 1e9
            return allocated, cached
        return 0, 0
    
    @contextmanager
    def memory_context(self, operation_name="operation"):
        """Context manager for memory monitoring"""
        self.clear_cache()
        start_allocated, start_cached = self.get_memory_info()
        
        try:
            yield
        finally:
            end_allocated, end_cached = self.get_memory_info()
            logger.debug(f"{operation_name} - Memory: {end_allocated:.2f}GB allocated, {end_cached:.2f}GB cached")
            self.clear_cache()

class CUDAOptimizedLSTMVAD(nn.Module):
    """CUDA-optimized LSTM Voice Activity Detection"""
    
    def __init__(self, input_dim=40, hidden_dim=128, num_layers=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, dropout=0.1
        )
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=8, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Classification layers
        x = F.relu(self.fc1(attn_out))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return self.sigmoid(x)

class CUDAOptimizedTransformer(nn.Module):
    """CUDA-optimized Transformer for speech enhancement"""
    
    def __init__(self, n_fft=512, d_model=256, nhead=8, num_layers=6, dim_feedforward=1024):
        super().__init__()
        self.d_model = d_model
        
        # Calculate correct input dimension based on n_fft
        input_dim = n_fft // 2 + 1  # For n_fft=512 -> 257, for n_fft=1024 -> 513
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, d_model))
        
        # Transformer encoder with CUDA optimizations
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation='gelu',  # Better for GPU
            batch_first=True,
            norm_first=True     # Pre-LN for better training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, input_dim),  # Match input dimension
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
        
        # Output projection (enhancement mask)
        mask = self.output_projection(x)
        
        return mask

class CUDAOptimizedAutoencoder(nn.Module):
    """CUDA-optimized Convolutional Autoencoder - simplified for reliability"""
    
    def __init__(self):
        super().__init__()
        
        # Simplified encoder without skip connections to avoid shape issues
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=1, padding=7),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 32, kernel_size=15, stride=2, padding=7),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=7),
            nn.ReLU(),
            nn.BatchNorm1d(64),
        )
        
        # Simplified decoder without skip connections
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=15, stride=2, padding=7, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.ConvTranspose1d(32, 16, kernel_size=15, stride=2, padding=7, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.ConvTranspose1d(16, 1, kernel_size=15, stride=1, padding=7),
            nn.Tanh()
        )
        
    def forward(self, x):
        # Simple autoencoder without skip connections
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class CUDAAdvancedAudioEnhancer:
    """CUDA-optimized main enhancement system"""
    
    def __init__(self, config: Optional[CUDAEnhancementConfig] = None):
        self.config = config or CUDAEnhancementConfig()
        
        # Setup CUDA
        self.device = setup_cuda_environment()
        self.memory_manager = CUDAMemoryManager(self.device)
        
        # Initialize mixed precision scaler
        self.scaler = GradScaler() if self.config.mixed_precision else None
        
        # Performance tracking
        self.metrics = defaultdict(list)
        
        # Initialize models
        self._initialize_cuda_models()
        
        logger.info("CUDA-Optimized Audio Enhancer initialized successfully")
    
    def _initialize_cuda_models(self):
        """Initialize all models with CUDA optimization"""
        with self.memory_manager.memory_context("model_initialization"):
            try:
                # VAD Model
                self.vad_model = CUDAOptimizedLSTMVAD().to(self.device)
                self.vad_model.eval()
                
                # Transformer Enhancement Model
                self.transformer_enhancer = CUDAOptimizedTransformer(
                    n_fft=self.config.n_fft,
                    d_model=256,  # Reduced for RTX 3050
                    nhead=8,      # Reduced for RTX 3050
                    num_layers=4, # Reduced for RTX 3050
                    dim_feedforward=512  # Reduced for RTX 3050
                ).to(self.device)
                self.transformer_enhancer.eval()
                
                # Autoencoder Model
                self.autoencoder = CUDAOptimizedAutoencoder().to(self.device)
                self.autoencoder.eval()
                
                # Try to compile models for extra speed (PyTorch 2.0+)
                # Skip compilation if Triton is not available (which is causing errors)
                compilation_enabled = False
                try:
                    # Check if triton is available before attempting compilation
                    import triton
                    if hasattr(torch, 'compile'):
                        # Test compilation with a simple tensor first
                        test_tensor = torch.randn(1, 10, 40).to(self.device)
                        with torch.no_grad():
                            _ = self.vad_model(test_tensor)
                        
                        # Only compile if test passes and triton is available
                        self.vad_model = torch.compile(self.vad_model, mode="reduce-overhead")
                        self.transformer_enhancer = torch.compile(self.transformer_enhancer, mode="reduce-overhead")
                        self.autoencoder = torch.compile(self.autoencoder, mode="reduce-overhead")
                        compilation_enabled = True
                        logger.info("Models compiled for optimized execution")
                except ImportError:
                    logger.info("Triton not available - running without model compilation (still fast on GPU)")
                except Exception as e:
                    logger.warning(f"Model compilation skipped: {e}")
                
                if not compilation_enabled:
                    logger.info("Running with eager execution - models will work normally")
                
                # Load pre-trained weights if available
                self._load_pretrained_weights()
                
                logger.info("All CUDA models initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize CUDA models: {e}")
                raise
    
    def _load_pretrained_weights(self):
        """Load pre-trained weights with CUDA optimization"""
        model_dir = Path("models")
        
        if model_dir.exists():
            try:
                # Load with map_location for CUDA
                def load_model_weights(model, model_path):
                    if model_path.exists():
                        state_dict = torch.load(model_path, map_location=self.device)
                        model.load_state_dict(state_dict)
                        return True
                    return False
                
                vad_loaded = load_model_weights(self.vad_model, model_dir / "vad_model.pth")
                transformer_loaded = load_model_weights(self.transformer_enhancer, model_dir / "transformer_model.pth")
                autoencoder_loaded = load_model_weights(self.autoencoder, model_dir / "autoencoder_model.pth")
                
                loaded_count = sum([vad_loaded, transformer_loaded, autoencoder_loaded])
                logger.info(f"Loaded {loaded_count}/3 pre-trained models")
                
            except Exception as e:
                logger.warning(f"Failed to load some pre-trained weights: {e}")
        else:
            logger.info("No pre-trained models found, using randomly initialized weights")
    
    def load_audio_global(self) -> Tuple[np.ndarray, int]:
        """Load audio from global INPUT_AUDIO_PATH"""
        return self.load_audio(INPUT_AUDIO_PATH)
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load and preprocess audio file with validation"""
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
                raise ValueError("Audio contains invalid values")
            
            # Resample if necessary
            if sr != self.config.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.config.sample_rate)
                sr = self.config.sample_rate
            
            # Normalize to prevent clipping while preserving dynamics
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val * 0.95
            
            load_time = time.time() - start_time
            self.metrics['audio_loading'].append(load_time)
            
            logger.info(f"Audio loaded: {len(audio)/sr:.2f}s at {sr}Hz ({load_time:.3f}s)")
            return audio, sr
            
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            raise
    
    def cuda_spectral_subtraction(self, audio: np.ndarray) -> np.ndarray:
        """GPU-accelerated spectral subtraction"""
        with self.memory_manager.memory_context("spectral_subtraction"):
            try:
                # Convert to tensor and move to GPU
                audio_tensor = torch.FloatTensor(audio).to(self.device)
                
                # GPU STFT
                stft = torch.stft(
                    audio_tensor,
                    n_fft=self.config.n_fft,
                    hop_length=self.config.hop_length,
                    window=torch.hann_window(self.config.n_fft).to(self.device),
                    return_complex=True
                )
                
                magnitude = torch.abs(stft)
                phase = torch.angle(stft)
                
                # Estimate noise profile (first 0.5 seconds)
                noise_frames = min(int(0.5 * self.config.sample_rate / self.config.hop_length), 
                                 magnitude.shape[1] // 4)
                noise_profile = torch.median(magnitude[:, :noise_frames], dim=1, keepdim=True)[0]
                
                # Spectral subtraction with adaptive over-subtraction
                alpha = 2.0
                enhanced_magnitude = magnitude - alpha * noise_profile
                
                # Apply spectral floor
                floor_magnitude = self.config.spectral_floor * magnitude
                enhanced_magnitude = torch.maximum(enhanced_magnitude, floor_magnitude)
                
                # Reconstruct signal
                enhanced_stft = enhanced_magnitude * torch.exp(1j * phase)
                enhanced_audio = torch.istft(
                    enhanced_stft,
                    n_fft=self.config.n_fft,
                    hop_length=self.config.hop_length,
                    window=torch.hann_window(self.config.n_fft).to(self.device)
                )
                
                return enhanced_audio.cpu().numpy()
                
            except Exception as e:
                logger.error(f"CUDA spectral subtraction failed: {e}")
                return audio
    
    def cuda_wiener_filter(self, audio: np.ndarray) -> np.ndarray:
        """GPU-accelerated Wiener filtering"""
        with self.memory_manager.memory_context("wiener_filter"):
            try:
                audio_tensor = torch.FloatTensor(audio).to(self.device)
                
                stft = torch.stft(
                    audio_tensor,
                    n_fft=self.config.n_fft,
                    hop_length=self.config.hop_length,
                    window=torch.hann_window(self.config.n_fft).to(self.device),
                    return_complex=True
                )
                
                magnitude = torch.abs(stft)
                phase = torch.angle(stft)
                
                # Noise power estimation
                noise_frames = min(20, magnitude.shape[1] // 5)
                noise_power = torch.mean(magnitude[:, :noise_frames] ** 2, dim=1, keepdim=True)
                
                # Signal power
                signal_power = magnitude ** 2
                
                # Wiener gain
                wiener_gain = signal_power / (signal_power + self.config.wiener_alpha * noise_power + 1e-8)
                
                # Apply filter
                filtered_magnitude = wiener_gain * magnitude
                
                # Reconstruct
                filtered_stft = filtered_magnitude * torch.exp(1j * phase)
                filtered_audio = torch.istft(
                    filtered_stft,
                    n_fft=self.config.n_fft,
                    hop_length=self.config.hop_length,
                    window=torch.hann_window(self.config.n_fft).to(self.device)
                )
                
                return filtered_audio.cpu().numpy()
                
            except Exception as e:
                logger.error(f"CUDA Wiener filtering failed: {e}")
                return audio
    
    def cuda_transformer_enhancement(self, audio: np.ndarray) -> np.ndarray:
        """GPU-accelerated transformer enhancement with mixed precision and fallback"""
        with self.memory_manager.memory_context("transformer_enhancement"):
            try:
                audio_tensor = torch.FloatTensor(audio).to(self.device)
                
                # Compute STFT
                stft = torch.stft(
                    audio_tensor,
                    n_fft=self.config.n_fft,
                    hop_length=self.config.hop_length,
                    window=torch.hann_window(self.config.n_fft).to(self.device),
                    return_complex=True
                )
                
                magnitude = torch.abs(stft)
                phase = torch.angle(stft)
                
                # Normalize for neural network
                magnitude_norm = magnitude / (torch.max(magnitude) + 1e-8)
                
                # Process in smaller batches to avoid memory issues
                batch_size = min(self.config.gpu_batch_size, 4)  # Smaller batches for RTX 3050
                seq_len = magnitude_norm.shape[1]
                enhanced_magnitude = torch.zeros_like(magnitude_norm)
                
                # Track failed batches
                failed_batches = 0
                total_batches = (seq_len + batch_size - 1) // batch_size
                
                for i in range(0, seq_len, batch_size):
                    try:
                        end_idx = min(i + batch_size, seq_len)
                        batch_mag = magnitude_norm[:, i:end_idx].permute(1, 0).unsqueeze(0)
                        
                        # Mixed precision inference with proper error handling
                        with torch.no_grad():
                            if self.config.mixed_precision:
                                try:
                                    with autocast(enabled=True):
                                        mask = self.transformer_enhancer(batch_mag)
                                except Exception as amp_error:
                                    # Only log first few failures to avoid spam
                                    if failed_batches < 3:
                                        logger.warning(f"Mixed precision failed, using regular precision: {amp_error}")
                                    mask = self.transformer_enhancer(batch_mag)
                            else:
                                mask = self.transformer_enhancer(batch_mag)
                        
                        enhanced_magnitude[:, i:end_idx] = (mask.squeeze(0).permute(1, 0) * magnitude[:, i:end_idx])
                        
                        # Clear intermediate tensors
                        del batch_mag, mask
                        
                    except RuntimeError as e:
                        failed_batches += 1
                        if "out of memory" in str(e).lower():
                            logger.warning("GPU memory error in transformer, using traditional enhancement")
                            return self._traditional_enhancement_fallback(audio)
                        elif "mat1 and mat2 shapes cannot be multiplied" in str(e):
                            # Only log first few shape errors
                            if failed_batches <= 3:
                                logger.error(f"Transformer shape error: {e}")
                            # Use original magnitude for failed batches
                            enhanced_magnitude[:, i:end_idx] = magnitude[:, i:end_idx]
                        else:
                            # For other CUDA errors, continue with unprocessed segment
                            if failed_batches <= 3:
                                logger.warning(f"Transformer batch {i} failed: {e}")
                            enhanced_magnitude[:, i:end_idx] = magnitude[:, i:end_idx]
                
                # Log summary of failures
                if failed_batches > 0:
                    failure_rate = failed_batches / total_batches * 100
                    if failure_rate > 50:
                        logger.warning(f"Transformer failed on {failed_batches}/{total_batches} batches ({failure_rate:.1f}%), using traditional fallback")
                        return self._traditional_enhancement_fallback(audio)
                    else:
                        logger.info(f"Transformer completed with {failed_batches}/{total_batches} failed batches ({failure_rate:.1f}%)")
                
                # Reconstruct with error handling
                try:
                    enhanced_stft = enhanced_magnitude * torch.exp(1j * phase)
                    enhanced_audio = torch.istft(
                        enhanced_stft,
                        n_fft=self.config.n_fft,
                        hop_length=self.config.hop_length,
                        window=torch.hann_window(self.config.n_fft).to(self.device)
                    )
                    
                    return enhanced_audio.cpu().numpy()
                    
                except Exception as istft_error:
                    logger.error(f"ISTFT reconstruction failed: {istft_error}")
                    return self._traditional_enhancement_fallback(audio)
                
            except Exception as e:
                logger.error(f"CUDA transformer enhancement failed: {e}")
                return self._traditional_enhancement_fallback(audio)
    
    def _traditional_enhancement_fallback(self, audio: np.ndarray) -> np.ndarray:
        """Traditional enhancement fallback when transformer fails"""
        try:
            # Simple frequency domain enhancement
            stft = librosa.stft(audio, n_fft=self.config.n_fft, hop_length=self.config.hop_length)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Apply simple enhancement (emphasize mid frequencies)
            freq_bins = magnitude.shape[0]
            enhancement_curve = np.ones(freq_bins)
            
            # Boost speech frequencies (300-3400 Hz)
            speech_start = int(300 * freq_bins / (self.config.sample_rate / 2))
            speech_end = int(3400 * freq_bins / (self.config.sample_rate / 2))
            enhancement_curve[speech_start:speech_end] = 1.2
            
            enhanced_magnitude = magnitude * enhancement_curve[:, np.newaxis]
            
            # Reconstruct
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            return librosa.istft(enhanced_stft, hop_length=self.config.hop_length)
            
        except Exception as e:
            logger.error(f"Traditional enhancement fallback failed: {e}")
            return audio
    
    def cuda_autoencoder_denoising(self, audio: np.ndarray) -> np.ndarray:
        """GPU-accelerated autoencoder denoising with chunked processing"""
        with self.memory_manager.memory_context("autoencoder_denoising"):
            try:
                original_length = len(audio)
                chunk_size = self.config.chunk_size
                overlap_size = self.config.overlap_size
                
                # Process overlapping chunks for smooth reconstruction
                enhanced_chunks = []
                
                for i in range(0, len(audio), chunk_size - overlap_size):
                    # Extract chunk with overlap
                    start_idx = max(0, i - overlap_size // 2)
                    end_idx = min(len(audio), i + chunk_size + overlap_size // 2)
                    chunk = audio[start_idx:end_idx]
                    
                    # Pad to fixed size if necessary
                    if len(chunk) < chunk_size:
                        chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
                    
                    # Convert to tensor
                    chunk_tensor = torch.FloatTensor(chunk).unsqueeze(0).unsqueeze(0).to(self.device)
                    
                    # Mixed precision denoising
                    with autocast(enabled=self.config.mixed_precision):
                        with torch.no_grad():
                            denoised_chunk = self.autoencoder(chunk_tensor)
                    
                    # Extract the main part (without overlap)
                    chunk_start = overlap_size // 2 if i > 0 else 0
                    chunk_end = chunk_size - overlap_size // 2 if i + chunk_size < len(audio) else len(chunk)
                    
                    main_part = denoised_chunk.squeeze().cpu().numpy()[chunk_start:chunk_end]
                    enhanced_chunks.append(main_part)
                
                # Concatenate chunks
                enhanced_audio = np.concatenate(enhanced_chunks)[:original_length]
                
                return enhanced_audio
                
            except Exception as e:
                logger.error(f"CUDA autoencoder denoising failed: {e}")
                return audio
    
    def cuda_vad_processing(self, audio: np.ndarray) -> np.ndarray:
        """GPU-accelerated Voice Activity Detection with fixed broadcasting"""
        with self.memory_manager.memory_context("vad_processing"):
            try:
                # Extract MFCC features
                mfcc = librosa.feature.mfcc(y=audio, sr=self.config.sample_rate, n_mfcc=40)
                
                # Convert to tensor and move to GPU
                mfcc_tensor = torch.FloatTensor(mfcc.T).unsqueeze(0).to(self.device)
                
                # Mixed precision VAD
                with autocast(enabled=self.config.mixed_precision):
                    with torch.no_grad():
                        vad_output = self.vad_model(mfcc_tensor)
                
                # Convert to mask
                vad_mask = vad_output.squeeze().cpu().numpy() > self.config.vad_threshold
                
                # Smooth VAD decisions
                vad_mask = signal.medfilt(vad_mask.astype(float), kernel_size=5)
                
                # Calculate proper frame length for MFCC
                # MFCC frames correspond to hop_length samples in audio
                samples_per_mfcc_frame = self.config.hop_length
                
                # Expand VAD mask to match audio length
                vad_mask_expanded = np.repeat(vad_mask, samples_per_mfcc_frame)
                
                # Ensure exact length match with audio
                if len(vad_mask_expanded) > len(audio):
                    vad_mask_expanded = vad_mask_expanded[:len(audio)]
                elif len(vad_mask_expanded) < len(audio):
                    # Pad with the last value
                    padding_length = len(audio) - len(vad_mask_expanded)
                    last_value = vad_mask_expanded[-1] if len(vad_mask_expanded) > 0 else 1.0
                    vad_mask_expanded = np.concatenate([
                        vad_mask_expanded, 
                        np.full(padding_length, last_value)
                    ])
                
                # Apply smoothing window for gradual transitions
                if len(vad_mask_expanded) > 0:
                    window_size = min(self.config.hop_length, len(vad_mask_expanded) // 10)
                    if window_size > 3:
                        window = signal.windows.hann(window_size)
                        vad_mask_smooth = np.convolve(vad_mask_expanded, window/window.sum(), mode='same')
                    else:
                        vad_mask_smooth = vad_mask_expanded
                else:
                    vad_mask_smooth = np.ones_like(audio)
                
                # Final safety check
                if len(vad_mask_smooth) != len(audio):
                    logger.warning(f"VAD mask length mismatch, using original audio")
                    return audio
                
                return audio * vad_mask_smooth
                
            except Exception as e:
                logger.error(f"CUDA VAD processing failed: {e}")
                return audio
    
    def optimize_parameters_cuda(self, audio: np.ndarray) -> Dict:
        """CUDA-accelerated parameter optimization"""
        if not self.config.enable_optimization:
            return {}
        
        logger.info("Starting CUDA-accelerated parameter optimization...")
        
        def objective(params):
            """Objective function for optimization"""
            wiener_alpha, spectral_floor, vad_threshold = params
            
            # Temporarily update parameters
            original_params = (self.config.wiener_alpha, self.config.spectral_floor, self.config.vad_threshold)
            self.config.wiener_alpha = wiener_alpha
            self.config.spectral_floor = spectral_floor
            self.config.vad_threshold = vad_threshold
            
            try:
                # Quick processing pipeline for optimization
                processed = self.cuda_wiener_filter(audio[:min(len(audio), 16000)])  # Use first second
                snr = self._calculate_snr(processed)
                
                # Restore original parameters
                self.config.wiener_alpha, self.config.spectral_floor, self.config.vad_threshold = original_params
                
                return -snr  # Minimize negative SNR
                
            except Exception:
                # Restore parameters on error
                self.config.wiener_alpha, self.config.spectral_floor, self.config.vad_threshold = original_params
                return 100  # High penalty for failed optimization
        
        # Parameter bounds
        bounds = [
            (0.5, 1.0),    # wiener_alpha
            (0.001, 0.01), # spectral_floor
            (0.3, 0.7)     # vad_threshold
        ]
        
        try:
            result = differential_evolution(
                objective, bounds, 
                maxiter=self.config.optimization_iterations,
                seed=42, workers=1  # Single worker to avoid GPU conflicts
            )
            
            # Update config with optimal parameters
            self.config.wiener_alpha = result.x[0]
            self.config.spectral_floor = result.x[1]
            self.config.vad_threshold = result.x[2]
            
            optimal_params = {
                'wiener_alpha': result.x[0],
                'spectral_floor': result.x[1],
                'vad_threshold': result.x[2],
                'estimated_snr': -result.fun
            }
            
            logger.info(f"Optimization complete: {optimal_params}")
            return optimal_params
            
        except Exception as e:
            logger.error(f"Parameter optimization failed: {e}")
            return {}
    
    def enhance_audio_cuda_pipeline(self, audio_path: str = None) -> Tuple[np.ndarray, int]:
        """Main CUDA-optimized enhancement pipeline with quality validation"""
        # Use global path if not provided
        if audio_path is None:
            audio_path = INPUT_AUDIO_PATH
            logger.info(f"Using global input path: {INPUT_AUDIO_PATH}")
        
        logger.info(f"Starting CUDA enhancement pipeline for: {audio_path}")
        
        # Load audio
        start_time = time.time()
        audio, sr = self.load_audio(audio_path)
        original_audio = audio.copy()
        
        # Step 1: Parameter optimization (only if models are pre-trained)
        if self.config.enable_optimization and self._has_pretrained_models():
            optimal_params = self.optimize_parameters_cuda(audio)
            self.metrics['optimization_params'] = [optimal_params]
        else:
            logger.info("Skipping optimization - using random model weights")
        
        # Step 2: Quality-validated enhancement pipeline
        logger.info("Starting quality-validated enhancement...")
        
        # Traditional methods only (always safe)
        logger.info("Applying CUDA spectral subtraction...")
        start_time_stage = time.time()
        enhanced_audio = self.cuda_spectral_subtraction(audio)
        self.metrics['spectral_subtraction'].append(time.time() - start_time_stage)
        
        # Validate improvement after each stage
        original_snr = self._calculate_snr(original_audio)
        current_snr = self._calculate_snr(enhanced_audio)
        
        if current_snr < original_snr - 5:  # If SNR drops by more than 5dB, stop
            logger.warning(f"Spectral subtraction degraded quality (SNR: {original_snr:.1f} -> {current_snr:.1f}), reverting")
            enhanced_audio = original_audio.copy()
        else:
            logger.info(f"Spectral subtraction: SNR {original_snr:.1f} -> {current_snr:.1f} dB")
        
        # Step 3: Wiener filtering
        logger.info("Applying CUDA Wiener filter...")
        start_time_stage = time.time()
        wiener_result = self.cuda_wiener_filter(enhanced_audio)
        wiener_snr = self._calculate_snr(wiener_result)
        self.metrics['wiener_filter'].append(time.time() - start_time_stage)
        
        if wiener_snr >= current_snr - 2:  # Only apply if not much worse
            enhanced_audio = wiener_result
            current_snr = wiener_snr
            logger.info(f"Wiener filter applied: SNR now {current_snr:.1f} dB")
        else:
            logger.warning(f"Wiener filter degraded quality, skipping")
        
        # Step 4: Advanced noise reduction (if available and beneficial)
        if NOISEREDUCE_AVAILABLE:
            logger.info("Applying advanced noise reduction...")
            start_time_stage = time.time()
            try:
                nr_result = nr.reduce_noise(y=enhanced_audio, sr=sr, stationary=False, prop_decrease=0.6)  # Gentler
                nr_snr = self._calculate_snr(nr_result)
                
                if nr_snr >= current_snr - 2:
                    enhanced_audio = nr_result
                    current_snr = nr_snr
                    logger.info(f"Noise reduction applied: SNR now {current_snr:.1f} dB")
                else:
                    logger.warning("Noise reduction degraded quality, skipping")
                    
            except Exception as e:
                logger.warning(f"Noise reduction failed: {e}")
            
            self.metrics['advanced_noise_reduction'].append(time.time() - start_time_stage)
        
        # Step 5: Only apply deep learning if we have pre-trained models AND they're enabled
        if (self.config.enable_deep_learning and 
            hasattr(self, 'autoencoder') and 
            self._has_pretrained_models()):
            logger.info("Applying deep learning enhancement with pre-trained models...")
            
            # Autoencoder denoising
            start_time_stage = time.time()
            ae_result = self.cuda_autoencoder_denoising(enhanced_audio)
            ae_snr = self._calculate_snr(ae_result)
            self.metrics['autoencoder_denoising'].append(time.time() - start_time_stage)
            
            if ae_snr >= current_snr - 2:
                enhanced_audio = ae_result
                current_snr = ae_snr
                logger.info(f"Autoencoder applied: SNR now {current_snr:.1f} dB")
            else:
                logger.warning("Autoencoder degraded quality, skipping")
            
            # Transformer enhancement
            start_time_stage = time.time()
            transformer_result = self.cuda_transformer_enhancement(enhanced_audio)
            transformer_snr = self._calculate_snr(transformer_result)
            self.metrics['transformer_enhancement'].append(time.time() - start_time_stage)
            
            if transformer_snr >= current_snr - 2:
                enhanced_audio = transformer_result
                current_snr = transformer_snr
                logger.info(f"Transformer applied: SNR now {current_snr:.1f} dB")
            else:
                logger.warning("Transformer degraded quality, skipping")
            
            # VAD processing
            start_time_stage = time.time()
            vad_result = self.cuda_vad_processing(enhanced_audio)
            vad_snr = self._calculate_snr(vad_result)
            self.metrics['vad_processing'].append(time.time() - start_time_stage)
            
            if vad_snr >= current_snr - 3:  # VAD can reduce SNR slightly but remove noise
                enhanced_audio = vad_result
                current_snr = vad_snr
                logger.info(f"VAD applied: SNR now {current_snr:.1f} dB")
            else:
                logger.warning("VAD degraded quality significantly, skipping")
        else:
            logger.info("Skipping deep learning models (no pre-trained weights or disabled for safety)")
            # Add placeholder metrics
            self.metrics['autoencoder_denoising'].append(0.0)
            self.metrics['transformer_enhancement'].append(0.0)
            self.metrics['vad_processing'].append(0.0)
        
        # Step 6: Final safe processing
        logger.info("Applying final processing...")
        start_time_stage = time.time()
        
        # Gentle high-pass filter
        sos = signal.butter(2, 60, 'hp', fs=sr, output='sos')  # Gentler filter
        filtered_audio = signal.sosfilt(sos, enhanced_audio)
        
        # Gentle compression
        final_audio = self._apply_gentle_compression(filtered_audio)
        
        # Gentle normalization
        if PYLOUDNORM_AVAILABLE:
            try:
                meter = pyln.Meter(sr)
                loudness = meter.integrated_loudness(final_audio)
                if loudness < -50:  # Only normalize if not too quiet
                    final_audio = pyln.normalize.loudness(final_audio, loudness, -20.0)  # Gentler target
            except Exception as e:
                logger.warning(f"Loudness normalization failed: {e}")
        
        # Safety clipping - much gentler
        final_audio = np.clip(final_audio, -0.95, 0.95)
        
        # Final validation
        final_snr = self._calculate_snr(final_audio)
        if final_snr < original_snr - 10:  # If final result is much worse
            logger.warning(f"Final result much worse than original ({original_snr:.1f} -> {final_snr:.1f}), using minimal processing")
            # Fall back to very gentle processing
            final_audio = self._minimal_safe_enhancement(original_audio, sr)
            final_snr = self._calculate_snr(final_audio)
        
        self.metrics['final_processing'].append(time.time() - start_time_stage)
        
        # Calculate metrics
        total_time = time.time() - start_time
        snr_improvement = final_snr - original_snr
        
        # Store final metrics
        self.metrics['total_processing_time'] = [total_time]
        self.metrics['snr_improvement'] = [snr_improvement]
        
        logger.info(f"CUDA enhancement complete!")
        logger.info(f"Processing time: {total_time:.2f}s")
        logger.info(f"Original SNR: {original_snr:.2f} dB")
        logger.info(f"Enhanced SNR: {final_snr:.2f} dB")
        logger.info(f"SNR improvement: {snr_improvement:.2f} dB")
        
        if snr_improvement > 0:
            logger.info("✅ Enhancement successful - audio quality improved")
        else:
            logger.warning("⚠️  Enhancement may not have improved quality significantly")
        
        # Memory cleanup
        self.memory_manager.clear_cache()
        
        return final_audio, sr
    
    def _has_pretrained_models(self) -> bool:
        """Check if we have actual pre-trained models loaded"""
        # For now, assume we don't have pre-trained models
        # In production, this would check if actual weights were loaded
        model_dir = Path("models")
        return (model_dir.exists() and 
                (model_dir / "vad_model.pth").exists() and
                (model_dir / "transformer_model.pth").exists() and
                (model_dir / "autoencoder_model.pth").exists())
    
    def _apply_gentle_compression(self, audio: np.ndarray) -> np.ndarray:
        """Apply very gentle dynamic range compression"""
        try:
            # Much gentler compression
            threshold_db = -30  # Higher threshold
            ratio = 1.5  # Lower ratio
            
            audio_db = 20 * np.log10(np.abs(audio) + 1e-8)
            mask = audio_db > threshold_db
            audio_db[mask] = threshold_db + (audio_db[mask] - threshold_db) / ratio
            
            compressed_audio = np.sign(audio) * (10 ** (audio_db / 20))
            return compressed_audio
            
        except Exception as e:
            logger.error(f"Gentle compression failed: {e}")
            return audio
    
    def _minimal_safe_enhancement(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Minimal safe enhancement that won't introduce artifacts"""
        try:
            # Only very basic, proven-safe processing
            
            # 1. Gentle high-pass filter to remove rumble
            sos = signal.butter(1, 40, 'hp', fs=sr, output='sos')
            filtered = signal.sosfilt(sos, audio)
            
            # 2. Very gentle normalization
            max_val = np.max(np.abs(filtered))
            if max_val > 0:
                normalized = filtered / max_val * 0.7  # Conservative level
            else:
                normalized = filtered
            
            logger.info("Applied minimal safe enhancement")
            return normalized
            
        except Exception as e:
            logger.error(f"Minimal enhancement failed: {e}")
            return audio
    
    def _apply_compression(self, audio: np.ndarray) -> np.ndarray:
        """Apply dynamic range compression"""
        try:
            # Simple compression
            threshold_db = -20
            ratio = self.config.compression_ratio
            
            audio_db = 20 * np.log10(np.abs(audio) + 1e-8)
            mask = audio_db > threshold_db
            audio_db[mask] = threshold_db + (audio_db[mask] - threshold_db) / ratio
            
            compressed_audio = np.sign(audio) * (10 ** (audio_db / 20))
            return compressed_audio
            
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            return audio
    
    def _calculate_snr(self, audio: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio with improved, more conservative estimation"""
        try:
            if len(audio) == 0:
                return 0.0
            
            # Remove DC component
            audio = audio - np.mean(audio)
            
            # Method 1: RMS-based approach (most reliable)
            # Use moving window to find quiet and active sections
            window_size = min(1600, len(audio) // 50)  # 0.1 second windows
            if window_size < 100:
                # For very short audio, use simple approach
                signal_power = np.mean(audio ** 2)
                if signal_power > 0:
                    return 10 * np.log10(signal_power / (signal_power * 0.1 + 1e-12))
                else:
                    return 0.0
            
            # Calculate RMS for overlapping windows
            rms_values = []
            step = window_size // 2
            for i in range(0, len(audio) - window_size, step):
                window = audio[i:i + window_size]
                rms = np.sqrt(np.mean(window ** 2))
                rms_values.append(rms)
            
            rms_values = np.array(rms_values)
            
            if len(rms_values) < 4:
                # Not enough windows for reliable estimation
                signal_power = np.mean(audio ** 2)
                return 10 * np.log10(signal_power / (signal_power * 0.1 + 1e-12))
            
            # Sort RMS values to find noise floor and signal peaks
            sorted_rms = np.sort(rms_values)
            
            # Noise floor: average of quietest 25% of windows
            noise_floor_idx = max(1, len(sorted_rms) // 4)
            noise_rms = np.mean(sorted_rms[:noise_floor_idx])
            
            # Signal level: average of loudest 25% of windows  
            signal_start_idx = max(len(sorted_rms) * 3 // 4, noise_floor_idx + 1)
            signal_rms = np.mean(sorted_rms[signal_start_idx:])
            
            # Ensure signal is actually higher than noise
            if signal_rms <= noise_rms * 1.1:  # Very little dynamic range
                # Use overall RMS vs minimum RMS
                overall_rms = np.sqrt(np.mean(audio ** 2))
                min_rms = np.min(rms_values[rms_values > 1e-8])  # Avoid zeros
                if min_rms > 0:
                    snr = 20 * np.log10(overall_rms / min_rms)
                else:
                    snr = 20.0
            else:
                # Calculate SNR from noise floor and signal level
                snr = 20 * np.log10(signal_rms / (noise_rms + 1e-12))
            
            # Sanity check and clip to reasonable range
            if np.isnan(snr) or np.isinf(snr):
                snr = 15.0  # Default reasonable value
            
            # Audio with very low noise should have high SNR, very noisy audio should have low SNR
            return np.clip(snr, 0, 60)  # Reasonable range for audio SNR
            
        except Exception as e:
            logger.debug(f"SNR calculation error: {e}")
            return 15.0  # Default reasonable value
    
    def save_enhanced_audio(self, audio: np.ndarray, sr: int, output_path: str = None):
        """Save enhanced audio with comprehensive metadata"""
        if output_path is None:
            output_path = OUTPUT_AUDIO_PATH
        
        try:
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save audio with high quality
            sf.write(output_path, audio, sr, subtype='PCM_16')
            
            # Get GPU memory info for metadata
            allocated, cached = self.memory_manager.get_memory_info()
            
            # Safe calculation functions
            def safe_mean(metric_list):
                if not metric_list or len(metric_list) == 0:
                    return 0.0
                try:
                    if isinstance(metric_list[0], dict):
                        # For optimization params, extract numeric values safely
                        result = {}
                        for key, value in metric_list[0].items():
                            if isinstance(value, (np.floating, np.integer)):
                                result[key] = float(value)
                            elif isinstance(value, np.ndarray):
                                result[key] = float(value.item()) if value.size == 1 else value.tolist()
                            else:
                                result[key] = value
                        return result
                    else:
                        return float(np.mean([float(x) for x in metric_list]))
                except Exception:
                    return 0.0
            
            def safe_sum(metric_list):
                if not metric_list or len(metric_list) == 0:
                    return 0.0
                try:
                    return float(np.sum([float(x) for x in metric_list]))
                except Exception:
                    return 0.0
            
            def safe_float(value):
                """Convert any numeric type to float"""
                try:
                    if isinstance(value, (np.floating, np.integer)):
                        return float(value)
                    elif isinstance(value, np.ndarray):
                        return float(value.item()) if value.size == 1 else value.tolist()
                    else:
                        return float(value)
                except Exception:
                    return 0.0
            
            # Build metadata with safe conversions
            metadata = {
                'processing_info': {
                    'input_file': str(INPUT_AUDIO_PATH),
                    'output_file': str(output_path),
                    'sample_rate': int(sr),
                    'duration': safe_float(len(audio) / sr),
                    'processing_time': safe_sum(self.metrics.get('total_processing_time', [])),
                    'snr_improvement': safe_mean(self.metrics.get('snr_improvement', []))
                },
                'cuda_info': {
                    'device': str(self.device),
                    'gpu_name': str(torch.cuda.get_device_name(self.device)),
                    'memory_allocated_gb': safe_float(allocated),
                    'memory_cached_gb': safe_float(cached),
                    'mixed_precision': bool(self.config.mixed_precision),
                    'gpu_memory_total': safe_float(torch.cuda.get_device_properties(self.device).total_memory / 1e9)
                },
                'enhancement_config': {
                    'sample_rate': int(self.config.sample_rate),
                    'n_fft': int(self.config.n_fft),
                    'hop_length': int(self.config.hop_length),
                    'chunk_size': int(self.config.chunk_size),
                    'wiener_alpha': safe_float(self.config.wiener_alpha),
                    'spectral_floor': safe_float(self.config.spectral_floor),
                    'vad_threshold': safe_float(self.config.vad_threshold)
                },
                'optimization_params': safe_mean(self.metrics.get('optimization_params', [{}])),
                'performance_metrics': {
                    'audio_loading_time': safe_mean(self.metrics.get('audio_loading', [])),
                    'spectral_subtraction_time': safe_mean(self.metrics.get('spectral_subtraction', [])),
                    'wiener_filter_time': safe_mean(self.metrics.get('wiener_filter', [])),
                    'advanced_noise_reduction_time': safe_mean(self.metrics.get('advanced_noise_reduction', [])),
                    'transformer_enhancement_time': safe_mean(self.metrics.get('transformer_enhancement', [])),
                    'autoencoder_denoising_time': safe_mean(self.metrics.get('autoencoder_denoising', [])),
                    'vad_processing_time': safe_mean(self.metrics.get('vad_processing', [])),
                    'final_processing_time': safe_mean(self.metrics.get('final_processing', []))
                }
            }
            
            # Save metadata
            metadata_path = output_path.replace('.wav', '_cuda_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Enhanced audio saved to: {output_path}")
            logger.info(f"CUDA metadata saved to: {metadata_path}")
            
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
    """Main execution function for CUDA-optimized enhancement"""
    
    try:
        print(f"\n{'='*80}")
        print("CUDA-OPTIMIZED AUDIO ENHANCEMENT FOR ASR")
        print(f"{'='*80}")
        print(f"Input file: {INPUT_AUDIO_PATH}")
        print(f"Output file: {OUTPUT_AUDIO_PATH}")
        print(f"GPU Device: {CUDA_DEVICE}")
        print(f"Mixed Precision: {'Enabled' if MIXED_PRECISION else 'Disabled'}")
        print(f"{'='*80}")
        
        # Initialize CUDA enhancer
        config = CUDAEnhancementConfig()
        enhancer = CUDAAdvancedAudioEnhancer(config)
        
        # Process audio using global path
        enhanced_audio, sample_rate = enhancer.enhance_audio_cuda_pipeline()
        
        # Save enhanced audio
        enhancer.save_enhanced_audio(enhanced_audio, sample_rate)
        
        # Print final summary
        print(f"\n{'='*80}")
        print("✅ CUDA ENHANCEMENT COMPLETE!")
        print(f"{'='*80}")
        print(f"📁 Input: {INPUT_AUDIO_PATH}")
        print(f"📁 Output: {OUTPUT_AUDIO_PATH}")
        print(f"⏱️  Duration: {len(enhanced_audio)/sample_rate:.2f} seconds")
        print(f"🚀 GPU: {torch.cuda.get_device_name(enhancer.device)}")
        print(f"📊 SNR Improvement: {np.mean(enhancer.metrics.get('snr_improvement', [0])):.2f} dB")
        
        snr_improvement = np.mean(enhancer.metrics.get('snr_improvement', [0]))
        if snr_improvement > 0:
            print(f"✅ Enhancement Status: SUCCESS - Audio quality improved!")
        elif snr_improvement > -3:
            print(f"⚠️  Enhancement Status: NEUTRAL - Minor changes")
        else:
            print(f"❌ Enhancement Status: DEGRADED - Consider using traditional methods only")
        
        print(f"⚡ Processing Time: {np.sum(enhancer.metrics.get('total_processing_time', [0])):.2f} seconds")
        print(f"💾 GPU Memory Used: {enhancer.memory_manager.get_memory_info()[0]:.2f} GB")
        
        # Additional performance breakdown
        total_time = np.sum(enhancer.metrics.get('total_processing_time', [0]))
        if total_time > 0:
            print(f"\n🔍 Processing Breakdown:")
            breakdown = [
                ('Spectral Subtraction', enhancer.metrics.get('spectral_subtraction', [])),
                ('Wiener Filter', enhancer.metrics.get('wiener_filter', [])),
                ('Advanced Noise Reduction', enhancer.metrics.get('advanced_noise_reduction', [])),
                ('Autoencoder Denoising', enhancer.metrics.get('autoencoder_denoising', [])),
                ('Transformer Enhancement', enhancer.metrics.get('transformer_enhancement', [])),
                ('VAD Processing', enhancer.metrics.get('vad_processing', [])),
                ('Final Processing', enhancer.metrics.get('final_processing', []))
            ]
            
            for name, times in breakdown:
                if times:
                    avg_time = np.mean(times)
                    percentage = (avg_time / total_time) * 100
                    print(f"  • {name}: {avg_time:.2f}s ({percentage:.1f}%)")
        
        print(f"{'='*80}")
        
    except Exception as e:
        logger.error(f"CUDA enhancement failed: {e}")
        print(f"\n❌ Enhancement failed: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure CUDA is properly installed")
        print("2. Check if your GPU has sufficient memory")
        print("3. Verify the input audio file exists and is valid")
        print("4. Try reducing GPU_MEMORY_FRACTION or BATCH_SIZE_GPU")
        sys.exit(1)

if __name__ == "__main__":
    # ============= CONFIGURATION SECTION =============
    # 🔥 IMPORTANT: EDIT THE LINE BELOW WITH YOUR AUDIO FILE PATH! 🔥
    
    print("\n" + "="*60)
    print("🎵 CUDA AUDIO ENHANCEMENT CONFIGURATION")
    print("="*60)
    
    # Check if user has set their file path
    if INPUT_AUDIO_PATH in ["path/to/your/audio.wav", "your_noisy_audio.wav", "path/to/your/noisy_audio.wav"]:
        print("❌ CONFIGURATION REQUIRED!")
        print("\nPlease edit this script and change the INPUT_AUDIO_PATH variable.")
        print("Find this line in the script and change it:")
        print("INPUT_AUDIO_PATH = \"your_noisy_audio.wav\"  # ← CHANGE THIS!")
        print("\nTo your actual file path, for example:")
        print("INPUT_AUDIO_PATH = \"C:/my_audio/noisy_recording.wav\"")
        print("INPUT_AUDIO_PATH = \"/home/user/audio/my_file.wav\"")
        print("INPUT_AUDIO_PATH = \"my_audio.wav\"  # if in same folder")
        
        # Try to help user find audio files in current directory
        current_dir = Path(".")
        audio_files = list(current_dir.glob("*.wav")) + list(current_dir.glob("*.mp3")) + list(current_dir.glob("*.flac"))
        
        if audio_files:
            print(f"\n📁 Found these audio files in current directory:")
            for i, file in enumerate(audio_files[:5], 1):
                print(f"   {i}. {file.name}")
            print(f"\nYou could use: INPUT_AUDIO_PATH = \"{audio_files[0].name}\"")
        
        print("\n" + "="*60)
        sys.exit(1)
    
    # Validate the path
    if not Path(INPUT_AUDIO_PATH).exists():
        print(f"❌ ERROR: Audio file not found!")
        print(f"Current path: {INPUT_AUDIO_PATH}")
        print(f"Full path: {Path(INPUT_AUDIO_PATH).absolute()}")
        print(f"\nPlease check:")
        print(f"1. File exists at the specified location")
        print(f"2. Path is correctly spelled") 
        print(f"3. Use forward slashes (/) or double backslashes (\\\\) on Windows")
        print(f"\nExample valid paths:")
        print(f"INPUT_AUDIO_PATH = \"my_audio.wav\"  # Same folder")
        print(f"INPUT_AUDIO_PATH = \"C:/Users/YourName/Desktop/audio.wav\"  # Windows")
        print(f"INPUT_AUDIO_PATH = \"/home/user/Documents/audio.wav\"  # Linux/Mac")
        sys.exit(1)
    
    # GPU Settings (optimized for RTX 3050 - 4GB GPU)
    CUDA_DEVICE = "cuda:0"        # Change to cuda:1, cuda:2, etc. for multi-GPU
    MIXED_PRECISION = True        # Keep enabled for RTX series
    GPU_MEMORY_FRACTION = 0.5     # Very conservative for 4GB GPU
    BATCH_SIZE_GPU = 2           # Reduced for maximum stability
    
    print(f"✅ Configuration validated!")
    print(f"📁 Input file: {INPUT_AUDIO_PATH}")
    print(f"📁 Output file: {OUTPUT_AUDIO_PATH}")
    print("="*60)
    
    main()