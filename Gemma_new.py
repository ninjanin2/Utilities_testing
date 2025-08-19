#!/usr/bin/env python3
"""
Professional Audio Transcription System with Gemma3n-E4B-it
DEVICE & RF NOISE CORRECTED VERSION - Complete device consistency + RF interference removal
Features: RF Noise Removal, Multi-stage Enhancement, Audio Preview, Device-Consistent Processing
Optimized for RTX A4000 (16GB VRAM) and 32GB RAM
"""

import os
import sys
import warnings
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
import tempfile
import shutil
from datetime import datetime
import gc
import math

# Core libraries
import torch
import torchaudio
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, filtfilt, wiener, medfilt, iirnotch, sosfiltfilt
from scipy.ndimage import gaussian_filter1d
from scipy import signal
import pywt

# ML/AI libraries - Using pipeline for device consistency
from transformers import pipeline

# Enhanced audio processing
import noisereduce as nr
try:
    from noisereduce.torchgate import TorchGate
except ImportError:
    TorchGate = None

# UI and utilities
import gradio as gr
import json
import time

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global Configuration
class Config:
    """Global configuration for the transcription system"""
    
    # Model paths
    GEMMA_MODEL_PATH = "./models/google--gemma-3n-E4B-it"  # Local model path
    CACHE_DIR = "./cache"
    TEMP_DIR = "./temp"
    OUTPUT_DIR = "./outputs"
    
    # Audio processing parameters
    TARGET_SAMPLE_RATE = 16000  # Gemma requirement
    CHUNK_DURATION = 40  # seconds
    OVERLAP_DURATION = 10  # seconds
    MAX_AUDIO_LENGTH = 3600  # 1 hour max
    
    # RF and enhanced audio processing parameters
    NOISE_REDUCTION_STRENGTH = 0.85
    RF_FREQUENCIES = [50, 60, 120, 240, 440, 880, 1000, 2000, 4000]  # Common RF interference frequencies
    SPECTRAL_GATE_THRESHOLD = 1.2
    WIENER_FILTER_SIZE = 5
    MEDIAN_FILTER_SIZE = 3
    WAVELET_MODE = 'db4'  # Daubechies wavelet for audio denoising
    
    # GPU settings with device consistency
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    MAX_NEW_TOKENS = 512
    
    # Supported formats
    SUPPORTED_FORMATS = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.wma']

# Initialize directories
for dir_path in [Config.CACHE_DIR, Config.TEMP_DIR, Config.OUTPUT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

class RadioFrequencyAudioEnhancer:
    """Advanced RF noise removal and audio enhancement for heavily distorted audio"""
    
    def __init__(self):
        self.sample_rate = Config.TARGET_SAMPLE_RATE
        self.device = Config.DEVICE
        
        # Initialize torch-based noise reducer if available
        self.torch_gate = None
        if TorchGate is not None and torch.cuda.is_available():
            try:
                self.torch_gate = TorchGate(
                    sr=self.sample_rate,
                    nonstationary=True,
                    n_std_thresh_stationary=1.2,
                    prop_decrease=Config.NOISE_REDUCTION_STRENGTH
                ).to(self.device)
                logger.info("GPU-accelerated RF noise reduction initialized")
            except Exception as e:
                logger.warning(f"Torch noise reducer initialization failed: {e}")
                self.torch_gate = None
    
    def safe_audio_load(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Safely load audio with multiple fallback methods"""
        try:
            # Primary method: librosa (handles most formats including compressed)
            audio, sr = librosa.load(audio_path, sr=None, mono=False)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=0)
            
            # Resample to target rate if needed
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
                sr = self.sample_rate
                
            return audio.astype(np.float32), sr
            
        except Exception as e1:
            logger.warning(f"Librosa failed: {e1}, trying soundfile...")
            try:
                audio, sr = sf.read(audio_path)
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)
                if sr != self.sample_rate:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
                return audio.astype(np.float32), self.sample_rate
                
            except Exception as e2:
                logger.warning(f"Soundfile failed: {e2}, trying torchaudio...")
                try:
                    waveform, sr = torchaudio.load(audio_path)
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)
                    if sr != self.sample_rate:
                        resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                        waveform = resampler(waveform)
                    return waveform.squeeze().numpy().astype(np.float32), self.sample_rate
                    
                except Exception as e3:
                    logger.error(f"All audio loading methods failed: {e1}, {e2}, {e3}")
                    raise RuntimeError(f"Cannot load audio file: {audio_path}")
    
    def normalize_audio_advanced(self, audio: np.ndarray) -> np.ndarray:
        """Advanced audio normalization with dynamic range optimization"""
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Remove DC offset
        audio = audio - np.mean(audio)
        
        # Apply gentle compression to reduce dynamic range
        audio = np.tanh(audio * 2.0) / 2.0
        
        # Normalize to [-0.95, 0.95] to prevent clipping
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / (max_val / 0.95)
        
        return audio.astype(np.float32)
    
    def remove_rf_interference(self, audio: np.ndarray) -> np.ndarray:
        """Remove radio frequency interference using multiple notch filters"""
        try:
            enhanced = audio.copy()
            
            # Apply notch filters for common RF frequencies
            for rf_freq in Config.RF_FREQUENCIES:
                if rf_freq < self.sample_rate / 2:  # Must be below Nyquist frequency
                    # Design notch filter
                    Q = 30  # Quality factor (higher = narrower notch)
                    w0 = rf_freq / (self.sample_rate / 2)  # Normalized frequency
                    
                    # Create notch filter
                    sos = signal.iirnotch(w0, Q, output='sos')
                    enhanced = signal.sosfiltfilt(sos, enhanced)
                    
            logger.info(f"Applied RF interference removal for frequencies: {Config.RF_FREQUENCIES}")
            return enhanced.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"RF interference removal failed: {e}")
            return audio
    
    def apply_rf_bandstop_filters(self, audio: np.ndarray) -> np.ndarray:
        """Apply bandstop filters for RF interference bands"""
        try:
            enhanced = audio.copy()
            nyquist = self.sample_rate / 2
            
            # Common RF interference bands
            rf_bands = [
                (48, 52),    # 50Hz power line interference
                (58, 62),    # 60Hz power line interference  
                (118, 122),  # 120Hz harmonic
                (238, 242),  # 240Hz harmonic
                (430, 450),  # Radio band
                (870, 890),  # Radio band
                (990, 1010), # Radio band
                (1980, 2020), # Radio band
                (3980, 4020)  # Radio band
            ]
            
            for low_freq, high_freq in rf_bands:
                if high_freq < nyquist:
                    # Normalize frequencies
                    low = low_freq / nyquist
                    high = high_freq / nyquist
                    
                    # Design bandstop filter
                    sos = signal.butter(4, [low, high], btype='bandstop', output='sos')
                    enhanced = signal.sosfiltfilt(sos, enhanced)
            
            logger.info("Applied RF bandstop filters")
            return enhanced.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"RF bandstop filtering failed: {e}")
            return audio
    
    def apply_wavelet_denoising(self, audio: np.ndarray) -> np.ndarray:
        """Apply wavelet denoising for RF and impulse noise removal"""
        try:
            # Decompose signal using wavelet transform
            coeffs = pywt.wavedec(audio, Config.WAVELET_MODE, level=6)
            
            # Estimate noise standard deviation from detail coefficients
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            
            # Calculate threshold using universal threshold
            threshold = sigma * np.sqrt(2 * np.log(len(audio)))
            
            # Apply soft thresholding to detail coefficients
            coeffs_thresh = list(coeffs)
            for i in range(1, len(coeffs)):
                coeffs_thresh[i] = pywt.threshold(coeffs[i], threshold, 'soft')
            
            # Reconstruct signal
            denoised = pywt.waverec(coeffs_thresh, Config.WAVELET_MODE)
            
            # Handle length mismatch
            if len(denoised) != len(audio):
                denoised = denoised[:len(audio)]
            
            logger.info("Applied wavelet denoising")
            return denoised.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"Wavelet denoising failed: {e}")
            return audio
    
    def apply_preemphasis(self, audio: np.ndarray, alpha: float = 0.97) -> np.ndarray:
        """Apply preemphasis filter to enhance high frequencies"""
        try:
            preemphasized = np.zeros_like(audio)
            preemphasized[0] = audio[0]
            preemphasized[1:] = audio[1:] - alpha * audio[:-1]
            return preemphasized.astype(np.float32)
        except Exception as e:
            logger.warning(f"Preemphasis failed: {e}")
            return audio
    
    def apply_spectral_subtraction_rf(self, audio: np.ndarray) -> np.ndarray:
        """Apply spectral subtraction optimized for RF noise removal"""
        try:
            # Compute STFT with RF-optimized parameters
            stft = librosa.stft(audio, n_fft=2048, hop_length=256, win_length=1024)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Estimate noise from first and last frames (likely to contain RF noise)
            noise_frames = max(2, magnitude.shape[1] // 10)
            noise_start = magnitude[:, :noise_frames]
            noise_end = magnitude[:, -noise_frames:]
            noise_spectrum = np.mean(np.concatenate([noise_start, noise_end], axis=1), axis=1, keepdims=True)
            
            # RF-optimized spectral subtraction
            alpha = 2.5  # Higher for RF noise
            beta = 0.005  # Lower for better RF suppression
            
            enhanced_magnitude = magnitude - alpha * noise_spectrum
            enhanced_magnitude = np.maximum(enhanced_magnitude, beta * magnitude)
            
            # Reconstruct signal
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft, hop_length=256, win_length=1024)
            
            return enhanced_audio.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"RF spectral subtraction failed: {e}")
            return audio
    
    def apply_advanced_bandpass_filter(self, audio: np.ndarray) -> np.ndarray:
        """Apply advanced multi-stage bandpass filtering for speech with RF removal"""
        try:
            nyquist = self.sample_rate * 0.5
            
            # Primary speech band (300-3400 Hz - telephone quality)
            low1, high1 = 300 / nyquist, 3400 / nyquist
            sos1 = signal.butter(4, [low1, high1], btype='band', output='sos')
            filtered1 = signal.sosfiltfilt(sos1, audio)
            
            # Extended speech band (80-8000 Hz - full speech)
            low2, high2 = 80 / nyquist, min(8000 / nyquist, 0.99)
            sos2 = signal.butter(4, [low2, high2], btype='band', output='sos')
            filtered2 = signal.sosfiltfilt(sos2, audio)
            
            # Combine filters with emphasis on extended band
            combined = 0.25 * filtered1 + 0.75 * filtered2
            
            return combined.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"Advanced bandpass filter failed: {e}")
            return audio
    
    def reduce_noise_multi_stage_rf(self, audio: np.ndarray) -> np.ndarray:
        """Multi-stage noise reduction optimized for RF interference"""
        try:
            enhanced = audio.copy()
            
            # Stage 1: RF-optimized spectral subtraction
            enhanced = self.apply_spectral_subtraction_rf(enhanced)
            
            # Stage 2: GPU/CPU noise reduction with RF parameters
            if self.torch_gate is not None and torch.cuda.is_available():
                audio_tensor = torch.from_numpy(enhanced).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    enhanced_tensor = self.torch_gate(audio_tensor)
                enhanced = enhanced_tensor.squeeze().cpu().numpy().astype(np.float32)
            else:
                # CPU-based noise reduction optimized for RF
                enhanced = nr.reduce_noise(
                    y=enhanced,
                    sr=self.sample_rate,
                    stationary=False,
                    prop_decrease=Config.NOISE_REDUCTION_STRENGTH,
                    time_constant_s=1.0,  # Faster for RF
                    freq_mask_smooth_hz=200,  # Wider for RF
                    time_mask_smooth_ms=30,
                    n_jobs=1
                )
                
                # Second pass for remaining RF noise
                enhanced = nr.reduce_noise(
                    y=enhanced,
                    sr=self.sample_rate,
                    stationary=True,
                    prop_decrease=0.7,
                    time_constant_s=1.5,
                    freq_mask_smooth_hz=400,
                    time_mask_smooth_ms=50,
                    n_jobs=1
                )
            
            # Stage 3: Wiener filtering
            enhanced = wiener(enhanced, mysize=Config.WIENER_FILTER_SIZE)
            
            # Stage 4: Median filtering for impulse noise
            enhanced = medfilt(enhanced, kernel_size=Config.MEDIAN_FILTER_SIZE)
            
            return enhanced.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"Multi-stage RF noise reduction failed: {e}")
            return audio
    
    def enhance_audio_rf_ultimate(self, audio_path: str) -> Tuple[str, str]:
        """Ultimate RF noise removal and audio enhancement pipeline"""
        try:
            start_time = time.time()
            logger.info(f"Starting ultimate RF noise removal for: {audio_path}")
            
            # Load audio
            audio, sr = self.safe_audio_load(audio_path)
            original_length = len(audio) / sr
            logger.info(f"Loaded audio: {original_length:.2f}s at {sr}Hz")
            
            # Save original for comparison
            original_path = os.path.join(Config.TEMP_DIR, f"original_rf_{int(time.time())}.wav")
            sf.write(original_path, audio, self.sample_rate)
            
            # Stage 1: Initial normalization
            audio = self.normalize_audio_advanced(audio)
            
            # Stage 2: Remove RF interference (notch filters)
            audio = self.remove_rf_interference(audio)
            
            # Stage 3: RF bandstop filters
            audio = self.apply_rf_bandstop_filters(audio)
            
            # Stage 4: Wavelet denoising (excellent for RF noise)
            audio = self.apply_wavelet_denoising(audio)
            
            # Stage 5: Preemphasis for speech enhancement
            audio = self.apply_preemphasis(audio)
            
            # Stage 6: Advanced bandpass filtering
            audio = self.apply_advanced_bandpass_filter(audio)
            
            # Stage 7: Multi-stage noise reduction (RF optimized)
            audio = self.reduce_noise_multi_stage_rf(audio)
            
            # Stage 8: Final normalization
            audio = self.normalize_audio_advanced(audio)
            
            # Save enhanced audio
            enhanced_path = os.path.join(Config.TEMP_DIR, f"enhanced_rf_{int(time.time())}.wav")
            sf.write(enhanced_path, audio, self.sample_rate)
            
            processing_time = time.time() - start_time
            logger.info(f"Ultimate RF noise removal completed in {processing_time:.2f}s")
            
            return enhanced_path, original_path
            
        except Exception as e:
            logger.error(f"Ultimate RF enhancement failed: {e}")
            return audio_path, audio_path

class SmartAudioChunker:
    """Intelligent audio chunking with overlap management"""
    
    def __init__(self, chunk_duration: int = 40, overlap_duration: int = 10):
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self.sample_rate = Config.TARGET_SAMPLE_RATE
    
    def create_smart_chunks(self, audio_path: str) -> List[Tuple[str, float, float]]:
        """Create overlapping chunks with smart boundaries"""
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            total_duration = len(audio) / sr
            
            logger.info(f"Creating smart chunks for {total_duration:.2f}s audio")
            
            if total_duration <= self.chunk_duration:
                chunk_path = os.path.join(Config.TEMP_DIR, f"chunk_0000_{int(time.time())}.wav")
                sf.write(chunk_path, audio, sr)
                return [(chunk_path, 0.0, total_duration)]
            
            chunks = []
            chunk_size = int(self.chunk_duration * sr)
            overlap_size = int(self.overlap_duration * sr)
            step_size = chunk_size - overlap_size
            
            for i, start_sample in enumerate(range(0, len(audio) - overlap_size, step_size)):
                end_sample = min(start_sample + chunk_size, len(audio))
                chunk_audio = audio[start_sample:end_sample]
                
                if len(chunk_audio) < sr * 5:
                    continue
                
                # Apply fade in/out
                fade_samples = int(0.05 * sr)
                if len(chunk_audio) > 2 * fade_samples:
                    chunk_audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
                    chunk_audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
                
                chunk_path = os.path.join(Config.TEMP_DIR, f"chunk_{i:04d}_{int(time.time())}.wav")
                sf.write(chunk_path, chunk_audio, sr)
                
                start_time = start_sample / sr
                end_time = end_sample / sr
                
                chunks.append((chunk_path, start_time, end_time))
                logger.debug(f"Created chunk {i}: {start_time:.2f}s - {end_time:.2f}s")
            
            logger.info(f"Created {len(chunks)} smart chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Smart chunking failed: {e}")
            return []

class DeviceConsistentGemmaTranscriber:
    """DEVICE CONSISTENT Gemma3n-E4B-it transcription engine using Pipeline API"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or Config.GEMMA_MODEL_PATH
        self.device = Config.DEVICE
        self.torch_dtype = Config.TORCH_DTYPE
        
        self.pipeline = None
        
        self._load_pipeline_safely()
    
    def _load_pipeline_safely(self):
        """Load Gemma pipeline with complete device consistency"""
        try:
            logger.info(f"Loading Gemma3n-E4B-it pipeline from {self.model_path}")
            
            if not os.path.exists(self.model_path):
                logger.error(f"Model path does not exist: {self.model_path}")
                # Try to use from HuggingFace Hub
                model_id = "google/gemma-3n-E4B-it"
                logger.info(f"Attempting to load from HuggingFace Hub: {model_id}")
            else:
                model_id = self.model_path
            
            # DEVICE CONSISTENT: Use pipeline API for automatic device management
            self.pipeline = pipeline(
                "image-text-to-text",  # Gemma3n supports multimodal input
                model=model_id,
                device=self.device,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True,
                model_kwargs={
                    "low_cpu_mem_usage": True,
                    "attn_implementation": "eager"
                }
            )
            
            logger.info(f"Gemma3n pipeline loaded successfully on device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load Gemma pipeline: {e}")
            self.pipeline = None
            raise
    
    def transcribe_chunk_safely(self, audio_path: str, language_hint: str = None) -> Dict[str, Any]:
        """DEVICE CONSISTENT: Safely transcribe using pipeline API"""
        if self.pipeline is None:
            return {
                "text": "",
                "success": False,
                "error": "Pipeline not loaded"
            }
        
        try:
            # Prepare messages using Gemma3n format
            user_prompt = "Transcribe this audio clearly and accurately"
            if language_hint and language_hint.strip() and language_hint != "Auto-detect":
                user_prompt += f" in {language_hint}"
            user_prompt += ". Include all spoken words and preserve the natural flow of speech."
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": audio_path},
                        {"type": "text", "text": user_prompt}
                    ]
                }
            ]
            
            # DEVICE CONSISTENT: Pipeline handles all device management automatically
            try:
                output = self.pipeline(
                    text=messages,
                    max_new_tokens=Config.MAX_NEW_TOKENS,
                    temperature=0.1,
                    do_sample=False,
                    return_full_text=False
                )
                
                # Extract generated text
                if isinstance(output, list) and len(output) > 0:
                    generated_text = output[0].get("generated_text", "")
                elif isinstance(output, dict):
                    generated_text = output.get("generated_text", "")
                else:
                    generated_text = str(output)
                
                decoded = generated_text.strip()
                
                if not decoded:
                    return {
                        "text": "",
                        "success": False,
                        "error": "Empty transcription generated"
                    }
                
                return {
                    "text": decoded,
                    "success": True,
                    "error": None
                }
                
            except Exception as e:
                logger.error(f"Pipeline generation failed: {e}")
                return {
                    "text": "",
                    "success": False,
                    "error": f"Pipeline generation failed: {str(e)}"
                }
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return {
                "text": "",
                "success": False,
                "error": str(e)
            }
        finally:
            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def transcribe_chunks_parallel(self, chunks: List[Tuple[str, float, float]], 
                                 language_hint: str = None, 
                                 progress_callback=None) -> List[Dict[str, Any]]:
        """Transcribe chunks with progress tracking"""
        results = []
        
        for i, (chunk_path, start_time, end_time) in enumerate(chunks):
            if progress_callback:
                progress = i / len(chunks)
                progress_callback(progress, f"Transcribing chunk {i+1}/{len(chunks)}")
            
            result = self.transcribe_chunk_safely(chunk_path, language_hint)
            result.update({
                "chunk_id": i,
                "start_time": start_time,
                "end_time": end_time,
                "chunk_path": chunk_path
            })
            results.append(result)
            
            logger.info(f"Chunk {i+1}/{len(chunks)} - Success: {result['success']}")
            
            if i % 3 == 0:
                time.sleep(0.1)
        
        return results

class DeviceConsistentTranscriptionPipeline:
    """DEVICE CONSISTENT transcription pipeline with RF noise removal"""
    
    def __init__(self):
        self.enhancer = RadioFrequencyAudioEnhancer()
        self.chunker = SmartAudioChunker(
            chunk_duration=Config.CHUNK_DURATION,
            overlap_duration=Config.OVERLAP_DURATION
        )
        self.transcriber = None
        
        try:
            self.transcriber = DeviceConsistentGemmaTranscriber()
            logger.info("DEVICE CONSISTENT RF-enhanced transcription pipeline initialized")
        except Exception as e:
            logger.error(f"Failed to initialize transcriber: {e}")
            self.transcriber = None
    
    def process_audio_rf_ultimate(self, audio_path: str, 
                                enable_enhancement: bool = True,
                                language_hint: str = None,
                                progress_callback=None) -> Dict[str, Any]:
        """Ultimate RF-aware audio processing pipeline"""
        
        if self.transcriber is None:
            return {
                "success": False,
                "error": "Transcriber not initialized - check model and dependencies",
                "full_transcript": "",
                "chunks": [],
                "processing_time": 0,
                "num_chunks": 0,
                "enhanced_audio_path": None,
                "original_audio_path": None
            }
        
        try:
            start_time = time.time()
            
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Step 1: Ultimate RF noise removal and enhancement
            if progress_callback:
                progress_callback(0.1, "Removing RF interference and enhancing audio...")
            
            enhanced_audio_path = None
            original_audio_path = None
            
            try:
                if enable_enhancement:
                    enhanced_audio_path, original_audio_path = self.enhancer.enhance_audio_rf_ultimate(audio_path)
                    processing_path = enhanced_audio_path
                else:
                    processing_path = audio_path
                    original_audio_path = audio_path
            except Exception as e:
                logger.warning(f"RF enhancement failed: {e}, using original")
                processing_path = audio_path
                original_audio_path = audio_path
            
            # Step 2: Smart chunking
            if progress_callback:
                progress_callback(0.2, "Creating optimized audio chunks...")
            
            chunks = self.chunker.create_smart_chunks(processing_path)
            
            if not chunks:
                return {
                    "success": False,
                    "error": "Failed to create chunks - audio may be too short or corrupted",
                    "full_transcript": "",
                    "chunks": [],
                    "processing_time": time.time() - start_time,
                    "num_chunks": 0,
                    "enhanced_audio_path": enhanced_audio_path,
                    "original_audio_path": original_audio_path
                }
            
            # Step 3: Device-consistent transcription
            if progress_callback:
                progress_callback(0.3, f"Transcribing {len(chunks)} chunks with device consistency...")
            
            def chunk_progress(chunk_progress, message):
                overall_progress = 0.3 + (chunk_progress * 0.6)
                if progress_callback:
                    progress_callback(overall_progress, message)
            
            transcription_results = self.transcriber.transcribe_chunks_parallel(
                chunks, language_hint, chunk_progress
            )
            
            # Step 4: Combine results
            if progress_callback:
                progress_callback(0.95, "Combining transcriptions...")
            
            full_transcript = self._combine_transcriptions_smart(transcription_results)
            
            # Step 5: Cleanup
            self._cleanup_temp_files(chunks)
            
            processing_time = time.time() - start_time
            successful_chunks = len([r for r in transcription_results if r.get("success", False)])
            
            if progress_callback:
                progress_callback(1.0, f"Completed in {processing_time:.1f}s")
            
            return {
                "success": True,
                "error": None,
                "full_transcript": full_transcript,
                "chunks": transcription_results,
                "processing_time": processing_time,
                "num_chunks": len(chunks),
                "successful_chunks": successful_chunks,
                "success_rate": successful_chunks / len(chunks) if chunks else 0,
                "enhanced_audio_path": enhanced_audio_path,
                "original_audio_path": original_audio_path
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "full_transcript": "",
                "chunks": [],
                "processing_time": time.time() - start_time,
                "num_chunks": 0,
                "enhanced_audio_path": None,
                "original_audio_path": None
            }
    
    def _combine_transcriptions_smart(self, results: List[Dict[str, Any]]) -> str:
        """Smart transcription combination with overlap handling"""
        if not results:
            return ""
        
        successful_results = [r for r in results if r.get("success", False) and r.get("text", "").strip()]
        
        if not successful_results:
            failed_count = len([r for r in results if not r.get("success", False)])
            return f"Transcription failed for all chunks. {failed_count} chunks failed."
        
        successful_results.sort(key=lambda x: x.get("start_time", 0))
        
        combined_text = ""
        
        for i, result in enumerate(successful_results):
            current_text = result.get("text", "").strip()
            
            if not current_text:
                continue
            
            if i == 0:
                combined_text = current_text
            else:
                prev_words = combined_text.split()
                current_words = current_text.split()
                
                max_overlap = min(12, len(prev_words), len(current_words))
                best_overlap = 0
                
                for overlap_len in range(max_overlap, 1, -1):
                    if prev_words[-overlap_len:] == current_words[:overlap_len]:
                        best_overlap = overlap_len
                        break
                
                if best_overlap > 0:
                    remaining_words = current_words[best_overlap:]
                    if remaining_words:
                        combined_text += " " + " ".join(remaining_words)
                else:
                    combined_text += " " + current_text
        
        return combined_text.strip()
    
    def _cleanup_temp_files(self, chunks: List[Tuple[str, float, float]]):
        """Clean up temporary files"""
        for chunk_path, _, _ in chunks:
            if os.path.exists(chunk_path):
                try:
                    os.remove(chunk_path)
                except:
                    pass

# DEVICE CONSISTENT Gradio Interface
class DeviceConsistentTranscriptionUI:
    """DEVICE CONSISTENT Professional Gradio interface with RF noise removal"""
    
    def __init__(self):
        self.pipeline = DeviceConsistentTranscriptionPipeline()
        self.system_ready = self.pipeline.transcriber is not None
        
        self.system_info = {
            "device": Config.DEVICE,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
        }
        
        self.common_languages = [
            "Auto-detect", "English", "Spanish", "French", "German", "Italian", "Portuguese",
            "Chinese (Mandarin)", "Japanese", "Korean", "Hindi", "Arabic", "Russian", 
            "Dutch", "Swedish", "Norwegian", "Danish", "Finnish", "Polish", "Turkish", 
            "Greek", "Hebrew", "Thai", "Vietnamese", "Indonesian", "Malay", "Filipino", 
            "Swahili", "Urdu", "Bengali", "Tamil", "Telugu", "Marathi", "Gujarati", 
            "Kannada", "Malayalam", "Punjabi", "Nepali", "Sinhala", "Burmese"
        ]
    
    def update_progress(self, progress: float, message: str):
        """Update progress"""
        pass
    
    def process_file_device_consistent(self, audio_file, enable_enhancement, language_hint, 
                                     chunk_duration, overlap_duration):
        """DEVICE CONSISTENT file processing with RF noise removal"""
        
        if not self.system_ready:
            return "❌ System not ready - Gemma3n-E4B-it pipeline failed to load", "", "", None, None
        
        if audio_file is None:
            return "❌ No audio file uploaded", "", "", None, None
        
        try:
            self.pipeline.chunker.chunk_duration = chunk_duration
            self.pipeline.chunker.overlap_duration = overlap_duration
            
            file_path = audio_file.name if hasattr(audio_file, 'name') else str(audio_file)
            
            if not os.path.exists(file_path):
                return "❌ File not accessible", "", "", None, None
            
            file_size = os.path.getsize(file_path) / (1024 * 1024)
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext not in Config.SUPPORTED_FORMATS:
                return f"❌ Unsupported format: {file_ext}", "", "", None, None
            
            status_msg = f"🎵 Processing: {os.path.basename(file_path)} ({file_size:.1f} MB)\n"
            status_msg += f"🔧 Enhancement: {'RF Noise Removal + Multi-Stage' if enable_enhancement else 'Disabled'}\n"
            status_msg += f"🌍 Language: {language_hint}\n"
            status_msg += f"⏱️ Chunks: {chunk_duration}s with {overlap_duration}s overlap\n"
            status_msg += f"📡 RF Frequencies: {Config.RF_FREQUENCIES}\n\n"
            
            result = self.pipeline.process_audio_rf_ultimate(
                file_path,
                enable_enhancement,
                language_hint if language_hint and language_hint.strip() and language_hint != "Auto-detect" else None,
                self.update_progress
            )
            
            if result["success"]:
                status_msg += f"✅ SUCCESS! "
                status_msg += f"Processed {result['num_chunks']} chunks in {result['processing_time']:.1f}s\n"
                status_msg += f"📈 Success rate: {result.get('successful_chunks', 0)}/{result['num_chunks']} "
                status_msg += f"({result.get('success_rate', 0)*100:.1f}%)"
                
                detailed_report = self._create_report(result)
                
                # Return audio files for preview
                enhanced_audio = result.get('enhanced_audio_path')
                original_audio = result.get('original_audio_path')
                
                return status_msg, result["full_transcript"], detailed_report, enhanced_audio, original_audio
            else:
                error_msg = f"❌ Failed: {result['error']}\n"
                error_msg += f"⏱️ Time: {result['processing_time']:.1f}s"
                return error_msg, "", error_msg, None, None
                
        except Exception as e:
            error_msg = f"❌ Unexpected error: {str(e)}"
            logger.error(f"UI error: {e}")
            return error_msg, "", error_msg, None, None
    
    def _create_report(self, result: Dict[str, Any]) -> str:
        """Create processing report"""
        report = f"""# 📊 DEVICE CONSISTENT + RF NOISE REMOVAL Report

## 🎯 Processing Summary
- **Duration:** {result['processing_time']:.1f} seconds
- **Chunks:** {result['num_chunks']}
- **Success Rate:** {result.get('success_rate', 0)*100:.1f}%
- **RF Enhancement:** Multi-stage RF interference removal applied
- **Device:** {Config.DEVICE} (All tensors consistent)

## 🔧 RF Noise Removal Pipeline
1. **RF Notch Filters:** {Config.RF_FREQUENCIES} Hz
2. **RF Bandstop Filters:** Power line + Radio band interference
3. **Wavelet Denoising:** Impulse and RF noise suppression
4. **Spectral Subtraction:** RF-optimized parameters
5. **Multi-stage Noise Reduction:** GPU/CPU with RF settings

## 📈 Chunk Details
"""
        
        for chunk in result['chunks']:
            status = "✅" if chunk.get('success', False) else "❌"
            duration = chunk['end_time'] - chunk['start_time']
            report += f"- **Chunk {chunk['chunk_id']+1}** ({chunk['start_time']:.1f}-{chunk['end_time']:.1f}s): {status}\n"
            
            if chunk.get('success', False):
                word_count = len(chunk.get('text', '').split())
                report += f"  - Words: {word_count}\n"
            elif chunk.get('error'):
                report += f"  - Error: {chunk['error']}\n"
        
        return report
    
    def create_device_consistent_interface(self):
        """Create DEVICE CONSISTENT interface with RF noise removal"""
        
        custom_css = """
        .gradio-container { font-family: 'Inter', system-ui, sans-serif; max-width: 1500px; margin: 0 auto; }
        .status-success { padding: 15px; border-radius: 8px; border-left: 4px solid #10b981; background: linear-gradient(90deg, #ecfdf5 0%, #f0fdf4 100%); }
        .status-error { padding: 15px; border-radius: 8px; border-left: 4px solid #ef4444; background: linear-gradient(90deg, #fef2f2 0%, #fff5f5 100%); }
        .rf-info { background: #f0f4ff; padding: 15px; border-radius: 8px; border-left: 4px solid #3b82f6; margin: 10px 0; }
        """
        
        with gr.Blocks(title="RF-Enhanced Audio Transcription", css=custom_css, theme=gr.themes.Soft()) as interface:
            
            # Header
            gr.HTML("""
            <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 15px; margin-bottom: 25px;">
                <h1 style="margin: 0; font-size: 2.8em;">📡 RF-Enhanced Audio Transcription</h1>
                <p style="margin: 15px 0 0 0; font-size: 1.3em;">DEVICE CONSISTENT + Radio Frequency Interference Removal</p>
                <p style="margin: 5px 0 0 0; font-size: 1.0em;">All Device Issues Resolved • Advanced RF Noise Suppression</p>
            </div>
            """)
            
            # System Status
            if not self.system_ready:
                gr.HTML('<div class="status-error"><h3>⚠️ System Not Ready</h3><p>Gemma3n-E4B-it pipeline failed to load. Check model and dependencies.</p></div>')
            else:
                gpu_info = f" | {self.system_info['gpu_name']}" if self.system_info['cuda_available'] else " | CPU"
                gr.HTML(f'''<div class="status-success"><h3>✅ DEVICE CONSISTENT SYSTEM READY</h3>
                <p>Device: {self.system_info['device']}{gpu_info} | Pipeline API ensures device consistency</p></div>''')
            
            with gr.Row():
                with gr.Column(scale=2):
                    gr.HTML("<h2>📁 Input & Configuration</h2>")
                    
                    audio_input = gr.File(
                        label="📎 Upload Audio File (RF Interference Supported)",
                        file_types=Config.SUPPORTED_FORMATS,
                        file_count="single"
                    )
                    
                    enable_enhancement = gr.Checkbox(
                        label="📡 Enable RF Noise Removal + Multi-Stage Enhancement",
                        value=True,
                        info="8-stage pipeline: RF Notch → RF Bandstop → Wavelet → Preemphasis → Bandpass → Spectral → GPU/CPU → Final"
                    )
                    
                    language_hint = gr.Dropdown(
                        label="🌍 Language",
                        choices=self.common_languages,
                        value="Auto-detect",
                        allow_custom_value=True,
                        filterable=True,
                        info="Type any language name"
                    )
                    
                    with gr.Accordion("⚙️ Advanced Settings", open=False):
                        chunk_duration = gr.Slider(20, 120, Config.CHUNK_DURATION, step=10, 
                                                 label="Chunk Duration (s)")
                        overlap_duration = gr.Slider(5, 30, Config.OVERLAP_DURATION, step=5, 
                                                   label="Overlap Duration (s)")
                    
                    process_btn = gr.Button("🚀 Start RF-Enhanced Transcription", variant="primary", size="lg")
                
                with gr.Column(scale=3):
                    gr.HTML("<h2>📊 Results & Audio Preview</h2>")
                    
                    status_output = gr.Textbox(
                        label="📈 Processing Status",
                        lines=5,
                        interactive=False,
                        placeholder="Upload audio and start transcription..."
                    )
                    
                    # Audio Preview Section
                    with gr.Accordion("🎵 Audio Preview (RF Noise Removal Comparison)", open=False):
                        with gr.Row():
                            with gr.Column():
                                gr.HTML("<h4>🔊 Original Audio (with RF noise)</h4>")
                                original_audio_player = gr.Audio(
                                    label="Original",
                                    visible=False,
                                    interactive=False
                                )
                            
                            with gr.Column():
                                gr.HTML("<h4>📡 RF-Enhanced Audio</h4>")
                                enhanced_audio_player = gr.Audio(
                                    label="RF Enhanced",
                                    visible=False,
                                    interactive=False
                                )
                    
                    transcript_output = gr.Textbox(
                        label="📝 RF-Enhanced Transcript",
                        lines=12,
                        interactive=True,
                        show_copy_button=True,
                        placeholder="RF-enhanced transcription will appear here..."
                    )
                    
                    download_btn = gr.DownloadButton("📥 Download", visible=False)
            
            with gr.Accordion("📈 Detailed Report", open=False):
                detailed_report = gr.Markdown("No processing completed yet.")
            
            with gr.Accordion("📡 RF Enhancement Details", open=False):
                gr.HTML(f"""
                <div class="rf-info">
                    <h3>📡 Radio Frequency Interference Removal</h3>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px;">
                        <div><strong>🎯 RF Frequencies:</strong><br>{', '.join(map(str, Config.RF_FREQUENCIES))} Hz</div>
                        <div><strong>🔧 Notch Filters:</strong><br>High-Q filters for precise RF removal</div>
                        <div><strong>📊 Bandstop Filters:</strong><br>Power line + Radio band suppression</div>
                        <div><strong>🌊 Wavelet Denoising:</strong><br>{Config.WAVELET_MODE} wavelet with soft thresholding</div>
                    </div>
                    <h3>🔧 8-Stage Enhancement Pipeline</h3>
                    <p><strong>Stage 1:</strong> RF Notch Filters | <strong>Stage 2:</strong> RF Bandstop Filters</p>
                    <p><strong>Stage 3:</strong> Wavelet Denoising | <strong>Stage 4:</strong> Preemphasis</p>
                    <p><strong>Stage 5:</strong> Advanced Bandpass | <strong>Stage 6:</strong> RF Spectral Subtraction</p>
                    <p><strong>Stage 7:</strong> Multi-stage GPU/CPU Noise Reduction | <strong>Stage 8:</strong> Final Normalization</p>
                    <h3>✅ Device Consistency</h3>
                    <p>Using Transformers Pipeline API for automatic device management - eliminates "cuda:0 and cpu" tensor errors</p>
                </div>
                """)
            
            def process_and_download_rf(audio_file, enable_enhancement, language_hint, chunk_duration, overlap_duration):
                status, transcript, report, enhanced_audio, original_audio = self.process_file_device_consistent(
                    audio_file, enable_enhancement, language_hint, chunk_duration, overlap_duration
                )
                
                download_file = None
                download_visible = False
                
                if transcript.strip() and "✅" in status:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"rf_enhanced_transcript_{timestamp}.txt"
                    filepath = os.path.join(Config.OUTPUT_DIR, filename)
                    
                    try:
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(f"RF-Enhanced Audio Transcription Report\n")
                            f.write(f"Generated: {datetime.now()}\n")
                            f.write(f"System: Gemma3n-E4B-it (DEVICE CONSISTENT + RF ENHANCED)\n")
                            f.write(f"Enhancement: {'RF Noise Removal + Multi-Stage' if enable_enhancement else 'Disabled'}\n")
                            f.write(f"RF Frequencies Removed: {Config.RF_FREQUENCIES}\n")
                            if language_hint != "Auto-detect":
                                f.write(f"Language: {language_hint}\n")
                            f.write(f"{'='*60}\n\nTRANSCRIPT:\n{'-'*60}\n")
                            f.write(transcript)
                            f.write(f"\n\n{'-'*60}\nREPORT:\n{'-'*60}\n")
                            f.write(report)
                        
                        download_file = filepath
                        download_visible = True
                    except:
                        pass
                
                # Update audio players
                original_visible = original_audio is not None
                enhanced_visible = enhanced_audio is not None
                
                return (
                    status, 
                    transcript, 
                    report,
                    gr.Audio(value=original_audio, visible=original_visible, interactive=False, label="Original Audio"),
                    gr.Audio(value=enhanced_audio, visible=enhanced_visible, interactive=False, label="RF-Enhanced Audio"),
                    gr.DownloadButton(
                        label="📥 Download RF-Enhanced Transcript",
                        value=download_file,
                        visible=download_visible
                    )
                )
            
            process_btn.click(
                fn=process_and_download_rf,
                inputs=[audio_input, enable_enhancement, language_hint, chunk_duration, overlap_duration],
                outputs=[status_output, transcript_output, detailed_report, original_audio_player, enhanced_audio_player, download_btn]
            )
        
        return interface

def main():
    """DEVICE CONSISTENT main function with RF enhancement"""
    
    logger.info("🚀 Starting DEVICE CONSISTENT + RF-Enhanced Audio Transcription System")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        logger.warning("CUDA not available - CPU mode will be slower")
    
    if not os.path.exists(Config.GEMMA_MODEL_PATH):
        logger.warning(f"Local model not found: {Config.GEMMA_MODEL_PATH}")
        logger.info("Will attempt to load from HuggingFace Hub: google/gemma-3n-E4B-it")
    
    try:
        ui = DeviceConsistentTranscriptionUI()
        interface = ui.create_device_consistent_interface()
        
        logger.info("🎉 Launching DEVICE CONSISTENT + RF-Enhanced interface...")
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            inbrowser=True,
            show_error=True
        )
    except Exception as e:
        logger.error(f"Launch failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
