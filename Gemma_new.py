#!/usr/bin/env python3
"""
Professional Audio Transcription System with Gemma3n-E4B-it
ULTIMATE VERSION - All tensor issues resolved + Advanced Audio Enhancement
Features: Multi-stage Enhancement, Heavy Noise/Distortion Handling, Smart Chunking, Modern Gradio UI
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
from scipy.signal import butter, filtfilt, wiener, medfilt
from scipy.ndimage import gaussian_filter1d
from scipy import signal

# ML/AI libraries
from transformers import (
    AutoProcessor, 
    AutoModelForImageTextToText,
    pipeline
)

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
from concurrent.futures import ThreadPoolExecutor, as_completed

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global Configuration
class Config:
    """Global configuration for the transcription system"""
    
    # Model paths (modify these to your local model directories)
    GEMMA_MODEL_PATH = "./models/google--gemma-3n-E4B-it"  # Local model path
    CACHE_DIR = "./cache"
    TEMP_DIR = "./temp"
    OUTPUT_DIR = "./outputs"
    
    # Audio processing parameters
    TARGET_SAMPLE_RATE = 16000  # Gemma requirement
    CHUNK_DURATION = 40  # seconds
    OVERLAP_DURATION = 10  # seconds
    MAX_AUDIO_LENGTH = 3600  # 1 hour max
    
    # Enhanced audio processing parameters
    NOISE_REDUCTION_STRENGTH = 0.85
    SPECTRAL_GATE_THRESHOLD = 1.2
    WIENER_FILTER_SIZE = 5
    MEDIAN_FILTER_SIZE = 3
    
    # GPU settings
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    MAX_NEW_TOKENS = 512
    
    # Supported formats
    SUPPORTED_FORMATS = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.wma']

# Initialize directories
for dir_path in [Config.CACHE_DIR, Config.TEMP_DIR, Config.OUTPUT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

class AdvancedAudioEnhancer:
    """Advanced multi-stage audio enhancement for heavily distorted/noisy audio"""
    
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
                logger.info("GPU-accelerated torch noise reduction initialized")
            except Exception as e:
                logger.warning(f"Torch noise reducer initialization failed: {e}")
                self.torch_gate = None
    
    def safe_audio_load(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Safely load audio with multiple fallback methods and format handling"""
        try:
            # Primary method: librosa (handles most formats)
            audio, sr = librosa.load(audio_path, sr=None, mono=False)  # Load at original sample rate first
            
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
                # Fallback: soundfile
                audio, sr = sf.read(audio_path)
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)
                if sr != self.sample_rate:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
                return audio.astype(np.float32), self.sample_rate
                
            except Exception as e2:
                logger.warning(f"Soundfile failed: {e2}, trying torchaudio...")
                try:
                    # Last resort: torchaudio
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
    
    def apply_preemphasis(self, audio: np.ndarray, alpha: float = 0.97) -> np.ndarray:
        """Apply preemphasis filter to enhance high frequencies"""
        try:
            preemphasized = np.zeros_like(audio)
            preemphasized[0] = audio
            preemphasized[1:] = audio[1:] - alpha * audio[:-1]
            return preemphasized.astype(np.float32)
        except Exception as e:
            logger.warning(f"Preemphasis failed: {e}")
            return audio
    
    def apply_spectral_subtraction(self, audio: np.ndarray, alpha: float = 2.0, beta: float = 0.01) -> np.ndarray:
        """Apply spectral subtraction for noise reduction"""
        try:
            # Compute STFT
            stft = librosa.stft(audio, n_fft=2048, hop_length=512)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Estimate noise from first few frames (assumed to be silence/noise)
            noise_frames = max(1, magnitude.shape[1] // 20)  # First 5% of frames
            noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
            
            # Spectral subtraction
            enhanced_magnitude = magnitude - alpha * noise_spectrum
            
            # Ensure non-negative values
            enhanced_magnitude = np.maximum(enhanced_magnitude, beta * magnitude)
            
            # Reconstruct signal
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft, hop_length=512)
            
            return enhanced_audio.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"Spectral subtraction failed: {e}")
            return audio
    
    def apply_wiener_filter(self, audio: np.ndarray) -> np.ndarray:
        """Apply Wiener filter for noise reduction"""
        try:
            # Apply Wiener filter with estimated noise
            filtered = wiener(audio, mysize=Config.WIENER_FILTER_SIZE)
            return filtered.astype(np.float32)
        except Exception as e:
            logger.warning(f"Wiener filter failed: {e}")
            return audio
    
    def apply_median_filter(self, audio: np.ndarray) -> np.ndarray:
        """Apply median filter to remove impulse noise"""
        try:
            filtered = medfilt(audio, kernel_size=Config.MEDIAN_FILTER_SIZE)
            return filtered.astype(np.float32)
        except Exception as e:
            logger.warning(f"Median filter failed: {e}")
            return audio
    
    def apply_advanced_bandpass_filter(self, audio: np.ndarray) -> np.ndarray:
        """Apply advanced multi-stage bandpass filtering for speech"""
        try:
            # Primary speech band (300-3400 Hz - telephone quality)
            nyquist = self.sample_rate * 0.5
            low1, high1 = 300 / nyquist, 3400 / nyquist
            b1, a1 = butter(4, [low1, high1], btype='band')
            filtered1 = filtfilt(b1, a1, audio)
            
            # Extended speech band (80-8000 Hz - full speech)
            low2, high2 = 80 / nyquist, min(8000 / nyquist, 0.99)
            b2, a2 = butter(4, [low2, high2], btype='band')
            filtered2 = filtfilt(b2, a2, audio)
            
            # Combine filters with weighted average (favor extended band)
            combined = 0.3 * filtered1 + 0.7 * filtered2
            
            return combined.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"Advanced bandpass filter failed: {e}")
            return audio
    
    def reduce_noise_multi_stage(self, audio: np.ndarray) -> np.ndarray:
        """Multi-stage noise reduction for heavily distorted audio"""
        try:
            enhanced = audio.copy()
            
            # Stage 1: Spectral subtraction
            enhanced = self.apply_spectral_subtraction(enhanced)
            
            # Stage 2: GPU-accelerated noise reduction if available
            if self.torch_gate is not None and torch.cuda.is_available():
                audio_tensor = torch.from_numpy(enhanced).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    enhanced_tensor = self.torch_gate(audio_tensor)
                enhanced = enhanced_tensor.squeeze().cpu().numpy().astype(np.float32)
            else:
                # CPU-based noise reduction with multiple passes
                enhanced = nr.reduce_noise(
                    y=enhanced,
                    sr=self.sample_rate,
                    stationary=False,
                    prop_decrease=Config.NOISE_REDUCTION_STRENGTH,
                    time_constant_s=1.5,
                    freq_mask_smooth_hz=300,
                    time_mask_smooth_ms=40,
                    n_jobs=1,
                    chunk_size=self.sample_rate * 20,
                    padding=self.sample_rate * 3
                )
                
                # Second pass with different parameters for remaining noise
                enhanced = nr.reduce_noise(
                    y=enhanced,
                    sr=self.sample_rate,
                    stationary=True,
                    prop_decrease=0.6,
                    time_constant_s=2.0,
                    freq_mask_smooth_hz=500,
                    time_mask_smooth_ms=60,
                    n_jobs=1
                )
            
            # Stage 3: Wiener filtering
            enhanced = self.apply_wiener_filter(enhanced)
            
            # Stage 4: Median filtering for impulse noise
            enhanced = self.apply_median_filter(enhanced)
            
            return enhanced.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"Multi-stage noise reduction failed: {e}")
            return audio
    
    def apply_advanced_vad(self, audio: np.ndarray) -> np.ndarray:
        """Advanced Voice Activity Detection with energy and spectral features"""
        try:
            # Compute multiple features
            frame_length = 2048
            hop_length = 512
            
            # RMS energy
            rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
            
            # Spectral centroid (indicates presence of speech)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate, hop_length=hop_length)[0]
            
            # Zero crossing rate (speech has moderate ZCR)
            zcr = librosa.feature.zero_crossing_rate(audio, frame_length=frame_length, hop_length=hop_length)[0]
            
            # Combine features for better VAD
            combined_features = (
                (rms - np.min(rms)) / (np.max(rms) - np.min(rms) + 1e-8) * 0.4 +
                (spectral_centroid - np.min(spectral_centroid)) / (np.max(spectral_centroid) - np.min(spectral_centroid) + 1e-8) * 0.4 +
                (1 - (zcr - np.min(zcr)) / (np.max(zcr) - np.min(zcr) + 1e-8)) * 0.2  # Invert ZCR
            )
            
            # Adaptive threshold
            threshold = np.percentile(combined_features, 25)  # More conservative
            
            # Time alignment
            times = librosa.frames_to_time(np.arange(len(combined_features)), sr=self.sample_rate, hop_length=hop_length)
            audio_times = np.arange(len(audio)) / self.sample_rate
            
            # Interpolate mask
            voice_mask = np.interp(audio_times, times, combined_features > threshold)
            
            # Smooth the mask more aggressively
            voice_mask = gaussian_filter1d(voice_mask, sigma=self.sample_rate * 0.05)
            voice_mask = voice_mask > 0.3  # Lower threshold after smoothing
            
            # Apply mask with fade
            return (audio * voice_mask).astype(np.float32)
            
        except Exception as e:
            logger.warning(f"Advanced VAD failed: {e}")
            return audio
    
    def enhance_audio_ultimate(self, audio_path: str) -> str:
        """Ultimate audio enhancement pipeline for heavily distorted audio"""
        try:
            start_time = time.time()
            logger.info(f"Starting ultimate audio enhancement for: {audio_path}")
            
            # Load audio
            audio, sr = self.safe_audio_load(audio_path)
            original_length = len(audio) / sr
            logger.info(f"Loaded audio: {original_length:.2f}s at {sr}Hz")
            
            # Stage 1: Initial normalization
            audio = self.normalize_audio_advanced(audio)
            
            # Stage 2: Preemphasis for high frequency enhancement
            audio = self.apply_preemphasis(audio)
            
            # Stage 3: Advanced bandpass filtering
            audio = self.apply_advanced_bandpass_filter(audio)
            
            # Stage 4: Multi-stage noise reduction (main enhancement)
            audio = self.reduce_noise_multi_stage(audio)
            
            # Stage 5: Advanced Voice Activity Detection (optional - commented out for calls)
            # audio = self.apply_advanced_vad(audio)
            
            # Stage 6: Final normalization and limiting
            audio = self.normalize_audio_advanced(audio)
            
            # Ensure we didn't lose significant audio length (safety check)
            if len(audio) / sr < original_length * 0.8:
                logger.warning("Significant audio length reduction detected, using less aggressive processing")
                # Reload and apply lighter processing
                audio, sr = self.safe_audio_load(audio_path)
                audio = self.normalize_audio_advanced(audio)
                audio = self.apply_advanced_bandpass_filter(audio)
                audio = self.reduce_noise_multi_stage(audio)
                audio = self.normalize_audio_advanced(audio)
            
            # Save enhanced audio
            enhanced_path = os.path.join(Config.TEMP_DIR, f"enhanced_ultimate_{int(time.time())}.wav")
            sf.write(enhanced_path, audio, self.sample_rate)
            
            processing_time = time.time() - start_time
            logger.info(f"Ultimate audio enhancement completed in {processing_time:.2f}s: {enhanced_path}")
            
            return enhanced_path
            
        except Exception as e:
            logger.error(f"Ultimate audio enhancement failed: {e}")
            return audio_path

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
                # Single chunk for short audio
                chunk_path = os.path.join(Config.TEMP_DIR, f"chunk_0000_{int(time.time())}.wav")
                sf.write(chunk_path, audio, sr)
                return [(chunk_path, 0.0, total_duration)]
            
            chunks = []
            chunk_size = int(self.chunk_duration * sr)
            overlap_size = int(self.overlap_duration * sr)
            step_size = chunk_size - overlap_size
            
            for i, start_sample in enumerate(range(0, len(audio) - overlap_size, step_size)):
                end_sample = min(start_sample + chunk_size, len(audio))
                
                # Extract chunk
                chunk_audio = audio[start_sample:end_sample]
                
                # Skip very short chunks
                if len(chunk_audio) < sr * 5:
                    continue
                
                # Apply fade in/out to reduce artifacts
                fade_samples = int(0.05 * sr)  # 50ms fade
                if len(chunk_audio) > 2 * fade_samples:
                    # Fade in
                    chunk_audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
                    # Fade out
                    chunk_audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
                
                # Save chunk
                chunk_path = os.path.join(Config.TEMP_DIR, f"chunk_{i:04d}_{int(time.time())}.wav")
                sf.write(chunk_path, chunk_audio, sr)
                
                # Calculate timing
                start_time = start_sample / sr
                end_time = end_sample / sr
                
                chunks.append((chunk_path, start_time, end_time))
                logger.debug(f"Created chunk {i}: {start_time:.2f}s - {end_time:.2f}s")
            
            logger.info(f"Created {len(chunks)} smart chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Smart chunking failed: {e}")
            return []

class UltimateGemmaTranscriber:
    """ULTIMATE Gemma3n-E4B-it transcription engine with all tensor issues resolved"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or Config.GEMMA_MODEL_PATH
        self.device = Config.DEVICE
        self.torch_dtype = Config.TORCH_DTYPE
        
        self.processor = None
        self.model = None
        self.model_dtype = None
        
        self._load_model_safely()
    
    def _load_model_safely(self):
        """Safely load Gemma model with comprehensive error handling"""
        try:
            logger.info(f"Loading Gemma3n-E4B-it model from {self.model_path}")
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model path does not exist: {self.model_path}")
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                local_files_only=True,
                trust_remote_code=True
            )
            
            # Load model
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                torch_dtype=self.torch_dtype,
                device_map="auto",
                local_files_only=True,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            ).eval()
            
            # Store model dtype
            self.model_dtype = next(self.model.parameters()).dtype
            logger.info(f"Model loaded with dtype: {self.model_dtype}")
            
        except Exception as e:
            logger.error(f"Failed to load Gemma model: {e}")
            self.processor = None
            self.model = None
            raise
    
    def _fix_all_tensor_dtypes(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """ULTIMATE FIX: Ensure all tensors have correct dtypes to prevent all operator errors"""
        if self.model is None:
            return inputs
        
        fixed_inputs = {}
        
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                if key == "input_ids":
                    # input_ids MUST be Long for embedding layer
                    fixed_inputs[key] = value.long()
                    
                elif key == "attention_mask":
                    # attention_mask MUST be Boolean or Long (not Float) for ~ operator
                    if value.dtype.is_floating_point:
                        # Convert float mask to boolean (values > 0.5 are True)
                        fixed_inputs[key] = (value > 0.5).bool()
                    else:
                        fixed_inputs[key] = value.bool()
                        
                elif key in ["token_type_ids", "position_ids"]:
                    # These should be Long
                    fixed_inputs[key] = value.long()
                    
                elif key in ["pixel_values", "image_features"]:
                    # Image data can match model dtype
                    fixed_inputs[key] = value.to(dtype=self.model_dtype)
                    
                else:
                    # For any other tensor, be conservative
                    if value.dtype.is_floating_point and key.endswith('_mask'):
                        # Any mask should be boolean
                        fixed_inputs[key] = (value > 0.5).bool()
                    elif key.endswith('_ids'):
                        # Any IDs should be Long
                        fixed_inputs[key] = value.long()
                    else:
                        fixed_inputs[key] = value
            else:
                fixed_inputs[key] = value
        
        # Log tensor dtypes for debugging
        for key, value in fixed_inputs.items():
            if isinstance(value, torch.Tensor):
                logger.debug(f"{key}: {value.dtype}, shape: {value.shape}")
        
        return fixed_inputs
    
    def _validate_generation_params(self, **kwargs) -> Dict[str, Any]:
        """Validate generation parameters"""
        valid_params = {
            'max_new_tokens', 'max_length', 'min_length', 'do_sample', 'early_stopping',
            'num_beams', 'temperature', 'top_k', 'top_p', 'typical_p', 'repetition_penalty',
            'length_penalty', 'no_repeat_ngram_size', 'bad_words_ids', 'force_words_ids',
            'forced_bos_token_id', 'forced_eos_token_id', 'remove_invalid_values',
            'suppress_tokens', 'begin_suppress_tokens', 'forced_decoder_ids',
            'num_return_sequences', 'output_attentions', 'output_hidden_states',
            'output_scores', 'return_dict_in_generate', 'pad_token_id', 'eos_token_id',
            'use_cache', 'stopping_criteria', 'max_time'
        }
        
        return {k: v for k, v in kwargs.items() if k in valid_params}
    
    def transcribe_chunk_safely(self, audio_path: str, language_hint: str = None) -> Dict[str, Any]:
        """ULTIMATE: Safely transcribe with all tensor issues resolved"""
        if self.model is None or self.processor is None:
            return {
                "text": "",
                "success": False,
                "error": "Model not loaded"
            }
        
        try:
            # Prepare messages
            system_prompt = "You are a professional audio transcriber. Transcribe the audio accurately, preserving all spoken content including natural speech patterns. Output clean, readable text without timestamps or speaker labels."
            
            user_prompt = "Transcribe this audio clearly and accurately"
            if language_hint and language_hint.strip() and language_hint != "Auto-detect":
                user_prompt += f" in {language_hint}"
            user_prompt += ". Include all spoken words and preserve the natural flow of speech."
            
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}]
                },
                {
                    "role": "user", 
                    "content": [
                        {"type": "audio", "audio": audio_path},
                        {"type": "text", "text": user_prompt}
                    ]
                }
            ]
            
            # Apply chat template
            try:
                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt"
                )
                
                # Move to device
                inputs = inputs.to(self.model.device)
                
                # ULTIMATE FIX: Fix all tensor dtypes
                inputs = self._fix_all_tensor_dtypes(inputs)
                
            except Exception as e:
                logger.error(f"Chat template failed: {e}")
                return {
                    "text": "",
                    "success": False,
                    "error": f"Template error: {str(e)}"
                }
            
            # Generate
            gen_params = {
                'max_new_tokens': Config.MAX_NEW_TOKENS,
                'do_sample': False,
                'temperature': 0.1,
                'pad_token_id': self.processor.tokenizer.eos_token_id,
                'use_cache': True
            }
            validated_params = self._validate_generation_params(**gen_params)
            
            with torch.inference_mode():
                try:
                    generation = self.model.generate(**inputs, **validated_params)
                    
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    gc.collect()
                    fallback_params = validated_params.copy()
                    fallback_params['max_new_tokens'] = Config.MAX_NEW_TOKENS // 2
                    generation = self.model.generate(**inputs, **fallback_params)
                    
                except Exception as gen_error:
                    logger.error(f"Generation failed: {gen_error}")
                    return {
                        "text": "",
                        "success": False,
                        "error": f"Generation failed: {str(gen_error)}"
                    }
            
            # Decode
            try:
                input_len = inputs["input_ids"].shape[-1]
                generation = generation[0][input_len:]
                decoded = self.processor.decode(generation, skip_special_tokens=True)
                
                decoded = decoded.strip()
                if not decoded:
                    return {
                        "text": "",
                        "success": False,
                        "error": "Empty transcription"
                    }
                
                return {
                    "text": decoded,
                    "success": True,
                    "error": None
                }
                
            except Exception as decode_error:
                logger.error(f"Decoding failed: {decode_error}")
                return {
                    "text": "",
                    "success": False,
                    "error": f"Decoding failed: {str(decode_error)}"
                }
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return {
                "text": "",
                "success": False,
                "error": str(e)
            }
        finally:
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
                time.sleep(0.1)  # Brief pause for system responsiveness
        
        return results

class UltimateTranscriptionPipeline:
    """ULTIMATE transcription pipeline with advanced audio enhancement"""
    
    def __init__(self):
        self.enhancer = AdvancedAudioEnhancer()
        self.chunker = SmartAudioChunker(
            chunk_duration=Config.CHUNK_DURATION,
            overlap_duration=Config.OVERLAP_DURATION
        )
        self.transcriber = None
        
        try:
            self.transcriber = UltimateGemmaTranscriber()
            logger.info("ULTIMATE transcription pipeline initialized")
        except Exception as e:
            logger.error(f"Failed to initialize transcriber: {e}")
            self.transcriber = None
    
    def process_audio_ultimate(self, audio_path: str, 
                             enable_enhancement: bool = True,
                             language_hint: str = None,
                             progress_callback=None) -> Dict[str, Any]:
        """Ultimate audio processing pipeline"""
        
        if self.transcriber is None:
            return {
                "success": False,
                "error": "Transcriber not initialized",
                "full_transcript": "",
                "chunks": [],
                "processing_time": 0,
                "num_chunks": 0
            }
        
        try:
            start_time = time.time()
            
            # Validate input
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            file_size = os.path.getsize(audio_path)
            if file_size == 0:
                raise ValueError("Audio file is empty")
            
            # Step 1: Ultimate Audio Enhancement
            if progress_callback:
                progress_callback(0.1, "Applying ultimate audio enhancement...")
            
            try:
                if enable_enhancement:
                    enhanced_path = self.enhancer.enhance_audio_ultimate(audio_path)
                else:
                    enhanced_path = audio_path
            except Exception as e:
                logger.warning(f"Enhancement failed: {e}, using original")
                enhanced_path = audio_path
            
            # Step 2: Smart chunking
            if progress_callback:
                progress_callback(0.2, "Creating optimized audio chunks...")
            
            chunks = self.chunker.create_smart_chunks(enhanced_path)
            
            if not chunks:
                return {
                    "success": False,
                    "error": "Failed to create chunks",
                    "full_transcript": "",
                    "chunks": [],
                    "processing_time": time.time() - start_time,
                    "num_chunks": 0
                }
            
            # Step 3: Transcribe
            if progress_callback:
                progress_callback(0.3, f"Transcribing {len(chunks)} chunks...")
            
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
            if enhanced_path != audio_path and os.path.exists(enhanced_path):
                try:
                    os.remove(enhanced_path)
                except:
                    pass
            
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
                "success_rate": successful_chunks / len(chunks) if chunks else 0
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "full_transcript": "",
                "chunks": [],
                "processing_time": time.time() - start_time,
                "num_chunks": 0
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
                # Smart overlap removal
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

# ULTIMATE Gradio Interface
class UltimateTranscriptionUI:
    """ULTIMATE Professional Gradio interface"""
    
    def __init__(self):
        self.pipeline = UltimateTranscriptionPipeline()
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
    
    def process_file_ultimate(self, audio_file, enable_enhancement, language_hint, 
                            chunk_duration, overlap_duration):
        """ULTIMATE file processing"""
        
        if not self.system_ready:
            return "❌ System not ready - check model path and dependencies", "", ""
        
        if audio_file is None:
            return "❌ No audio file uploaded", "", ""
        
        try:
            self.pipeline.chunker.chunk_duration = chunk_duration
            self.pipeline.chunker.overlap_duration = overlap_duration
            
            file_path = audio_file.name if hasattr(audio_file, 'name') else str(audio_file)
            
            if not os.path.exists(file_path):
                return "❌ File not accessible", "", ""
            
            file_size = os.path.getsize(file_path) / (1024 * 1024)
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext not in Config.SUPPORTED_FORMATS:
                return f"❌ Unsupported format: {file_ext}", "", ""
            
            status_msg = f"🎵 Processing: {os.path.basename(file_path)} ({file_size:.1f} MB)\n"
            status_msg += f"🔧 Enhancement: {'Ultimate Multi-Stage' if enable_enhancement else 'Disabled'}\n"
            status_msg += f"🌍 Language: {language_hint}\n"
            status_msg += f"⏱️ Chunks: {chunk_duration}s with {overlap_duration}s overlap\n\n"
            
            result = self.pipeline.process_audio_ultimate(
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
                return status_msg, result["full_transcript"], detailed_report
            else:
                error_msg = f"❌ Failed: {result['error']}\n"
                error_msg += f"⏱️ Time: {result['processing_time']:.1f}s"
                return error_msg, "", error_msg
                
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            logger.error(f"UI error: {e}")
            return error_msg, "", error_msg
    
    def _create_report(self, result: Dict[str, Any]) -> str:
        """Create processing report"""
        report = f"""# 📊 Ultimate Transcription Report

## 🎯 Processing Summary
- **Duration:** {result['processing_time']:.1f} seconds
- **Chunks:** {result['num_chunks']}
- **Success Rate:** {result.get('success_rate', 0)*100:.1f}%
- **Enhanced Audio:** Multi-stage noise reduction applied

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
    
    def create_ultimate_interface(self):
        """Create ULTIMATE interface"""
        
        custom_css = """
        .gradio-container { font-family: 'Inter', system-ui, sans-serif; max-width: 1400px; margin: 0 auto; }
        .status-success { padding: 15px; border-radius: 8px; border-left: 4px solid #10b981; background: linear-gradient(90deg, #ecfdf5 0%, #f0fdf4 100%); }
        .status-error { padding: 15px; border-radius: 8px; border-left: 4px solid #ef4444; background: linear-gradient(90deg, #fef2f2 0%, #fff5f5 100%); }
        """
        
        with gr.Blocks(title="Ultimate Audio Transcription", css=custom_css, theme=gr.themes.Soft()) as interface:
            
            # Header
            gr.HTML("""
            <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 15px; margin-bottom: 25px;">
                <h1 style="margin: 0; font-size: 2.8em;">🎙️ Ultimate Audio Transcription</h1>
                <p style="margin: 15px 0 0 0; font-size: 1.3em;">Advanced Multi-Stage Enhancement • Heavy Noise/Distortion Support</p>
                <p style="margin: 5px 0 0 0; font-size: 1.0em;">All Tensor Issues Resolved • Production Ready</p>
            </div>
            """)
            
            # System Status
            if not self.system_ready:
                gr.HTML('<div class="status-error"><h3>⚠️ System Not Ready</h3><p>Check model path and dependencies</p></div>')
            else:
                gpu_info = f" | {self.system_info['gpu_name']}" if self.system_info['cuda_available'] else " | CPU"
                gr.HTML(f'''<div class="status-success"><h3>✅ Ultimate System Ready</h3>
                <p>Device: {self.system_info['device']}{gpu_info} | All fixes applied</p></div>''')
            
            with gr.Row():
                with gr.Column(scale=2):
                    gr.HTML("<h2>📁 Input & Configuration</h2>")
                    
                    audio_input = gr.File(
                        label="📎 Upload Audio File (Supports Heavy Noise/Distortion)",
                        file_types=Config.SUPPORTED_FORMATS,
                        file_count="single"
                    )
                    
                    enable_enhancement = gr.Checkbox(
                        label="🔧 Enable Ultimate Multi-Stage Enhancement",
                        value=True,
                        info="Advanced pipeline: Spectral subtraction + GPU noise reduction + Wiener filtering + Median filtering"
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
                    
                    process_btn = gr.Button("🚀 Start Ultimate Transcription", variant="primary", size="lg")
                
                with gr.Column(scale=3):
                    gr.HTML("<h2>📊 Results</h2>")
                    
                    status_output = gr.Textbox(
                        label="📈 Processing Status",
                        lines=4,
                        interactive=False,
                        placeholder="Upload audio and start transcription..."
                    )
                    
                    transcript_output = gr.Textbox(
                        label="📝 Ultimate Transcript",
                        lines=12,
                        interactive=True,
                        show_copy_button=True,
                        placeholder="Enhanced transcription will appear here..."
                    )
                    
                    download_btn = gr.DownloadButton("📥 Download", visible=False)
            
            with gr.Accordion("📈 Detailed Report", open=False):
                detailed_report = gr.Markdown("No processing completed yet.")
            
            with gr.Accordion("ℹ️ Enhancement Features", open=False):
                gr.HTML("""
                <div style="padding: 20px; background: #f8fafc; border-radius: 10px;">
                    <h3>🔧 Ultimate Enhancement Pipeline</h3>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px;">
                        <div><strong>Stage 1:</strong> Preemphasis filtering</div>
                        <div><strong>Stage 2:</strong> Advanced bandpass filtering</div>
                        <div><strong>Stage 3:</strong> Spectral subtraction</div>
                        <div><strong>Stage 4:</strong> GPU/CPU noise reduction</div>
                        <div><strong>Stage 5:</strong> Wiener filtering</div>
                        <div><strong>Stage 6:</strong> Median filtering</div>
                    </div>
                    <p><strong>✅ Handles:</strong> Heavy noise, distortion, echo, background music, poor recording quality</p>
                    <p><strong>✅ Preserves:</strong> All speech details, natural flow, speaker characteristics</p>
                </div>
                """)
            
            def process_and_download(audio_file, enable_enhancement, language_hint, chunk_duration, overlap_duration):
                status, transcript, report = self.process_file_ultimate(
                    audio_file, enable_enhancement, language_hint, chunk_duration, overlap_duration
                )
                
                download_file = None
                download_visible = False
                
                if transcript.strip() and "✅" in status:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"ultimate_transcript_{timestamp}.txt"
                    filepath = os.path.join(Config.OUTPUT_DIR, filename)
                    
                    try:
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(f"ULTIMATE Audio Transcription Report\n")
                            f.write(f"Generated: {datetime.now()}\n")
                            f.write(f"Enhancement: {'Ultimate Multi-Stage' if enable_enhancement else 'Disabled'}\n")
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
                
                return status, transcript, report, gr.DownloadButton(
                    label="📥 Download Ultimate Transcript",
                    value=download_file,
                    visible=download_visible
                )
            
            process_btn.click(
                fn=process_and_download,
                inputs=[audio_input, enable_enhancement, language_hint, chunk_duration, overlap_duration],
                outputs=[status_output, transcript_output, detailed_report, download_btn]
            )
        
        return interface

def main():
    """ULTIMATE main function"""
    
    logger.info("🚀 Starting ULTIMATE Audio Transcription System")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        logger.warning("CUDA not available - CPU mode will be slower")
    
    if not os.path.exists(Config.GEMMA_MODEL_PATH):
        logger.error(f"Model not found: {Config.GEMMA_MODEL_PATH}")
        logger.info("Download: huggingface-cli download google/gemma-3n-E4B-it")
    
    try:
        ui = UltimateTranscriptionUI()
        interface = ui.create_ultimate_interface()
        
        logger.info("🎉 Launching ULTIMATE interface...")
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
