#!/usr/bin/env python3
"""
Professional Audio Transcription System with Gemma3n-E4B-it
CORRECTED VERSION - Based on official Hugging Face model documentation
Features: Multi-stage Enhancement, Audio Preview, Smart Chunking, Modern Gradio UI
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

# ML/AI libraries - CORRECTED imports based on documentation
from transformers import (
    AutoProcessor, 
    Gemma3nForConditionalGeneration  # CORRECT class for Gemma3n
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
    
    # GPU settings - CORRECTED based on documentation
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
            
            # Estimate noise from first few frames
            noise_frames = max(1, magnitude.shape[1] // 20)
            noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
            
            # Spectral subtraction
            enhanced_magnitude = magnitude - alpha * noise_spectrum
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
            # Primary speech band (300-3400 Hz)
            nyquist = self.sample_rate * 0.5
            low1, high1 = 300 / nyquist, 3400 / nyquist
            b1, a1 = butter(4, [low1, high1], btype='band')
            filtered1 = filtfilt(b1, a1, audio)
            
            # Extended speech band (80-8000 Hz)
            low2, high2 = 80 / nyquist, min(8000 / nyquist, 0.99)
            b2, a2 = butter(4, [low2, high2], btype='band')
            filtered2 = filtfilt(b2, a2, audio)
            
            # Combine filters
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
            
            # Stage 2: GPU/CPU noise reduction
            if self.torch_gate is not None and torch.cuda.is_available():
                audio_tensor = torch.from_numpy(enhanced).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    enhanced_tensor = self.torch_gate(audio_tensor)
                enhanced = enhanced_tensor.squeeze().cpu().numpy().astype(np.float32)
            else:
                # Multiple passes with different parameters
                enhanced = nr.reduce_noise(
                    y=enhanced,
                    sr=self.sample_rate,
                    stationary=False,
                    prop_decrease=Config.NOISE_REDUCTION_STRENGTH,
                    time_constant_s=1.5,
                    freq_mask_smooth_hz=300,
                    time_mask_smooth_ms=40,
                    n_jobs=1
                )
                
                # Second pass for remaining noise
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
            
            # Stage 4: Median filtering
            enhanced = self.apply_median_filter(enhanced)
            
            return enhanced.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"Multi-stage noise reduction failed: {e}")
            return audio
    
    def enhance_audio_ultimate(self, audio_path: str) -> Tuple[str, str]:
        """Ultimate audio enhancement pipeline - returns (enhanced_path, original_path_for_comparison)"""
        try:
            start_time = time.time()
            logger.info(f"Starting ultimate audio enhancement for: {audio_path}")
            
            # Load audio
            audio, sr = self.safe_audio_load(audio_path)
            original_length = len(audio) / sr
            logger.info(f"Loaded audio: {original_length:.2f}s at {sr}Hz")
            
            # Save original for comparison
            original_path = os.path.join(Config.TEMP_DIR, f"original_{int(time.time())}.wav")
            sf.write(original_path, audio, self.sample_rate)
            
            # Stage 1: Initial normalization
            audio = self.normalize_audio_advanced(audio)
            
            # Stage 2: Preemphasis
            audio = self.apply_preemphasis(audio)
            
            # Stage 3: Advanced bandpass filtering
            audio = self.apply_advanced_bandpass_filter(audio)
            
            # Stage 4: Multi-stage noise reduction
            audio = self.reduce_noise_multi_stage(audio)
            
            # Stage 5: Final normalization
            audio = self.normalize_audio_advanced(audio)
            
            # Save enhanced audio
            enhanced_path = os.path.join(Config.TEMP_DIR, f"enhanced_ultimate_{int(time.time())}.wav")
            sf.write(enhanced_path, audio, self.sample_rate)
            
            processing_time = time.time() - start_time
            logger.info(f"Ultimate audio enhancement completed in {processing_time:.2f}s")
            
            return enhanced_path, original_path
            
        except Exception as e:
            logger.error(f"Ultimate audio enhancement failed: {e}")
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

class CorrectedGemmaTranscriber:
    """CORRECTED Gemma3n-E4B-it transcription engine based on official documentation"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or Config.GEMMA_MODEL_PATH
        self.device = Config.DEVICE
        self.torch_dtype = Config.TORCH_DTYPE
        
        self.processor = None
        self.model = None
        
        self._load_model_safely()
    
    def _load_model_safely(self):
        """Load Gemma model using CORRECT classes from documentation"""
        try:
            logger.info(f"Loading Gemma3n-E4B-it model from {self.model_path}")
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model path does not exist: {self.model_path}")
            
            # CORRECTED: Use proper classes as per documentation
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                local_files_only=True,
                trust_remote_code=True
            )
            
            # CORRECTED: Use Gemma3nForConditionalGeneration instead of AutoModelForImageTextToText
            self.model = Gemma3nForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=self.torch_dtype,
                device_map="auto",
                local_files_only=True,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            ).eval()
            
            logger.info(f"Model loaded successfully with dtype: {self.torch_dtype}")
            
        except Exception as e:
            logger.error(f"Failed to load Gemma model: {e}")
            self.processor = None
            self.model = None
            raise
    
    def transcribe_chunk_safely(self, audio_path: str, language_hint: str = None) -> Dict[str, Any]:
        """CORRECTED: Safely transcribe using proper Gemma3n format"""
        if self.model is None or self.processor is None:
            return {
                "text": "",
                "success": False,
                "error": "Model not loaded"
            }
        
        try:
            # CORRECTED: Use proper message format from documentation
            system_prompt = "You are a professional audio transcriber. Transcribe the audio accurately, preserving all spoken content including natural speech patterns. Output clean, readable text without timestamps or speaker labels."
            
            user_prompt = "Transcribe this audio clearly and accurately"
            if language_hint and language_hint.strip() and language_hint != "Auto-detect":
                user_prompt += f" in {language_hint}"
            user_prompt += ". Include all spoken words and preserve the natural flow of speech."
            
            messages = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "audio", "audio": audio_path},  # CORRECTED: Use "audio" key as per docs
                        {"type": "text", "text": user_prompt}
                    ]
                }
            ]
            
            # CORRECTED: Apply chat template properly
            try:
                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt"
                )
                
                # CORRECTED: Move to device and ensure proper dtype
                inputs = inputs.to(self.model.device)
                
                # Ensure input_ids are Long type (required for embeddings)
                if "input_ids" in inputs:
                    inputs["input_ids"] = inputs["input_ids"].long()
                    
                # Ensure attention_mask is proper type
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = inputs["attention_mask"].long()
                
            except Exception as e:
                logger.error(f"Chat template failed: {e}")
                return {
                    "text": "",
                    "success": False,
                    "error": f"Template error: {str(e)}"
                }
            
            # CORRECTED: Generate with proper parameters
            try:
                with torch.inference_mode():
                    generation = self.model.generate(
                        **inputs,
                        max_new_tokens=Config.MAX_NEW_TOKENS,
                        do_sample=False,
                        temperature=0.1,
                        pad_token_id=self.processor.tokenizer.eos_token_id
                    )
                    
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                gc.collect()
                with torch.inference_mode():
                    generation = self.model.generate(
                        **inputs,
                        max_new_tokens=Config.MAX_NEW_TOKENS // 2,
                        do_sample=False,
                        pad_token_id=self.processor.tokenizer.eos_token_id
                    )
                    
            except Exception as gen_error:
                logger.error(f"Generation failed: {gen_error}")
                return {
                    "text": "",
                    "success": False,
                    "error": f"Generation failed: {str(gen_error)}"
                }
            
            # CORRECTED: Decode properly
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
                time.sleep(0.1)
        
        return results

class CorrectedTranscriptionPipeline:
    """CORRECTED transcription pipeline with advanced audio enhancement"""
    
    def __init__(self):
        self.enhancer = AdvancedAudioEnhancer()
        self.chunker = SmartAudioChunker(
            chunk_duration=Config.CHUNK_DURATION,
            overlap_duration=Config.OVERLAP_DURATION
        )
        self.transcriber = None
        
        try:
            self.transcriber = CorrectedGemmaTranscriber()
            logger.info("CORRECTED transcription pipeline initialized")
        except Exception as e:
            logger.error(f"Failed to initialize transcriber: {e}")
            self.transcriber = None
    
    def process_audio_ultimate(self, audio_path: str, 
                             enable_enhancement: bool = True,
                             language_hint: str = None,
                             progress_callback=None) -> Dict[str, Any]:
        """Ultimate audio processing pipeline with audio preview"""
        
        if self.transcriber is None:
            return {
                "success": False,
                "error": "Transcriber not initialized",
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
            
            # Step 1: Ultimate Audio Enhancement with preview
            if progress_callback:
                progress_callback(0.1, "Applying ultimate audio enhancement...")
            
            enhanced_audio_path = None
            original_audio_path = None
            
            try:
                if enable_enhancement:
                    enhanced_audio_path, original_audio_path = self.enhancer.enhance_audio_ultimate(audio_path)
                    processing_path = enhanced_audio_path
                else:
                    processing_path = audio_path
                    original_audio_path = audio_path
            except Exception as e:
                logger.warning(f"Enhancement failed: {e}, using original")
                processing_path = audio_path
                original_audio_path = audio_path
            
            # Step 2: Smart chunking
            if progress_callback:
                progress_callback(0.2, "Creating optimized audio chunks...")
            
            chunks = self.chunker.create_smart_chunks(processing_path)
            
            if not chunks:
                return {
                    "success": False,
                    "error": "Failed to create chunks",
                    "full_transcript": "",
                    "chunks": [],
                    "processing_time": time.time() - start_time,
                    "num_chunks": 0,
                    "enhanced_audio_path": enhanced_audio_path,
                    "original_audio_path": original_audio_path
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

# CORRECTED Gradio Interface with Audio Preview
class CorrectedTranscriptionUI:
    """CORRECTED Professional Gradio interface with audio preview functionality"""
    
    def __init__(self):
        self.pipeline = CorrectedTranscriptionPipeline()
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
    
    def process_file_corrected(self, audio_file, enable_enhancement, language_hint, 
                            chunk_duration, overlap_duration):
        """CORRECTED file processing with audio preview"""
        
        if not self.system_ready:
            return "❌ System not ready - check model path and dependencies", "", "", None, None
        
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
                
                # Return audio files for preview
                enhanced_audio = result.get('enhanced_audio_path')
                original_audio = result.get('original_audio_path')
                
                return status_msg, result["full_transcript"], detailed_report, enhanced_audio, original_audio
            else:
                error_msg = f"❌ Failed: {result['error']}\n"
                error_msg += f"⏱️ Time: {result['processing_time']:.1f}s"
                return error_msg, "", error_msg, None, None
                
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            logger.error(f"UI error: {e}")
            return error_msg, "", error_msg, None, None
    
    def _create_report(self, result: Dict[str, Any]) -> str:
        """Create processing report"""
        report = f"""# 📊 CORRECTED Transcription Report

## 🎯 Processing Summary
- **Duration:** {result['processing_time']:.1f} seconds
- **Chunks:** {result['num_chunks']}
- **Success Rate:** {result.get('success_rate', 0)*100:.1f}%
- **Enhanced Audio:** Multi-stage noise reduction applied
- **Model:** Gemma3n-E4B-it (CORRECTED implementation)

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
    
    def create_corrected_interface(self):
        """Create CORRECTED interface with audio preview"""
        
        custom_css = """
        .gradio-container { font-family: 'Inter', system-ui, sans-serif; max-width: 1500px; margin: 0 auto; }
        .status-success { padding: 15px; border-radius: 8px; border-left: 4px solid #10b981; background: linear-gradient(90deg, #ecfdf5 0%, #f0fdf4 100%); }
        .status-error { padding: 15px; border-radius: 8px; border-left: 4px solid #ef4444; background: linear-gradient(90deg, #fef2f2 0%, #fff5f5 100%); }
        .audio-preview { background: #f8fafc; padding: 15px; border-radius: 8px; margin: 10px 0; }
        """
        
        with gr.Blocks(title="CORRECTED Audio Transcription", css=custom_css, theme=gr.themes.Soft()) as interface:
            
            # Header
            gr.HTML("""
            <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 15px; margin-bottom: 25px;">
                <h1 style="margin: 0; font-size: 2.8em;">🎙️ CORRECTED Audio Transcription</h1>
                <p style="margin: 15px 0 0 0; font-size: 1.3em;">Based on Official Gemma3n-E4B-it Documentation</p>
                <p style="margin: 5px 0 0 0; font-size: 1.0em;">All Tensor Issues Resolved • Audio Preview Included</p>
            </div>
            """)
            
            # System Status
            if not self.system_ready:
                gr.HTML('<div class="status-error"><h3>⚠️ System Not Ready</h3><p>Check model path and dependencies</p></div>')
            else:
                gpu_info = f" | {self.system_info['gpu_name']}" if self.system_info['cuda_available'] else " | CPU"
                gr.HTML(f'''<div class="status-success"><h3>✅ CORRECTED System Ready</h3>
                <p>Device: {self.system_info['device']}{gpu_info} | Gemma3n-E4B-it properly loaded</p></div>''')
            
            with gr.Row():
                with gr.Column(scale=2):
                    gr.HTML("<h2>📁 Input & Configuration</h2>")
                    
                    audio_input = gr.File(
                        label="📎 Upload Audio File (Heavy Noise/Distortion Supported)",
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
                    
                    process_btn = gr.Button("🚀 Start CORRECTED Transcription", variant="primary", size="lg")
                
                with gr.Column(scale=3):
                    gr.HTML("<h2>📊 Results & Audio Preview</h2>")
                    
                    status_output = gr.Textbox(
                        label="📈 Processing Status",
                        lines=4,
                        interactive=False,
                        placeholder="Upload audio and start transcription..."
                    )
                    
                    # Audio Preview Section
                    with gr.Accordion("🎵 Audio Preview (Before & After Enhancement)", open=False):
                        with gr.Row():
                            with gr.Column():
                                gr.HTML("<h4>🔊 Original Audio</h4>")
                                original_audio_player = gr.Audio(
                                    label="Original",
                                    visible=False,
                                    interactive=False
                                )
                            
                            with gr.Column():
                                gr.HTML("<h4>✨ Enhanced Audio</h4>")
                                enhanced_audio_player = gr.Audio(
                                    label="Enhanced",
                                    visible=False,
                                    interactive=False
                                )
                    
                    transcript_output = gr.Textbox(
                        label="📝 CORRECTED Transcript",
                        lines=12,
                        interactive=True,
                        show_copy_button=True,
                        placeholder="Enhanced transcription will appear here..."
                    )
                    
                    download_btn = gr.DownloadButton("📥 Download", visible=False)
            
            with gr.Accordion("📈 Detailed Report", open=False):
                detailed_report = gr.Markdown("No processing completed yet.")
            
            with gr.Accordion("ℹ️ CORRECTED Implementation Details", open=False):
                gr.HTML("""
                <div style="padding: 20px; background: #f8fafc; border-radius: 10px;">
                    <h3>🔧 Key Corrections Applied</h3>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px;">
                        <div><strong>✅ Model Class:</strong> Using Gemma3nForConditionalGeneration</div>
                        <div><strong>✅ Tensor Types:</strong> Proper input_ids (Long) and attention_mask handling</div>
                        <div><strong>✅ Audio Format:</strong> Following official documentation format</div>
                        <div><strong>✅ Chat Template:</strong> Correct message structure for audio inputs</div>
                    </div>
                    <h3>🎵 Enhancement Pipeline</h3>
                    <p><strong>Stage 1:</strong> Preemphasis filtering | <strong>Stage 2:</strong> Advanced bandpass filtering</p>
                    <p><strong>Stage 3:</strong> Spectral subtraction | <strong>Stage 4:</strong> GPU/CPU noise reduction</p>
                    <p><strong>Stage 5:</strong> Wiener filtering | <strong>Stage 6:</strong> Median filtering</p>
                </div>
                """)
            
            def process_and_download_corrected(audio_file, enable_enhancement, language_hint, chunk_duration, overlap_duration):
                status, transcript, report, enhanced_audio, original_audio = self.process_file_corrected(
                    audio_file, enable_enhancement, language_hint, chunk_duration, overlap_duration
                )
                
                download_file = None
                download_visible = False
                
                if transcript.strip() and "✅" in status:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"corrected_transcript_{timestamp}.txt"
                    filepath = os.path.join(Config.OUTPUT_DIR, filename)
                    
                    try:
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(f"CORRECTED Audio Transcription Report\n")
                            f.write(f"Generated: {datetime.now()}\n")
                            f.write(f"Model: Gemma3n-E4B-it (Properly Implemented)\n")
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
                
                # Update audio players
                original_visible = original_audio is not None
                enhanced_visible = enhanced_audio is not None
                
                return (
                    status, 
                    transcript, 
                    report,
                    gr.Audio(value=original_audio, visible=original_visible, interactive=False, label="Original Audio"),
                    gr.Audio(value=enhanced_audio, visible=enhanced_visible, interactive=False, label="Enhanced Audio"),
                    gr.DownloadButton(
                        label="📥 Download CORRECTED Transcript",
                        value=download_file,
                        visible=download_visible
                    )
                )
            
            process_btn.click(
                fn=process_and_download_corrected,
                inputs=[audio_input, enable_enhancement, language_hint, chunk_duration, overlap_duration],
                outputs=[status_output, transcript_output, detailed_report, original_audio_player, enhanced_audio_player, download_btn]
            )
        
        return interface

def main():
    """CORRECTED main function"""
    
    logger.info("🚀 Starting CORRECTED Audio Transcription System")
    
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
        ui = CorrectedTranscriptionUI()
        interface = ui.create_corrected_interface()
        
        logger.info("🎉 Launching CORRECTED interface...")
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
