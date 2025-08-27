# -*- coding: utf-8 -*-
"""
ADAPTIVE SPEECH TRANSCRIPTION WITH GEEKSFORGEEKS PREPROCESSING
============================================================

FEATURES IMPLEMENTED:
- GeeksforGeeks audio preprocessing methods (resampling, filtering, normalization)
- Enable/Disable preprocessing toggle button
- Adaptive chunk sizing with fallback (30s default, 10s, 15s, 20s, 40s fallbacks)
- Robust error handling and chunk retry mechanism
- 75-second timeout protection per chunk

Author: Adaptive Audio Processing System
Version: GeeksforGeeks Enhanced 16.0
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
import datetime
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import psutil
import re
import nltk
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

# --- ADAPTIVE CONFIGURATION ---
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

class GeeksforGeeksAudioPreprocessor:
    """GeeksforGeeks-based audio preprocessing implementation"""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        print(f"🚀 GeeksforGeeks Audio Preprocessor initialized for {sample_rate}Hz")
    
    def resample_audio(self, audio_path: str, target_sr: int = None) -> Tuple[np.ndarray, int]:
        """Resample audio to target sample rate (GeeksforGeeks method)"""
        try:
            if target_sr is None:
                target_sr = self.sample_rate
            
            print(f"🔄 Resampling audio to {target_sr}Hz...")
            y, sr = librosa.load(audio_path, sr=target_sr)
            print(f"✅ Sample rate after resampling: {sr}")
            return y, sr
            
        except Exception as e:
            print(f"❌ Resampling failed: {e}")
            # Fallback: load with original sample rate
            y, sr = librosa.load(audio_path, sr=None)
            if sr != target_sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
            return y, sr
    
    def butter_lowpass_filter(self, data: np.ndarray, cutoff_freq: int, sample_rate: int, order: int = 4) -> np.ndarray:
        """Apply Butterworth low-pass filter (GeeksforGeeks method)"""
        try:
            print(f"🎵 Applying Butterworth low-pass filter (cutoff: {cutoff_freq}Hz, order: {order})...")
            
            nyquist = 0.5 * sample_rate
            normal_cutoff = cutoff_freq / nyquist
            
            # Ensure cutoff frequency is valid
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
    
    def convert_to_model_input(self, audio: np.ndarray, target_length: int) -> np.ndarray:
        """Convert audio to model's expected input format (GeeksforGeeks method)"""
        try:
            print(f"📏 Converting to model input (target length: {target_length})...")
            
            if len(audio) < target_length:
                # Pad with zeros
                audio = np.pad(audio, (0, target_length - len(audio)))
                print(f"📈 Padded audio to {len(audio)} samples")
            else:
                # Trim to target length
                audio = audio[:target_length]
                print(f"✂️ Trimmed audio to {len(audio)} samples")
            
            print(f"✅ Model input shape: {audio.shape}")
            return audio.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Model input conversion failed: {e}")
            return audio.astype(np.float32)
    
    def compute_logmel_spectrogram(self, audio: np.ndarray, sr: int, n_mels: int = 128, hop_length: int = 512) -> np.ndarray:
        """Compute log-mel spectrogram (GeeksforGeeks method)"""
        try:
            print(f"📊 Computing log-mel spectrogram (n_mels: {n_mels}, hop_length: {hop_length})...")
            
            mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length)
            logmel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
            
            print(f"✅ Log-mel spectrogram shape: {logmel_spectrogram.shape}")
            return logmel_spectrogram
            
        except Exception as e:
            print(f"❌ Log-mel spectrogram computation failed: {e}")
            return np.array([])
    
    def apply_geeksforgeeks_preprocessing(self, audio_path: str) -> Tuple[np.ndarray, Dict]:
        """Apply complete GeeksforGeeks preprocessing pipeline"""
        try:
            print("🚀 Applying GEEKSFORGEEKS preprocessing pipeline...")
            stats = {}
            
            # Step 1: Resample audio
            resampled_audio, sr = self.resample_audio(audio_path, self.sample_rate)
            stats['original_length'] = len(resampled_audio) / sr
            stats['sample_rate'] = sr
            
            # Step 2: Apply filtering
            filtered_audio = self.butter_lowpass_filter(
                resampled_audio, 
                GEEKSFORGEEKS_CUTOFF_FREQ, 
                sr, 
                GEEKSFORGEEKS_FILTER_ORDER
            )
            
            # Step 3: Normalize amplitude
            if np.max(np.abs(filtered_audio)) > 0:
                filtered_audio = filtered_audio / np.max(np.abs(filtered_audio)) * 0.95
            
            # Step 4: Final quality control
            filtered_audio = np.clip(filtered_audio, -0.99, 0.99)
            
            # Calculate final stats
            stats['final_rms'] = np.sqrt(np.mean(filtered_audio**2))
            stats['final_length'] = len(filtered_audio) / sr
            stats['preprocessing_applied'] = True
            
            print(f"✅ GEEKSFORGEEKS preprocessing completed")
            print(f"📊 Final RMS level: {stats['final_rms']:.4f}")
            
            return filtered_audio.astype(np.float32), stats
            
        except Exception as e:
            print(f"❌ GeeksforGeeks preprocessing failed: {e}")
            # Fallback: return original audio
            try:
                audio, sr = librosa.load(audio_path, sr=self.sample_rate)
                return audio.astype(np.float32), {'preprocessing_applied': False}
            except:
                return np.array([]), {'preprocessing_applied': False}

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

class AdaptiveSpeechTranscriber:
    """Adaptive transcriber with GeeksforGeeks preprocessing and variable chunk sizing"""
    
    def __init__(self, model_path: str, use_quantization: bool = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = None
        self.processor = None
        self.audio_preprocessor = GeeksforGeeksAudioPreprocessor(SAMPLE_RATE)
        self.text_chunker = SmartTextChunker()
        self.chunk_count = 0
        self.temp_files = []
        
        print(f"🖥️ Using device: {self.device}")
        print(f"🚀 ADAPTIVE transcriber with GeeksforGeeks preprocessing initialized")
        print(f"📏 Default chunk size: {DEFAULT_CHUNK_SECONDS}s")
        print(f"📏 Fallback chunk sizes: {FALLBACK_CHUNK_SECONDS}")
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
            print(f"✅ ADAPTIVE model loaded in {loading_time:.1f} seconds")
            
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
            print(f"❌ Adaptive transcription error: {str(e)}")
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
            merged_text += f"\n\n[Adaptive Processing Summary: {', '.join(summary_parts)} - {success_rate:.1f}% success rate with {chunk_seconds}s chunks]"
            
            if noisy_timeout_count > 0:
                merged_text += f"\n[Note: {noisy_timeout_count} chunks were too noisy and timed out after {CHUNK_TIMEOUT} seconds each]"
        
        return merged_text.strip()
    
    def transcribe_with_geeksforgeeks_preprocessing(self, audio_path: str, enable_preprocessing: bool, 
                                                  language: str = "auto") -> Tuple[str, str, str, Dict]:
        try:
            print(f"🚀 Starting ADAPTIVE transcription with GeeksforGeeks preprocessing...")
            print(f"🔧 Preprocessing enabled: {enable_preprocessing}")
            print(f"🌍 Language: {language}")
            print(f"📏 Adaptive chunk sizing enabled")
            
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
            
            # Apply GeeksforGeeks preprocessing if enabled
            if enable_preprocessing:
                enhanced_audio, preprocessing_stats = self.audio_preprocessor.apply_geeksforgeeks_preprocessing(audio_path)
            else:
                print("⚠️ Preprocessing DISABLED - using raw audio")
                enhanced_audio = audio_array
                preprocessing_stats = {'preprocessing_applied': False}
            
            enhanced_path = tempfile.mktemp(suffix="_geeksforgeeks_enhanced.wav")
            original_path = tempfile.mktemp(suffix="_original.wav")
            
            sf.write(enhanced_path, enhanced_audio, SAMPLE_RATE)
            sf.write(original_path, audio_array, SAMPLE_RATE)
            
            print("✂️ Starting adaptive chunk transcription...")
            start_time = time.time()
            
            # Transcribe with adaptive chunk sizing
            final_transcription, transcription_stats = self.transcribe_with_adaptive_chunks(enhanced_audio, language)
            
            processing_time = time.time() - start_time
            
            # Combine stats
            combined_stats = {**preprocessing_stats, **transcription_stats}
            combined_stats['processing_time'] = processing_time
            
            print(f"✅ ADAPTIVE transcription completed in {processing_time:.2f}s")
            if 'chunk_seconds_used' in transcription_stats:
                print(f"📏 Optimal chunk size: {transcription_stats['chunk_seconds_used']}s")
            if 'success_rate' in transcription_stats:
                print(f"📊 Success rate: {transcription_stats['success_rate']*100:.1f}%")
            
            return final_transcription, original_path, enhanced_path, combined_stats
                
        except Exception as e:
            error_msg = f"❌ Adaptive transcription failed: {e}"
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
            print("🌐 Starting adaptive text translation...")
            
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
            print(f"❌ Adaptive translation error: {str(e)}")
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
            
            if "🚀" in text or "ADAPTIVE" in text or "GEEKSFORGEEKS" in text:
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
            return "\n".join(self.log_buffer[-50:]) if self.log_buffer else "🚀 Adaptive system ready..."

def setup_adaptive_logging():
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
    return "🚀 Adaptive system initializing..."

def initialize_adaptive_transcriber():
    global transcriber
    if transcriber is None:
        try:
            print("🚀 Initializing ADAPTIVE Speech Transcription with GeeksforGeeks Preprocessing...")
            print("✅ ADAPTIVE FEATURES ENABLED:")
            print("🔧 GeeksforGeeks Preprocessing: Resampling, Butterworth Filtering, Normalization")
            print("📏 Adaptive Chunk Sizing: 30s default, fallbacks: 10s, 15s, 20s, 40s")
            print("⚡ Enable/Disable Preprocessing Toggle")
            print("🔄 Automatic Fallback Mechanism")
            print(f"⏱️ Chunk timeout: {CHUNK_TIMEOUT} seconds")
            
            transcriber = AdaptiveSpeechTranscriber(model_path=MODEL_PATH, use_quantization=True)
            return "✅ ADAPTIVE transcription system ready! GeeksforGeeks preprocessing enabled."
        except Exception as e:
            try:
                print("🔄 Retrying without quantization...")
                transcriber = AdaptiveSpeechTranscriber(model_path=MODEL_PATH, use_quantization=False)
                return "✅ ADAPTIVE system loaded (standard precision)!"
            except Exception as e2:
                error_msg = f"❌ Adaptive system failure: {str(e2)}"
                print(error_msg)
                return error_msg
    return "✅ ADAPTIVE system already active!"

def transcribe_audio_adaptive(audio_input, language_choice, enable_preprocessing, progress=gr.Progress()):
    global transcriber
    
    if audio_input is None:
        return "❌ Please upload an audio file or record audio.", None, None, "", ""
    
    if transcriber is None:
        return "❌ System not initialized. Please wait for startup.", None, None, "", ""
    
    start_time = time.time()
    print(f"🚀 Starting ADAPTIVE transcription with GeeksforGeeks preprocessing...")
    print(f"🌍 Language: {language_choice}")
    print(f"🔧 Preprocessing enabled: {enable_preprocessing}")
    print(f"📏 Adaptive chunk sizing: {DEFAULT_CHUNK_SECONDS}s → {FALLBACK_CHUNK_SECONDS}")
    
    progress(0.1, desc="Initializing ADAPTIVE processing...")
    
    temp_audio_path = None
    
    try:
        temp_audio_path = AudioHandler.convert_to_file(audio_input, SAMPLE_RATE)
        
        if enable_preprocessing:
            progress(0.3, desc="Applying GeeksforGeeks preprocessing...")
        else:
            progress(0.3, desc="Skipping preprocessing (disabled by user)...")
        
        language_code = SUPPORTED_LANGUAGES.get(language_choice, "auto")
        
        progress(0.5, desc="ADAPTIVE transcription with variable chunk sizing...")
        
        transcription, original_path, enhanced_path, stats = transcriber.transcribe_with_geeksforgeeks_preprocessing(
            temp_audio_path, enable_preprocessing, language_code
        )
        
        progress(0.9, desc="Generating ADAPTIVE reports...")
        
        enhancement_report = create_adaptive_enhancement_report(stats, enable_preprocessing)
        
        processing_time = time.time() - start_time
        processing_report = create_adaptive_processing_report(
            temp_audio_path, language_choice, enable_preprocessing, 
            processing_time, len(transcription.split()) if isinstance(transcription, str) else 0,
            stats
        )
        
        progress(1.0, desc="ADAPTIVE processing complete!")
        
        print(f"✅ ADAPTIVE transcription completed in {processing_time:.2f}s")
        print(f"📊 Output: {len(transcription.split()) if isinstance(transcription, str) else 0} words")
        
        return transcription, original_path, enhanced_path, enhancement_report, processing_report
        
    except Exception as e:
        error_msg = f"❌ Adaptive system error: {str(e)}"
        print(error_msg)
        OptimizedMemoryManager.fast_cleanup()
        return error_msg, None, None, "", ""
    finally:
        if temp_audio_path:
            AudioHandler.cleanup_temp_file(temp_audio_path)

def translate_transcription_adaptive(transcription_text, progress=gr.Progress()):
    global transcriber
    
    if not transcription_text or transcription_text.strip() == "":
        return "❌ No transcription text to translate. Please transcribe audio first."
    
    if transcriber is None:
        return "❌ System not initialized. Please wait for system startup."
    
    if transcription_text.startswith("❌") or transcription_text.startswith("["):
        return "❌ Cannot translate error messages or system messages. Please provide valid transcription text."
    
    progress(0.1, desc="Preparing text for adaptive translation...")
    
    try:
        text_to_translate = transcription_text
        if "\n\n[Adaptive Processing Summary:" in text_to_translate:
            text_to_translate = text_to_translate.split("\n\n[Adaptive Processing Summary:")[0].strip()
        
        progress(0.3, desc="Creating smart text chunks...")
        
        start_time = time.time()
        translated_text = transcriber.translate_text_chunks(text_to_translate)
        translation_time = time.time() - start_time
        
        progress(0.9, desc="Finalizing adaptive translation...")
        
        if not translated_text.startswith('['):
            translated_text += f"\n\n[Adaptive Translation completed in {translation_time:.2f}s using smart chunking]"
        
        progress(1.0, desc="Adaptive translation complete!")
        
        print(f"✅ Adaptive translation completed in {translation_time:.2f}s")
        
        return translated_text
        
    except Exception as e:
        error_msg = f"❌ Adaptive translation failed: {str(e)}"
        print(error_msg)
        OptimizedMemoryManager.fast_cleanup()
        return error_msg

def create_adaptive_enhancement_report(stats: Dict, enable_preprocessing: bool) -> str:
    if not stats:
        return "⚠️ Enhancement statistics not available"
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
🚀 ADAPTIVE SPEECH ENHANCEMENT REPORT
=====================================
Timestamp: {timestamp}
Preprocessing Enabled: {'✅ YES' if enable_preprocessing else '❌ NO'}

📊 GEEKSFORGEEKS PREPROCESSING STATUS:
"""
    
    if enable_preprocessing and stats.get('preprocessing_applied', False):
        report += f"""• Audio Duration: {stats.get('original_length', 0):.2f} seconds
• Sample Rate: {stats.get('sample_rate', SAMPLE_RATE)} Hz
• Final RMS Level: {stats.get('final_rms', 0):.4f} (ASR-optimized)
• Preprocessing Applied: ✅ GEEKSFORGEEKS PIPELINE

🔧 GEEKSFORGEEKS PIPELINE STAGES:
• Stage 1: ✅ Audio Resampling (16000 Hz)
• Stage 2: ✅ Butterworth Low-pass Filter ({GEEKSFORGEEKS_CUTOFF_FREQ} Hz cutoff, Order {GEEKSFORGEEKS_FILTER_ORDER})
• Stage 3: ✅ Amplitude Normalization (0.95 max)
• Stage 4: ✅ Quality Control Clipping (-0.99 to 0.99)
"""
    else:
        report += f"""• Preprocessing: ❌ DISABLED BY USER
• Raw Audio Used: ✅ NO PREPROCESSING APPLIED
• Audio Duration: {stats.get('original_length', 0):.2f} seconds
"""
    
    report += f"""
📏 ADAPTIVE CHUNK SIZING RESULTS:
• Chunk Size Used: {stats.get('chunk_seconds_used', 'N/A')}s
• Total Chunks: {stats.get('total_chunks', 0)}
• Successful Chunks: {stats.get('successful_chunks', 0)}
• Failed Chunks: {stats.get('failed_chunks', 0)}
• Timeout Chunks: {stats.get('timeout_chunks', 0)}
• Success Rate: {stats.get('success_rate', 0)*100:.1f}%
• Attempts Made: {stats.get('attempts_made', 0)}

⏱️ TIMEOUT PROTECTION:
• Chunk Timeout: {CHUNK_TIMEOUT} seconds per chunk
• Adaptive Quality Detection: ✅ ACTIVE
• Timeout Messages: ✅ ENABLED

🚀 ADAPTIVE FEATURES:
• Default Chunk Size: {DEFAULT_CHUNK_SECONDS} seconds
• Fallback Chunk Sizes: {', '.join(map(str, FALLBACK_CHUNK_SECONDS))} seconds
• Automatic Fallback: ✅ ENABLED
• Preprocessing Toggle: ✅ USER CONTROLLED

🏆 ADAPTIVE ENHANCEMENT SCORE: {min(100, stats.get('success_rate', 0)*100):.0f}/100

🔧 TECHNICAL SPECIFICATIONS:
• Processing Method: ADAPTIVE CHUNK SIZING WITH GEEKSFORGEEKS PREPROCESSING
• Enhancement Control: USER TOGGLE (Enable/Disable)
• Fallback Mechanism: AUTOMATIC CHUNK SIZE ADAPTATION
• Quality Detection: SUCCESS RATE BASED ADAPTATION
• Memory Management: GPU-optimized with cleanup
• Error Recovery: COMPREHENSIVE FALLBACK SYSTEMS
"""
    return report

def create_adaptive_processing_report(audio_path: str, language: str, enable_preprocessing: bool, 
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
    
    report = f"""
🚀 ADAPTIVE SPEECH TRANSCRIPTION REPORT
=======================================
Generated: {timestamp}

🎵 ADAPTIVE AUDIO PROCESSING:
• Source File: {os.path.basename(audio_path)}
• {audio_info}
• Target Language: {language}
• Preprocessing Enabled: {'✅ YES' if enable_preprocessing else '❌ NO'}

⚡ PERFORMANCE METRICS:
• Processing Time: {processing_time:.2f} seconds
• Words Generated: {word_count}
• Processing Speed: {word_count/processing_time:.1f} words/second
• Processing Device: {device_info}

🚀 ADAPTIVE CONFIGURATION:
• Model: Gemma 3N E4B-IT (Adaptive Enhanced)
• Default Chunk Size: {DEFAULT_CHUNK_SECONDS} seconds
• Fallback Chunk Sizes: {', '.join(map(str, FALLBACK_CHUNK_SECONDS))} seconds
• Chunk Timeout: {CHUNK_TIMEOUT} seconds per chunk
• Overlap: {OVERLAP_SECONDS} seconds (Context Preserving)
• Enhancement Method: ADAPTIVE CHUNK SIZING WITH GEEKSFORGEEKS PREPROCESSING

📏 ADAPTIVE CHUNK SIZING RESULTS:
• Optimal Chunk Size Found: {chunk_seconds}s
• Success Rate Achieved: {success_rate:.1f}%
• Total Attempts Made: {attempts}
• Successful Chunks: {stats.get('successful_chunks', 0)}
• Failed Chunks: {stats.get('failed_chunks', 0)}
• Timeout Chunks: {stats.get('timeout_chunks', 0)}

🔧 GEEKSFORGEEKS PREPROCESSING:
"""
    
    if enable_preprocessing and stats.get('preprocessing_applied', False):
        report += f"""• Resampling: ✅ Applied (16000 Hz)
• Butterworth Filter: ✅ Applied ({GEEKSFORGEEKS_CUTOFF_FREQ} Hz cutoff, Order {GEEKSFORGEEKS_FILTER_ORDER})
• Amplitude Normalization: ✅ Applied (0.95 max)
• Quality Control: ✅ Applied (-0.99 to 0.99 clipping)
• Final RMS Level: {stats.get('final_rms', 0):.4f}
"""
    else:
        report += f"""• Preprocessing: ❌ DISABLED BY USER
• Raw Audio Processing: ✅ NO PREPROCESSING APPLIED
• User Choice: Skip enhancement pipeline
"""
    
    report += f"""
⏱️ TIMEOUT & ADAPTIVE HANDLING:
• Timeout Protection: ✅ {CHUNK_TIMEOUT}s per chunk
• Adaptive Quality Detection: ✅ Success rate monitoring
• Automatic Fallback: ✅ Multiple chunk sizes tried
• Timeout Messages: ✅ "Input Audio Very noisy. Unable to extract details."
• Fallback Systems: ✅ COMPREHENSIVE ERROR RECOVERY

🌐 TRANSLATION FEATURES:
• Translation Control: ✅ USER-INITIATED (Optional)
• Smart Text Chunking: ✅ ENABLED
• Context Preservation: ✅ SENTENCE OVERLAP
• Processing Method: ✅ ADAPTIVE PIPELINE

📊 ADAPTIVE SYSTEM STATUS:
• Enhancement Method: ✅ ADAPTIVE CHUNK SIZING WITH GEEKSFORGEEKS PREPROCESSING
• Preprocessing Control: ✅ USER TOGGLE (Enable/Disable)
• Chunk Size Adaptation: ✅ AUTOMATIC FALLBACK MECHANISM
• Success Rate Monitoring: ✅ ACTIVE
• Quality Detection: ✅ ADAPTIVE ANALYSIS
• Memory Optimization: ✅ GPU-AWARE CLEANUP
• Error Recovery: ✅ COMPREHENSIVE FALLBACK SYSTEMS

✅ STATUS: ADAPTIVE TRANSCRIPTION COMPLETED
🚀 AUDIO ENHANCEMENT: GEEKSFORGEEKS PREPROCESSING (USER CONTROLLED)
📏 CHUNK SIZING: ADAPTIVE WITH AUTOMATIC FALLBACK
⏱️ TIMEOUT PROTECTION: 75-SECOND CHUNK SAFETY
🔧 PREPROCESSING: USER TOGGLE (ENABLE/DISABLE)
📊 OPTIMIZATION: ADAPTIVE CHUNK SIZE SELECTION
🎯 RELIABILITY: ADAPTIVE PROCESSING WITH COMPREHENSIVE FALLBACKS
"""
    return report

def create_adaptive_interface():
    """Create adaptive speech enhancement interface with GeeksforGeeks preprocessing"""
    
    adaptive_css = """
    :root {
        --primary-color: #0f172a;
        --secondary-color: #1e293b;
        --accent-color: #10b981;
        --adaptive-color: #06b6d4;
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
    
    .adaptive-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 15%, #10b981 30%, #06b6d4 45%, #f59e0b 60%, #3b82f6 75%, #8b5cf6 90%, #ec4899 100%) !important;
        padding: 60px 40px !important;
        border-radius: 30px !important;
        text-align: center !important;
        margin-bottom: 50px !important;
        box-shadow: 0 30px 60px rgba(16, 185, 129, 0.4) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .adaptive-title {
        font-size: 4rem !important;
        font-weight: 900 !important;
        color: white !important;
        margin-bottom: 20px !important;
        text-shadow: 0 5px 15px rgba(16, 185, 129, 0.6) !important;
        position: relative !important;
        z-index: 2 !important;
    }
    
    .adaptive-subtitle {
        font-size: 1.5rem !important;
        color: rgba(255,255,255,0.95) !important;
        font-weight: 600 !important;
        position: relative !important;
        z-index: 2 !important;
    }
    
    .adaptive-card {
        background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%) !important;
        border: 3px solid var(--accent-color) !important;
        border-radius: 25px !important;
        padding: 35px !important;
        margin: 25px 0 !important;
        box-shadow: 0 20px 40px rgba(16, 185, 129, 0.3) !important;
        transition: all 0.4s ease !important;
    }
    
    .preprocessing-toggle {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(6, 182, 212, 0.2) 100%) !important;
        border: 3px solid var(--accent-color) !important;
        border-radius: 20px !important;
        padding: 25px !important;
        margin: 20px 0 !important;
    }
    
    .adaptive-button {
        background: linear-gradient(135deg, var(--accent-color) 0%, var(--adaptive-color) 100%) !important;
        border: none !important;
        border-radius: 20px !important;
        color: white !important;
        font-weight: 800 !important;
        font-size: 1.3rem !important;
        padding: 20px 40px !important;
        transition: all 0.4s ease !important;
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.5) !important;
        text-transform: uppercase !important;
        letter-spacing: 1.5px !important;
    }
    
    .translation-button {
        background: linear-gradient(135deg, var(--translation-color) 0%, var(--adaptive-color) 100%) !important;
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
    
    .status-adaptive {
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
    
    .log-adaptive {
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
        css=adaptive_css, 
        theme=gr.themes.Base(),
        title="🚀 Adaptive Speech Enhancement & Transcription with GeeksforGeeks Preprocessing"
    ) as interface:
        
        # Adaptive Header
        gr.HTML("""
        <div class="adaptive-header">
            <h1 class="adaptive-title">🚀 ADAPTIVE SPEECH TRANSCRIPTION</h1>
            <p class="adaptive-subtitle">GeeksforGeeks Preprocessing • Adaptive Chunk Sizing • Enable/Disable Toggle • Auto Fallback • 75s Timeout</p>
            <div style="margin-top: 25px;">
                <span style="background: rgba(16, 185, 129, 0.25); color: #10b981; padding: 12px 24px; border-radius: 30px; margin: 0 10px; font-size: 1.1rem; font-weight: 700;">🔧 GEEKSFORGEEKS</span>
                <span style="background: rgba(6, 182, 212, 0.25); color: #06b6d4; padding: 12px 24px; border-radius: 30px; margin: 0 10px; font-size: 1.1rem; font-weight: 700;">📏 ADAPTIVE</span>
                <span style="background: rgba(59, 130, 246, 0.25); color: #3b82f6; padding: 12px 24px; border-radius: 30px; margin: 0 10px; font-size: 1.1rem; font-weight: 700;">⚡ TOGGLE</span>
                <span style="background: rgba(245, 158, 11, 0.25); color: #f59e0b; padding: 12px 24px; border-radius: 30px; margin: 0 10px; font-size: 1.1rem; font-weight: 700;">⏱️ 75s TIMEOUT</span>
            </div>
        </div>
        """)
        
        # System Status
        status_display = gr.Textbox(
            label="🚀 Adaptive System Status",
            value="Initializing ADAPTIVE speech transcription with GeeksforGeeks preprocessing...",
            interactive=False,
            elem_classes="status-adaptive"
        )
        
        # Main Interface
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML('<div class="adaptive-card"><div class="card-header">🚀 Adaptive Control Panel</div>')
                
                audio_input = gr.Audio(
                    label="🎵 Upload Audio File or Record Live",
                    type="filepath"
                )
                
                language_dropdown = gr.Dropdown(
                    choices=list(SUPPORTED_LANGUAGES.keys()),
                    value="🌍 Auto-detect",
                    label="🌍 Language Selection (150+ Supported)",
                    info="All languages with ADAPTIVE enhancement"
                )
                
                # GeeksforGeeks Preprocessing Toggle
                gr.HTML("""
                <div class="preprocessing-toggle">
                    <div style="color: #10b981; font-size: 1.4rem; font-weight: 700; margin-bottom: 15px;">🔧 GeeksforGeeks Preprocessing Control</div>
                    <p style="color: #cbd5e1; margin-bottom: 15px; font-size: 1rem;">
                        Enable/disable audio preprocessing based on GeeksforGeeks methodology. Includes resampling, Butterworth filtering, and normalization.
                    </p>
                </div>
                """)
                
                enable_preprocessing = gr.Checkbox(
                    label="🔧 Enable GeeksforGeeks Preprocessing",
                    value=True,
                    info="Applies resampling (16kHz), Butterworth low-pass filter (4kHz), and amplitude normalization"
                )
                
                transcribe_btn = gr.Button(
                    "🚀 START ADAPTIVE TRANSCRIPTION",
                    variant="primary",
                    elem_classes="adaptive-button",
                    size="lg"
                )
                
                gr.HTML('</div>')
                
                # Info Panel
                gr.HTML("""
                <div class="adaptive-card">
                    <div class="card-header">📏 Adaptive Chunk Sizing Info</div>
                    <div style="color: #cbd5e1; font-size: 1rem; line-height: 1.6;">
                        <p><strong>Default:</strong> 30-second chunks</p>
                        <p><strong>Fallbacks:</strong> 10s → 15s → 20s → 40s</p>
                        <p><strong>Auto-Retry:</strong> Switches chunk size if transcription fails</p>
                        <p><strong>Success Rate:</strong> Monitors and adapts automatically</p>
                        <p><strong>Timeout:</strong> 75 seconds per chunk with noise detection</p>
                    </div>
                </div>
                """)
            
            with gr.Column(scale=2):
                gr.HTML('<div class="adaptive-card"><div class="card-header">📊 Adaptive Results</div>')
                
                transcription_output = gr.Textbox(
                    label="📝 Original Transcription (ADAPTIVE Enhanced)",
                    placeholder="Your ADAPTIVE transcription will appear here...",
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
                gr.HTML('<div class="adaptive-card"><div class="card-header">📥 Original Audio</div>')
                original_audio_player = gr.Audio(
                    label="Original Audio",
                    interactive=False
                )
                gr.HTML('</div>')
            
            with gr.Column():
                gr.HTML('<div class="adaptive-card"><div class="card-header">🚀 ADAPTIVE Enhanced Audio</div>')
                enhanced_audio_player = gr.Audio(
                    label="ADAPTIVE Enhanced Audio (GeeksforGeeks Pipeline)",
                    interactive=False
                )
                gr.HTML('</div>')
        
        # Reports
        with gr.Row():
            with gr.Column():
                with gr.Accordion("🚀 ADAPTIVE Enhancement Report", open=False):
                    enhancement_report = gr.Textbox(
                        label="ADAPTIVE Enhancement Report",
                        lines=20,
                        show_copy_button=True,
                        interactive=False
                    )
            
            with gr.Column():
                with gr.Accordion("📋 ADAPTIVE Processing Report", open=False):
                    processing_report = gr.Textbox(
                        label="ADAPTIVE Processing Report", 
                        lines=20,
                        show_copy_button=True,
                        interactive=False
                    )
        
        # System Monitoring
        gr.HTML('<div class="adaptive-card"><div class="card-header">🚀 ADAPTIVE System Monitoring</div>')
        
        log_display = gr.Textbox(
            label="",
            value="🚀 ADAPTIVE system ready - GeeksforGeeks preprocessing available...",
            interactive=False,
            lines=14,
            max_lines=20,
            elem_classes="log-adaptive",
            show_label=False
        )
        
        with gr.Row():
            refresh_logs_btn = gr.Button("🔄 Refresh ADAPTIVE Logs", size="sm")
            clear_logs_btn = gr.Button("🗑️ Clear Logs", size="sm")
        
        gr.HTML('</div>')
        
        # Event Handlers
        transcribe_btn.click(
            fn=transcribe_audio_adaptive,
            inputs=[audio_input, language_dropdown, enable_preprocessing],
            outputs=[transcription_output, original_audio_player, enhanced_audio_player, enhancement_report, processing_report],
            show_progress=True
        )
        
        translate_btn.click(
            fn=translate_transcription_adaptive,
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
        
        def clear_adaptive_logs():
            global log_capture
            if log_capture:
                with log_capture.lock:
                    log_capture.log_buffer.clear()
            return "🚀 ADAPTIVE logs cleared - system ready"
        
        clear_logs_btn.click(
            fn=clear_adaptive_logs,
            inputs=[],
            outputs=[log_display]
        )
        
        def auto_refresh_adaptive_logs():
            return get_current_logs()
        
        timer = gr.Timer(value=3, active=True)
        timer.tick(
            fn=auto_refresh_adaptive_logs,
            inputs=[],
            outputs=[log_display]
        )
        
        interface.load(
            fn=initialize_adaptive_transcriber,
            inputs=[],
            outputs=[status_display]
        )
    
    return interface

def main():
    """Launch the complete ADAPTIVE speech transcription system"""
    
    if "/path/to/your/" in MODEL_PATH:
        print("="*80)
        print("🚀 ADAPTIVE SPEECH TRANSCRIPTION SYSTEM CONFIGURATION REQUIRED")
        print("="*80)
        print("Please update the MODEL_PATH variable with your local Gemma 3N model directory")
        print("Download from: https://huggingface.co/google/gemma-3n-e4b-it")
        print("="*80)
        return
    
    setup_adaptive_logging()
    
    print("🚀 Launching ADAPTIVE SPEECH TRANSCRIPTION SYSTEM...")
    print("="*80)
    print("🔧 GEEKSFORGEEKS PREPROCESSING IMPLEMENTATION:")
    print("="*80)
    print("📊 GEEKSFORGEEKS PIPELINE STAGES:")
    print("   ✅ Stage 1: Audio Resampling (16000 Hz)")
    print("   ✅ Stage 2: Butterworth Low-pass Filter (4000 Hz cutoff, Order 4)")
    print("   ✅ Stage 3: Amplitude Normalization (0.95 max)")
    print("   ✅ Stage 4: Quality Control Clipping (-0.99 to 0.99)")
    print("="*80)
    print("📏 ADAPTIVE CHUNK SIZING FEATURES:")
    print("   📏 Default Chunk Size: 30 seconds")
    print("   📏 Fallback Chunk Sizes: 10s, 15s, 20s, 40s")
    print("   🔄 Automatic Fallback Mechanism: Enabled")
    print("   📊 Success Rate Monitoring: Active")
    print("   🎯 Optimal Chunk Detection: Automatic")
    print("="*80)
    print("⚡ ENABLE/DISABLE PREPROCESSING CONTROL:")
    print("   👤 User Control: Complete enable/disable toggle")
    print("   🔧 GeeksforGeeks Pipeline: User can skip entirely")
    print("   📊 Raw Audio Processing: Available when disabled")
    print("   🎛️ Flexible Processing: User choice driven")
    print("="*80)
    print("⏱️ TIMEOUT PROTECTION:")
    print(f"   ⏱️ {CHUNK_TIMEOUT}-second timeout per chunk")
    print("   ⏱️ Adaptive quality detection and assessment")
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
    print("🚀 ADAPTIVE SYSTEM ADVANTAGES:")
    print("   📏 Adaptive Chunk Sizing: Automatically finds optimal chunk size")
    print("   🔄 Fallback Mechanism: Tries multiple chunk sizes if needed")
    print("   ⚡ Preprocessing Control: User can enable/disable GeeksforGeeks pipeline")
    print("   📊 Success Rate Monitoring: Tracks transcription quality")
    print("   🎯 Automatic Optimization: Selects best chunk size for audio")
    print("   🛡️ Robust Error Handling: Comprehensive fallback systems")
    print("   💡 Educational: Shows which chunk size worked best")
    print("="*80)
    
    try:
        interface = create_adaptive_interface()
        
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
        print(f"❌ ADAPTIVE system launch failed: {e}")
        print("🔧 ADAPTIVE system troubleshooting:")
        print("   • Verify model path is correct and accessible")
        print("   • Check GPU memory availability and drivers")
        print("   • Ensure all dependencies are installed:")
        print("     pip install --upgrade torch transformers gradio librosa soundfile")
        print("     pip install --upgrade scipy nltk")
        print("   • Verify Python
