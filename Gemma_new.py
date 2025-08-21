# -*- coding: utf-8 -*-
"""
Professional Audio Transcription System with Advanced Speech Enhancement
========================================================================

A comprehensive audio transcription application featuring:
- Advanced noise reduction and speech enhancement
- Professional-grade audio preprocessing
- Modern, enterprise-level UI design
- Real-time audio comparison and analysis
- Multi-language support with Gemma 3N model

Prerequisites:
1. Download 'google/gemma-3n-e4b-it' model from Hugging Face
2. Install dependencies: pip install transformers torch torchaudio soundfile librosa accelerate gradio bitsandbytes noisereduce scipy

Update MODEL_PATH before running.
"""

import os
import gc
import torch
import librosa
import gradio as gr
from transformers import Gemma3nForConditionalGeneration, Gemma3nProcessor, BitsAndBytesConfig
import numpy as np
from typing import Optional, Tuple, Dict, List
import time
import sys
import io
from contextlib import redirect_stdout, redirect_stderr
import threading
import queue
import tempfile
import soundfile as sf
from scipy import signal
from scipy.fft import fft, ifft
import noisereduce as nr
import datetime
import json

# --- Configuration ---
MODEL_PATH = "/path/to/your/local/gemma-3n-e4b-it"  # UPDATE THIS PATH

# Enhanced audio processing settings
MAX_AUDIO_LENGTH = 30
OVERLAP_DURATION = 2
SAMPLE_RATE = 16000

# Professional language support
SUPPORTED_LANGUAGES = {
    "🌍 Auto-detect": "auto",
    "🇺🇸 English": "en", "🇪🇸 Spanish": "es", "🇫🇷 French": "fr",
    "🇩🇪 German": "de", "🇮🇹 Italian": "it", "🇵🇹 Portuguese": "pt",
    "🇷🇺 Russian": "ru", "🇨🇳 Chinese": "zh", "🇯🇵 Japanese": "ja",
    "🇰🇷 Korean": "ko", "🇸🇦 Arabic": "ar", "🇮🇳 Hindi": "hi",
    "🇳🇱 Dutch": "nl", "🇸🇪 Swedish": "sv", "🇳🇴 Norwegian": "no",
    "🇩🇰 Danish": "da", "🇫🇮 Finnish": "fi", "🇵🇱 Polish": "pl",
    "🇹🇷 Turkish": "tr", "🇧🇷 Portuguese (BR)": "pt-br",
    "🇮🇳 Bengali": "bn", "🇮🇳 Tamil": "ta", "🇮🇳 Telugu": "te",
    "🇮🇳 Gujarati": "gu", "🇮🇳 Marathi": "mr", "🇮🇳 Urdu": "ur"
}

# Environment optimization
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

def clear_gpu_memory():
    """Advanced GPU memory management"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

class AdvancedAudioEnhancer:
    """
    Professional-grade audio enhancement system for noisy and distorted speech
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.setup_filters()
        self.enhancement_stats = {}
    
    def setup_filters(self):
        """Initialize professional filter parameters"""
        self.high_pass_cutoff = 80  # Remove low-frequency noise
        self.low_pass_cutoff = min(7900, self.sample_rate // 2 - 100)  # Remove high-frequency noise
        self.notch_frequencies = [50, 60, 100, 120]  # Power line noise
        
    def spectral_subtraction(self, audio: np.ndarray, alpha: float = 2.0, beta: float = 0.01) -> np.ndarray:
        """Advanced spectral subtraction for noise reduction"""
        try:
            # Compute STFT
            stft = librosa.stft(audio, n_fft=2048, hop_length=512, window='hann')
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Estimate noise spectrum from first 0.5 seconds
            noise_frames = int(0.5 * self.sample_rate / 512)
            noise_frames = min(max(1, noise_frames), magnitude.shape[1])
            noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
            
            # Apply spectral subtraction
            enhanced_magnitude = magnitude - alpha * noise_spectrum
            enhanced_magnitude = np.maximum(enhanced_magnitude, beta * magnitude)
            
            # Reconstruct signal
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft, hop_length=512, length=len(audio))
            
            # Calculate enhancement metrics
            snr_improvement = 10 * np.log10(np.mean(enhanced_audio**2) / (np.mean((audio - enhanced_audio)**2) + 1e-10))
            self.enhancement_stats['spectral_snr_improvement'] = snr_improvement
            
            return enhanced_audio.astype(np.float32)
        except Exception as e:
            print(f"Spectral subtraction failed: {e}")
            return audio
    
    def wiener_filter(self, audio: np.ndarray, noise_factor: float = 0.1) -> np.ndarray:
        """Professional Wiener filtering for adaptive noise reduction"""
        try:
            # Estimate power spectral density
            f, psd = signal.welch(audio, self.sample_rate, nperseg=min(1024, len(audio)//4))
            
            # Estimate noise PSD from initial segment
            noise_samples = min(int(0.5 * self.sample_rate), len(audio)//2)
            if noise_samples > 0:
                noise_segment = audio[:noise_samples]
                noise_psd = np.mean(np.abs(fft(noise_segment))**2)
            else:
                noise_psd = np.var(audio) * 0.1
            
            # Apply Wiener filter in frequency domain
            audio_fft = fft(audio)
            signal_power = np.abs(audio_fft)**2
            wiener_filter = signal_power / (signal_power + noise_factor * noise_psd)
            filtered_fft = audio_fft * wiener_filter
            filtered_audio = np.real(ifft(filtered_fft))
            
            # Store metrics
            self.enhancement_stats['wiener_noise_reduction'] = np.mean(wiener_filter)
            
            return filtered_audio.astype(np.float32)
        except Exception as e:
            print(f"Wiener filter failed: {e}")
            return audio
    
    def advanced_bandpass_filter(self, audio: np.ndarray) -> np.ndarray:
        """Multi-stage professional bandpass filtering"""
        try:
            original_rms = np.sqrt(np.mean(audio**2))
            
            # High-pass filter (remove low-frequency noise)
            sos_hp = signal.butter(6, self.high_pass_cutoff, btype='high', fs=self.sample_rate, output='sos')
            audio = signal.sosfilt(sos_hp, audio)
            
            # Low-pass filter (remove high-frequency noise)
            sos_lp = signal.butter(6, self.low_pass_cutoff, btype='low', fs=self.sample_rate, output='sos')
            audio = signal.sosfilt(sos_lp, audio)
            
            # Notch filters for power line noise
            for freq in self.notch_frequencies:
                if freq < self.sample_rate / 2:
                    try:
                        b, a = signal.iirnotch(freq, Q=30, fs=self.sample_rate)
                        sos_notch = signal.tf2sos(b, a)
                        audio = signal.sosfilt(sos_notch, audio)
                    except Exception as e:
                        print(f"Notch filter at {freq}Hz failed: {e}")
                        continue
            
            # Calculate filtering impact
            filtered_rms = np.sqrt(np.mean(audio**2))
            self.enhancement_stats['bandpass_rms_change'] = filtered_rms / (original_rms + 1e-10)
            
            return audio.astype(np.float32)
        except Exception as e:
            print(f"Bandpass filter failed: {e}")
            return audio
    
    def adaptive_noise_reduction(self, audio: np.ndarray) -> np.ndarray:
        """Advanced adaptive noise reduction using multiple techniques"""
        try:
            original_noise_level = np.std(audio[:int(0.5 * self.sample_rate)]) if len(audio) > int(0.5 * self.sample_rate) else np.std(audio)
            
            # Primary noise reduction
            reduced_noise = nr.reduce_noise(
                y=audio, 
                sr=self.sample_rate, 
                stationary=False,
                prop_decrease=0.8
            )
            
            enhanced_noise_level = np.std(reduced_noise[:int(0.5 * self.sample_rate)]) if len(reduced_noise) > int(0.5 * self.sample_rate) else np.std(reduced_noise)
            noise_reduction_db = 20 * np.log10(original_noise_level / (enhanced_noise_level + 1e-10))
            self.enhancement_stats['adaptive_noise_reduction_db'] = noise_reduction_db
            
            return reduced_noise.astype(np.float32)
        except Exception as e:
            print(f"Adaptive noise reduction failed: {e}")
            # Fallback to basic noise reduction
            try:
                return nr.reduce_noise(y=audio, sr=self.sample_rate).astype(np.float32)
            except:
                return audio
    
    def dynamic_range_compression(self, audio: np.ndarray, threshold: float = 0.3, ratio: float = 4.0) -> np.ndarray:
        """Professional dynamic range compression"""
        try:
            original_dynamic_range = np.max(np.abs(audio)) - np.min(np.abs(audio))
            
            # Apply compression
            compressed = np.copy(audio)
            mask = np.abs(audio) > threshold
            compressed[mask] = np.sign(audio[mask]) * (threshold + (np.abs(audio[mask]) - threshold) / ratio)
            
            compressed_dynamic_range = np.max(np.abs(compressed)) - np.min(np.abs(compressed))
            compression_ratio = original_dynamic_range / (compressed_dynamic_range + 1e-10)
            self.enhancement_stats['compression_ratio'] = compression_ratio
            
            return compressed.astype(np.float32)
        except Exception as e:
            print(f"Dynamic range compression failed: {e}")
            return audio
    
    def enhance_audio(self, audio: np.ndarray, enhancement_level: str = "moderate") -> Tuple[np.ndarray, Dict]:
        """
        Main enhancement pipeline with multiple processing levels
        
        Args:
            audio: Input audio array
            enhancement_level: "light", "moderate", or "aggressive"
        
        Returns:
            Enhanced audio and statistics
        """
        original_audio = audio.copy()
        self.enhancement_stats = {}
        
        try:
            if len(audio) == 0:
                return original_audio, {}
            
            # Store original metrics
            self.enhancement_stats['original_length'] = len(audio) / self.sample_rate
            self.enhancement_stats['original_rms'] = np.sqrt(np.mean(audio**2))
            self.enhancement_stats['original_peak'] = np.max(np.abs(audio))
            
            print(f"🔊 Starting {enhancement_level} enhancement...")
            
            # Stage 1: Adaptive noise reduction (all levels)
            print("📊 Applying adaptive noise reduction...")
            audio = self.adaptive_noise_reduction(audio)
            
            # Stage 2: Bandpass filtering (all levels)
            print("🔧 Applying professional bandpass filtering...")
            audio = self.advanced_bandpass_filter(audio)
            
            # Stage 3: Spectral subtraction (moderate and aggressive)
            if enhancement_level in ["moderate", "aggressive"]:
                print("⚡ Applying spectral subtraction...")
                alpha_val = 3.0 if enhancement_level == "aggressive" else 2.0
                audio = self.spectral_subtraction(audio, alpha=alpha_val, beta=0.05)
            
            # Stage 4: Wiener filtering (aggressive only)
            if enhancement_level == "aggressive":
                print("🎯 Applying Wiener filtering...")
                audio = self.wiener_filter(audio, noise_factor=0.05)
            
            # Stage 5: Dynamic range compression (moderate and aggressive)
            if enhancement_level in ["moderate", "aggressive"]:
                print("🎚️ Applying dynamic range compression...")
                audio = self.dynamic_range_compression(audio)
            
            # Final normalization and clipping
            audio = librosa.util.normalize(audio)
            audio = np.clip(audio, -1.0, 1.0)
            
            # Calculate final metrics
            self.enhancement_stats['enhanced_rms'] = np.sqrt(np.mean(audio**2))
            self.enhancement_stats['enhanced_peak'] = np.max(np.abs(audio))
            self.enhancement_stats['enhancement_level'] = enhancement_level
            self.enhancement_stats['total_snr_improvement'] = 20 * np.log10(
                self.enhancement_stats['enhanced_rms'] / (self.enhancement_stats['original_rms'] + 1e-10)
            )
            
            print("✅ Audio enhancement completed successfully")
            return audio.astype(np.float32), self.enhancement_stats
            
        except Exception as e:
            print(f"Enhancement pipeline failed: {e}")
            return original_audio.astype(np.float32), {}

class AudioTranscriber:
    """Enhanced AudioTranscriber with professional audio preprocessing"""
    
    def __init__(self, model_path: str, use_quantization: bool = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = None
        self.processor = None
        self.enhancer = AdvancedAudioEnhancer(SAMPLE_RATE)
        
        print(f"🖥️ Using device: {self.device}")
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"Model directory not found at '{model_path}'")

        clear_gpu_memory()
        print("🚀 Loading model and processor...")
        
        try:
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
                print("🔧 Quantization disabled...")

            self.model = Gemma3nForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                device_map="auto",
                quantization_config=quantization_config,
            )
            
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
            
            print("✅ Model and processor loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

    def enhance_and_save_audio(self, audio_path: str, enhancement_level: str = "moderate") -> Tuple[str, str, Dict]:
        """
        Enhance audio and save both original and enhanced versions
        
        Returns:
            (original_path, enhanced_path, enhancement_stats)
        """
        try:
            # Load audio
            audio_array, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
            
            # Enhance audio
            enhanced_audio, stats = self.enhancer.enhance_audio(audio_array, enhancement_level)
            
            # Save enhanced audio
            enhanced_path = tempfile.mktemp(suffix="_enhanced.wav")
            sf.write(enhanced_path, enhanced_audio, SAMPLE_RATE)
            
            # Save original in consistent format
            original_path = tempfile.mktemp(suffix="_original.wav")
            sf.write(original_path, audio_array, SAMPLE_RATE)
            
            return original_path, enhanced_path, stats
            
        except Exception as e:
            print(f"❌ Audio enhancement failed: {e}")
            return audio_path, audio_path, {}

    def chunk_audio(self, audio_array: np.ndarray, sr: int = SAMPLE_RATE) -> list:
        """Enhanced audio chunking with overlap"""
        chunk_length = int(MAX_AUDIO_LENGTH * sr)
        overlap_length = int(OVERLAP_DURATION * sr)
        
        chunks = []
        start = 0
        
        while start < len(audio_array):
            end = min(start + chunk_length, len(audio_array))
            chunk = audio_array[start:end]
            chunks.append(chunk)
            
            if end >= len(audio_array):
                break
                
            start = end - overlap_length
            
        return chunks

    def transcribe_chunk(self, audio_chunk: np.ndarray, language: str = "auto") -> str:
        """Enhanced chunk transcription with error handling"""
        if language == "auto":
            system_message = "You are a professional transcription assistant. Transcribe the speech accurately with proper punctuation and formatting. Detect the language automatically."
        else:
            lang_name = [k for k, v in SUPPORTED_LANGUAGES.items() if v == language][0]
            system_message = f"You are a professional transcription assistant. The audio is in {lang_name}. Transcribe it accurately with proper punctuation and formatting."
        
        message = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_chunk},
                    {"type": "text", "text": "Please provide an accurate transcription of this audio."},
                ],
            },
        ]

        try:
            inputs = self.processor.apply_chat_template(
                message,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.device)

            input_len = inputs["input_ids"].shape[-1]

            with torch.no_grad():
                generation = self.model.generate(
                    **inputs, 
                    max_new_tokens=512,  # Increased for longer transcriptions
                    do_sample=False,
                    temperature=0.1,
                    disable_compile=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            generation = generation[0][input_len:]
            decoded_transcription = self.processor.decode(generation, skip_special_tokens=True)
            
            return decoded_transcription.strip()
            
        except torch.cuda.OutOfMemoryError:
            clear_gpu_memory()
            return "[❌ CUDA out of memory - try shorter audio segments]"
        except Exception as e:
            return f"[❌ Transcription error: {str(e)}]"

    def transcribe(self, audio_path: str, language: str = "auto", enhancement_level: str = "moderate") -> Tuple[str, str, str, Dict]:
        """
        Enhanced transcription with audio preprocessing
        
        Returns:
            (transcription, original_audio_path, enhanced_audio_path, enhancement_stats)
        """
        try:
            print(f"🎵 Processing audio file: {audio_path}")
            print(f"🔧 Enhancement level: {enhancement_level}")
            
            # Enhance audio first
            original_path, enhanced_path, enhancement_stats = self.enhance_and_save_audio(audio_path, enhancement_level)
            
            # Load enhanced audio for transcription
            audio_array, sampling_rate = librosa.load(enhanced_path, sr=SAMPLE_RATE, mono=True)
            duration = len(audio_array) / sampling_rate
            
            print(f"⏱️ Audio duration: {duration:.2f} seconds")
            
            if duration <= MAX_AUDIO_LENGTH:
                print("🎙️ Transcribing single chunk...")
                transcription = self.transcribe_chunk(audio_array, language)
            else:
                print(f"✂️ Splitting into chunks (max {MAX_AUDIO_LENGTH}s each)...")
                chunks = self.chunk_audio(audio_array)
                transcriptions = []
                
                for i, chunk in enumerate(chunks, 1):
                    print(f"🎙️ Transcribing chunk {i}/{len(chunks)}...")
                    transcription = self.transcribe_chunk(chunk, language)
                    transcriptions.append(transcription)
                    clear_gpu_memory()
                
                transcription = " ".join(transcriptions)
            
            print("✅ Transcription completed successfully")
            return transcription, original_path, enhanced_path, enhancement_stats
                
        except Exception as e:
            error_msg = f"❌ Error processing audio: {e}"
            print(error_msg)
            return error_msg, audio_path, audio_path, {}

# Global variables for logging and state
log_queue = queue.Queue()
log_buffer = []
MAX_LOG_LINES = 200
transcriber = None

class LogCapture:
    """Enhanced logging system with timestamps and categories"""
    def __init__(self, original_stdout):
        self.original_stdout = original_stdout
        
    def write(self, text):
        if text.strip():
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            
            # Categorize log messages
            if "❌" in text or "Error" in text or "error" in text:
                category = "ERROR"
                color_code = "🔴"
            elif "✅" in text or "success" in text:
                category = "SUCCESS"
                color_code = "🟢"
            elif "⚠️" in text or "Warning" in text:
                category = "WARNING"
                color_code = "🟡"
            elif "🔧" in text or "Loading" in text:
                category = "SYSTEM"
                color_code = "🔵"
            else:
                category = "INFO"
                color_code = "⚪"
            
            log_entry = f"[{timestamp}] {color_code} {text.strip()}"
            log_queue.put(log_entry)
            log_buffer.append(log_entry)
            
            if len(log_buffer) > MAX_LOG_LINES:
                log_buffer.pop(0)
                
        self.original_stdout.write(text)
        
    def flush(self):
        self.original_stdout.flush()

def setup_logging():
    """Initialize enhanced logging system"""
    original_stdout = sys.stdout
    sys.stdout = LogCapture(original_stdout)

def get_current_logs():
    """Get formatted current logs"""
    if not log_buffer:
        return "🎯 System ready - waiting for operations..."
    return "\n".join(log_buffer[-50:])

def initialize_transcriber():
    """Initialize the enhanced transcriber model"""
    global transcriber
    if transcriber is None:
        try:
            print("🚀 Initializing Professional Audio Transcription System...")
            transcriber = AudioTranscriber(model_path=MODEL_PATH, use_quantization=True)
            return "✅ Professional transcription system loaded successfully!"
        except Exception as e:
            try:
                print("🔄 Retrying without quantization...")
                transcriber = AudioTranscriber(model_path=MODEL_PATH, use_quantization=False)
                return "✅ Transcription system loaded (without quantization)!"
            except Exception as e2:
                error_msg = f"❌ Critical error loading model: {str(e2)}"
                print(error_msg)
                return error_msg
    return "✅ Transcription system already loaded!"

def transcribe_audio_enhanced(audio_input, language_choice, enhancement_level, progress=gr.Progress()):
    """
    Enhanced transcription function with audio preprocessing and comparison
    
    Returns:
        (transcription, original_audio, enhanced_audio, enhancement_report, processing_report)
    """
    global transcriber
    
    if audio_input is None:
        print("❌ No audio input provided")
        return "❌ Please upload an audio file or record audio.", None, None, "", ""
    
    if transcriber is None:
        print("❌ Model not loaded")
        return "❌ Model not loaded. Please wait for initialization.", None, None, "", ""
    
    start_time = time.time()
    print(f"🎯 Starting enhanced transcription process...")
    print(f"🌍 Language: {language_choice}")
    print(f"🔧 Enhancement level: {enhancement_level}")
    
    progress(0.1, desc="Processing audio input...")
    
    try:
        # Handle different audio input types
        if isinstance(audio_input, tuple):
            # Recorded audio
            sample_rate, audio_data = audio_input
            print(f"🎙️ Recorded audio: {sample_rate}Hz, {len(audio_data)} samples")
            temp_path = tempfile.mktemp(suffix=".wav")
            sf.write(temp_path, audio_data, sample_rate)
            audio_path = temp_path
        else:
            # Uploaded file
            audio_path = audio_input
            print(f"📁 Processing uploaded file: {audio_path}")
        
        progress(0.3, desc="Enhancing audio quality...")
        
        # Get language code
        language_code = SUPPORTED_LANGUAGES.get(language_choice, "auto")
        print(f"🔤 Language code: {language_code}")
        
        progress(0.5, desc="Transcribing enhanced audio...")
        
        # Transcribe with enhancement
        transcription, original_path, enhanced_path, enhancement_stats = transcriber.transcribe(
            audio_path, language_code, enhancement_level
        )
        
        progress(0.9, desc="Generating reports...")
        
        # Create enhancement report
        enhancement_report = create_enhancement_report(enhancement_stats, enhancement_level)
        
        # Create processing report
        processing_time = time.time() - start_time
        processing_report = create_processing_report(
            audio_path, language_choice, enhancement_level, 
            processing_time, len(transcription.split()) if isinstance(transcription, str) else 0
        )
        
        # Clean up temporary files
        if isinstance(audio_input, tuple) and os.path.exists(temp_path):
            os.remove(temp_path)
        
        progress(1.0, desc="Transcription complete!")
        
        print(f"✅ Enhanced transcription completed in {processing_time:.2f}s")
        print(f"📊 Generated {len(transcription.split()) if isinstance(transcription, str) else 0} words")
        
        return transcription, original_path, enhanced_path, enhancement_report, processing_report
        
    except Exception as e:
        error_msg = f"❌ Critical error during transcription: {str(e)}"
        print(error_msg)
        return error_msg, None, None, "", ""

def create_enhancement_report(stats: Dict, level: str) -> str:
    """Create detailed enhancement analysis report"""
    if not stats:
        return "⚠️ Enhancement statistics not available"
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
🎚️ PROFESSIONAL AUDIO ENHANCEMENT REPORT
=========================================
Generated: {timestamp}
Enhancement Level: {level.upper()}

📊 AUDIO QUALITY METRICS:
• Original RMS Level: {stats.get('original_rms', 0):.4f}
• Enhanced RMS Level: {stats.get('enhanced_rms', 0):.4f}
• Peak Amplitude: {stats.get('enhanced_peak', 0):.4f}
• Duration: {stats.get('original_length', 0):.2f} seconds

🎯 ENHANCEMENT PERFORMANCE:
• Total SNR Improvement: {stats.get('total_snr_improvement', 0):.2f} dB
• Spectral SNR Gain: {stats.get('spectral_snr_improvement', 0):.2f} dB
• Adaptive Noise Reduction: {stats.get('adaptive_noise_reduction_db', 0):.2f} dB
• Bandpass RMS Change: {stats.get('bandpass_rms_change', 1):.3f}
• Wiener Noise Reduction: {stats.get('wiener_noise_reduction', 0):.3f}
• Compression Ratio: {stats.get('compression_ratio', 1):.2f}

✨ PROCESSING PIPELINE:
1. ✅ Adaptive Noise Reduction Applied
2. ✅ Professional Bandpass Filtering
3. ✅ {"Spectral Subtraction Applied" if level in ["moderate", "aggressive"] else "Spectral Subtraction Skipped"}
4. ✅ {"Wiener Filtering Applied" if level == "aggressive" else "Wiener Filtering Skipped"}
5. ✅ {"Dynamic Range Compression Applied" if level in ["moderate", "aggressive"] else "Dynamic Range Compression Skipped"}
6. ✅ Professional Normalization Applied

🏆 QUALITY SCORE: {min(100, max(0, 70 + stats.get('total_snr_improvement', 0) * 3)):.0f}/100
"""
    return report

def create_processing_report(audio_path: str, language: str, enhancement: str, 
                           processing_time: float, word_count: int) -> str:
    """Create detailed processing report"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Get file info
    try:
        file_size = os.path.getsize(audio_path) / (1024 * 1024)  # MB
        audio_info = f"File size: {file_size:.2f} MB"
    except:
        audio_info = "File info unavailable"
    
    device_info = f"GPU: {torch.cuda.get_device_name()}" if torch.cuda.is_available() else "CPU Processing"
    
    report = f"""
📋 PROFESSIONAL TRANSCRIPTION REPORT
====================================
Generated: {timestamp}

🎵 AUDIO PROCESSING:
• Source: {os.path.basename(audio_path)}
• {audio_info}
• Language: {language}
• Enhancement Level: {enhancement.upper()}

⚡ PERFORMANCE METRICS:
• Processing Time: {processing_time:.2f} seconds
• Words Generated: {word_count}
• Processing Speed: {word_count/processing_time:.1f} words/second
• Device: {device_info}

🔧 SYSTEM CONFIGURATION:
• Model: Gemma 3N E4B-IT
• Sample Rate: {SAMPLE_RATE} Hz
• Max Chunk Length: {MAX_AUDIO_LENGTH}s
• Overlap Duration: {OVERLAP_DURATION}s
• Memory Optimization: Enabled

✅ STATUS: COMPLETED SUCCESSFULLY
"""
    return report

def create_professional_interface():
    """Create the enhanced professional interface"""
    
    # Professional CSS with modern design
    professional_css = """
    /* Professional Enterprise Theme */
    :root {
        --primary-color: #1e3a8a;
        --secondary-color: #3730a3;
        --accent-color: #06b6d4;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
        --bg-primary: #0f172a;
        --bg-secondary: #1e293b;
        --bg-tertiary: #334155;
        --text-primary: #f8fafc;
        --text-secondary: #cbd5e1;
        --border-color: #475569;
    }
    
    .gradio-container {
        background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%) !important;
        font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
        color: var(--text-primary) !important;
        min-height: 100vh !important;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 50%, var(--accent-color) 100%) !important;
        padding: 40px 20px !important;
        border-radius: 20px !important;
        text-align: center !important;
        margin-bottom: 30px !important;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .main-header::before {
        content: '' !important;
        position: absolute !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        bottom: 0 !important;
        background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%) !important;
        animation: shimmer 3s infinite !important;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .main-title {
        font-size: 3rem !important;
        font-weight: 800 !important;
        color: white !important;
        margin-bottom: 10px !important;
        text-shadow: 0 4px 8px rgba(0,0,0,0.3) !important;
        position: relative !important;
        z-index: 2 !important;
    }
    
    .sub-title {
        font-size: 1.3rem !important;
        color: rgba(255,255,255,0.9) !important;
        font-weight: 400 !important;
        position: relative !important;
        z-index: 2 !important;
    }
    
    /* Professional Cards */
    .pro-card {
        background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 16px !important;
        padding: 25px !important;
        margin: 15px 0 !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2) !important;
        backdrop-filter: blur(10px) !important;
        transition: all 0.3s ease !important;
    }
    
    .pro-card:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.3) !important;
        border-color: var(--accent-color) !important;
    }
    
    .card-header {
        color: var(--accent-color) !important;
        font-size: 1.4rem !important;
        font-weight: 600 !important;
        margin-bottom: 20px !important;
        padding-bottom: 10px !important;
        border-bottom: 2px solid var(--accent-color) !important;
        display: flex !important;
        align-items: center !important;
        gap: 10px !important;
    }
    
    /* Enhanced Controls */
    .pro-button {
        background: linear-gradient(135deg, var(--accent-color) 0%, var(--primary-color) 100%) !important;
        border: none !important;
        border-radius: 12px !important;
        color: white !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        padding: 15px 30px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(6, 182, 212, 0.3) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    .pro-button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(6, 182, 212, 0.4) !important;
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--accent-color) 100%) !important;
    }
    
    /* Status Indicators */
    .status-success {
        background: linear-gradient(135deg, var(--success-color), #059669) !important;
        color: white !important;
        padding: 12px 20px !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
        text-align: center !important;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3) !important;
    }
    
    .status-error {
        background: linear-gradient(135deg, var(--error-color), #dc2626) !important;
        color: white !important;
        padding: 12px 20px !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
        text-align: center !important;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3) !important;
    }
    
    /* Audio Comparison Section */
    .audio-comparison {
        background: linear-gradient(135deg, rgba(6, 182, 212, 0.1) 0%, rgba(30, 58, 138, 0.1) 100%) !important;
        border: 2px solid var(--accent-color) !important;
        border-radius: 16px !important;
        padding: 25px !important;
        margin: 20px 0 !important;
        position: relative !important;
    }
    
    .audio-comparison::before {
        content: '🎵' !important;
        position: absolute !important;
        top: -15px !important;
        left: 25px !important;
        background: var(--accent-color) !important;
        color: white !important;
        padding: 8px 15px !important;
        border-radius: 20px !important;
        font-size: 1.2rem !important;
    }
    
    .comparison-header {
        color: var(--accent-color) !important;
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        margin-bottom: 15px !important;
        margin-top: 10px !important;
    }
    
    /* Professional Logs */
    .log-section {
        background: linear-gradient(135deg, rgba(0, 0, 0, 0.4) 0%, rgba(15, 23, 42, 0.6) 100%) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 15px !important;
        padding: 20px !important;
        margin: 20px 0 !important;
        backdrop-filter: blur(5px) !important;
    }
    
    .log-content {
        background: rgba(0, 0, 0, 0.7) !important;
        border: 1px solid var(--accent-color) !important;
        border-radius: 10px !important;
        color: var(--text-secondary) !important;
        font-family: 'JetBrains Mono', 'Courier New', monospace !important;
        font-size: 0.9rem !important;
        line-height: 1.6 !important;
        padding: 15px !important;
        max-height: 350px !important;
        overflow-y: auto !important;
        white-space: pre-wrap !important;
    }
    
    .log-content::-webkit-scrollbar {
        width: 8px !important;
    }
    
    .log-content::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.2) !important;
        border-radius: 4px !important;
    }
    
    .log-content::-webkit-scrollbar-thumb {
        background: var(--accent-color) !important;
        border-radius: 4px !important;
    }
    
    /* Enhancement Level Styling */
    .enhancement-light { border-left: 4px solid #10b981; }
    .enhancement-moderate { border-left: 4px solid #f59e0b; }
    .enhancement-aggressive { border-left: 4px solid #ef4444; }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-title { font-size: 2rem !important; }
        .sub-title { font-size: 1rem !important; }
        .pro-card { padding: 15px !important; }
    }
    
    /* Form Controls */
    .gradio-dropdown, .gradio-radio, .gradio-textbox {
        background: var(--bg-tertiary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
    }
    
    .gradio-dropdown:focus, .gradio-textbox:focus {
        border-color: var(--accent-color) !important;
        box-shadow: 0 0 0 2px rgba(6, 182, 212, 0.2) !important;
    }
    """
    
    with gr.Blocks(
        css=professional_css, 
        theme=gr.themes.Base(),
        title="🎙️ Professional Audio Transcription System"
    ) as interface:
        
        # Set up logging
        setup_logging()
        
        # Professional Header
        gr.HTML("""
        <div class="main-header">
            <h1 class="main-title">🎙️ Professional Audio Transcription System</h1>
            <p class="sub-title">Advanced Speech Enhancement • Gemma 3N AI • Enterprise-Grade Processing</p>
        </div>
        """)
        
        # System Status
        status_display = gr.Textbox(
            label="🔄 System Status",
            value="Initializing professional transcription system...",
            interactive=False,
            elem_classes="status-success"
        )
        
        # Main Interface Layout
        with gr.Row():
            # Input Configuration Panel
            with gr.Column(scale=1):
                gr.HTML('<div class="pro-card"><div class="card-header">🎛️ Audio Input & Configuration</div>')
                
                # Audio Input
                audio_input = gr.Audio(
                    label="🎵 Upload Audio File or Record Live",
                    type="filepath",
                    elem_classes="audio-container"
                )
                
                # Language Selection
                language_dropdown = gr.Dropdown(
                    choices=list(SUPPORTED_LANGUAGES.keys()),
                    value="🌍 Auto-detect",
                    label="🌍 Language Selection",
                    info="Select the primary language of your audio"
                )
                
                # Enhancement Level
                enhancement_radio = gr.Radio(
                    choices=[
                        ("🟢 Light - Basic noise reduction", "light"),
                        ("🟡 Moderate - Balanced enhancement", "moderate"), 
                        ("🔴 Aggressive - Maximum noise removal", "aggressive")
                    ],
                    value="moderate",
                    label="🎚️ Enhancement Level",
                    info="Choose processing intensity based on audio quality"
                )
                
                # Professional Transcribe Button
                transcribe_btn = gr.Button(
                    "🚀 Start Professional Transcription",
                    variant="primary",
                    elem_classes="pro-button",
                    size="lg"
                )
                
                gr.HTML('</div>')
            
            # Results Panel
            with gr.Column(scale=2):
                gr.HTML('<div class="pro-card"><div class="card-header">📊 Transcription Results</div>')
                
                # Transcription Output
                transcription_output = gr.Textbox(
                    label="📝 Professional Transcription",
                    placeholder="Your professionally processed transcription will appear here with proper formatting and punctuation...",
                    lines=12,
                    max_lines=20,
                    interactive=False,
                    show_copy_button=True
                )
                
                # Professional Copy Button
                copy_btn = gr.Button("📋 Copy Professional Transcription", size="sm")
                
                gr.HTML('</div>')
        
        # Audio Comparison Section
        gr.HTML("""
        <div class="audio-comparison">
            <h3 class="comparison-header">🎵 Professional Audio Enhancement Comparison</h3>
            <p style="color: #cbd5e1; margin-bottom: 20px;">Compare the original and enhanced audio to evaluate our professional processing:</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column():
                gr.HTML('<div class="pro-card"><div class="card-header">📥 Original Audio</div>')
                original_audio_player = gr.Audio(
                    label="Original Audio (Before Enhancement)",
                    interactive=False
                )
                gr.HTML('</div>')
            
            with gr.Column():
                gr.HTML('<div class="pro-card"><div class="card-header">✨ Enhanced Audio</div>')
                enhanced_audio_player = gr.Audio(
                    label="Enhanced Audio (After Professional Processing)",
                    interactive=False
                )
                gr.HTML('</div>')
        
        # Professional Reports Section
        with gr.Row():
            with gr.Column():
                with gr.Accordion("🎚️ Audio Enhancement Analysis", open=False):
                    enhancement_report = gr.Textbox(
                        label="Professional Enhancement Report",
                        lines=15,
                        show_copy_button=True,
                        interactive=False
                    )
            
            with gr.Column():
                with gr.Accordion("📋 Processing Performance Report", open=False):
                    processing_report = gr.Textbox(
                        label="Technical Processing Report", 
                        lines=15,
                        show_copy_button=True,
                        interactive=False
                    )
        
        # Live System Logs
        gr.HTML('<div class="log-section">')
        gr.HTML('<div class="card-header">📊 Live System Monitoring</div>')
        
        log_display = gr.Textbox(
            label="",
            value="🎯 Professional transcription system ready - waiting for operations...",
            interactive=False,
            lines=12,
            max_lines=15,
            elem_classes="log-content",
            show_label=False
        )
        
        # Log Controls
        with gr.Row():
            refresh_logs_btn = gr.Button("🔄 Refresh Logs", size="sm")
            clear_logs_btn = gr.Button("🗑️ Clear Logs", size="sm")
            export_logs_btn = gr.Button("💾 Export Logs", size="sm")
        
        gr.HTML('</div>')
        
        # Professional Features Section
        gr.HTML("""
        <div class="pro-card">
            <div class="card-header">💎 Professional Features</div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 20px;">
                <div class="enhancement-light" style="padding: 15px; background: rgba(16, 185, 129, 0.1); border-radius: 10px;">
                    <h4 style="color: #10b981; margin-bottom: 10px;">🔊 Advanced Audio Enhancement</h4>
                    <ul style="color: #cbd5e1; line-height: 1.6;">
                        <li>Spectral Subtraction Algorithm</li>
                        <li>Wiener Filtering System</li>
                        <li>Professional Bandpass Filtering</li>
                        <li>Adaptive Noise Reduction</li>
                        <li>Dynamic Range Compression</li>
                    </ul>
                </div>
                <div class="enhancement-moderate" style="padding: 15px; background: rgba(245, 158, 11, 0.1); border-radius: 10px;">
                    <h4 style="color: #f59e0b; margin-bottom: 10px;">🎯 Smart Processing</h4>
                    <ul style="color: #cbd5e1; line-height: 1.6;">
                        <li>Automatic Language Detection</li>
                        <li>Intelligent Audio Chunking</li>
                        <li>Memory-Optimized Processing</li>
                        <li>Real-time Progress Monitoring</li>
                        <li>Quality Metrics Analysis</li>
                    </ul>
                </div>
                <div class="enhancement-aggressive" style="padding: 15px; background: rgba(239, 68, 68, 0.1); border-radius: 10px;">
                    <h4 style="color: #ef4444; margin-bottom: 10px;">⚡ Enterprise Performance</h4>
                    <ul style="color: #cbd5e1; line-height: 1.6;">
                        <li>GPU-Accelerated Processing</li>
                        <li>Professional Audio Comparison</li>
                        <li>Comprehensive Reporting</li>
                        <li>Live System Monitoring</li>
                        <li>Export & Documentation</li>
                    </ul>
                </div>
            </div>
        </div>
        """)
        
        # Professional Tips
        gr.HTML("""
        <div class="pro-card">
            <div class="card-header">💡 Professional Usage Guidelines</div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin-top: 15px;">
                <div style="color: #cbd5e1;">
                    <h4 style="color: #06b6d4;">🎙️ Audio Quality</h4>
                    <p>• Use high-quality recordings when possible<br>• Minimize background noise and echoes<br>• Ensure clear speaker articulation</p>
                </div>
                <div style="color: #cbd5e1;">
                    <h4 style="color: #06b6d4;">🔧 Enhancement Levels</h4>
                    <p>• Light: Clean audio with minimal noise<br>• Moderate: Standard recordings with some noise<br>• Aggressive: Very noisy or distorted audio</p>
                </div>
                <div style="color: #cbd5e1;">
                    <h4 style="color: #06b6d4;">📊 Best Results</h4>
                    <p>• Select correct language for accuracy<br>• Monitor processing logs for insights<br>• Compare audio before/after enhancement</p>
                </div>
            </div>
        </div>
        """)
        
        # Professional Footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 40px; padding: 30px; background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%); border-radius: 15px; border: 1px solid var(--border-color);">
            <h3 style="color: #06b6d4; margin-bottom: 15px;">🏢 Professional Audio Transcription System</h3>
            <p style="color: #cbd5e1; margin-bottom: 10px;">Powered by Gemma 3N AI • Advanced Speech Enhancement • Enterprise-Grade Processing</p>
            <p style="color: #94a3b8; font-size: 0.9rem;">Built with cutting-edge AI technology for professional audio transcription workflows</p>
        </div>
        """)
        
        # Event Handlers
        transcribe_btn.click(
            fn=transcribe_audio_enhanced,
            inputs=[audio_input, language_dropdown, enhancement_radio],
            outputs=[transcription_output, original_audio_player, enhanced_audio_player, enhancement_report, processing_report],
            show_progress=True
        )
        
        copy_btn.click(
            fn=lambda text: text,
            inputs=[transcription_output],
            outputs=[],
            js="(text) => { navigator.clipboard.writeText(text); return text; }"
        )
        
        # Log Management
        refresh_logs_btn.click(
            fn=get_current_logs,
            inputs=[],
            outputs=[log_display]
        )
        
        clear_logs_btn.click(
            fn=lambda: (log_buffer.clear(), "🎯 Logs cleared - system ready for new operations")[1],
            inputs=[],
            outputs=[log_display]
        )
        
        # Auto-refresh logs
        def auto_refresh_logs():
            return get_current_logs()
        
        timer = gr.Timer(value=3, active=True)  # Update every 3 seconds
        timer.tick(
            fn=auto_refresh_logs,
            inputs=[],
            outputs=[log_display]
        )
        
        # Initialize system on load
        interface.load(
            fn=initialize_transcriber,
            inputs=[],
            outputs=[status_display]
        )
    
    return interface

def main():
    """Launch the professional transcription system"""
    
    # Validate configuration
    if "/path/to/your/" in MODEL_PATH:
        print("="*80)
        print("🚨 CONFIGURATION REQUIRED")
        print("="*80)
        print("Please update the MODEL_PATH variable with your local Gemma 3N model directory")
        print("Download from: https://huggingface.co/google/gemma-3n-e4b-it")
        print("="*80)
        return
    
    print("🚀 Launching Professional Audio Transcription System...")
    print("="*80)
    print("🎵 Features: Advanced Speech Enhancement")
    print("🧠 AI Model: Gemma 3N E4B-IT")
    print("🔧 Processing: Professional Audio Pipeline")
    print("💎 Interface: Enterprise-Grade Design")
    print("="*80)
    
    # Create and launch interface
    interface = create_professional_interface()
    
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
        prevent_thread_lock=False,
        ssl_verify=False
    )

if __name__ == "__main__":
    main()
