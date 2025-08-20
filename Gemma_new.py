"""
Professional Audio Transcription System with Gemma3n-e4b-it - FIXED VERSION
Author: Advanced AI Audio Processing System
Features: Speech Enhancement, Noise Removal, Long Audio Chunking, Multi-language Support
Fixed: Filter frequency issues, deprecated functions, added enhanced audio preview
"""

import os
import gc
import torch
import librosa
import numpy as np
import soundfile as sf
import gradio as gr
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import warnings
import tempfile
import shutil
warnings.filterwarnings("ignore")

# Audio processing libraries
from scipy import signal
from scipy.fft import fft, ifft
import noisereduce as nr

# Transformers for Gemma3n
from transformers import (
    AutoProcessor, 
    AutoModelForCausalLM,  # Fixed: Use AutoModelForCausalLM instead of Gemma3nForConditionalGeneration
    pipeline
)

# Configuration
class Config:
    # Model paths (set these to your local model directories)
    MODEL_PATH = "/path/to/local/models/google-gemma-3n-e4b-it"  # Update this path
    PROCESSOR_PATH = "/path/to/local/models/google-gemma-3n-e4b-it"  # Update this path
    
    # Audio processing parameters
    SAMPLE_RATE = 16000
    CHUNK_LENGTH = 40  # seconds
    OVERLAP_LENGTH = 10  # seconds
    MAX_AUDIO_LENGTH = 3600  # 1 hour max
    
    # GPU settings
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    MAX_MEMORY = "14GB"  # Adjust for RTX A4000

class AudioEnhancer:
    """Advanced audio enhancement and noise removal system - FIXED VERSION"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.setup_filters()
    
    def setup_filters(self):
        """Initialize filter parameters - FIXED frequency limits"""
        self.high_pass_cutoff = 80  # Hz
        # FIXED: Low pass cutoff must be < fs/2, not equal to fs/2
        self.low_pass_cutoff = min(7900, self.sample_rate // 2 - 100)  # Safe margin from Nyquist
        self.notch_freq = [50, 60]  # Power line noise frequencies
        
        # Validate frequencies
        nyquist = self.sample_rate / 2
        assert 0 < self.high_pass_cutoff < nyquist, f"High pass cutoff {self.high_pass_cutoff} invalid for fs={self.sample_rate}"
        assert 0 < self.low_pass_cutoff < nyquist, f"Low pass cutoff {self.low_pass_cutoff} invalid for fs={self.sample_rate}"
    
    def spectral_subtraction(self, audio: np.ndarray, alpha: float = 2.0, beta: float = 0.01) -> np.ndarray:
        """Advanced spectral subtraction for noise reduction - FIXED"""
        try:
            # Convert to frequency domain
            stft = librosa.stft(audio, n_fft=2048, hop_length=512)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Estimate noise from first 0.5 seconds
            noise_frames = max(1, int(0.5 * self.sample_rate / 512))
            noise_frames = min(noise_frames, magnitude.shape[1])  # Ensure we don't exceed available frames
            noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
            
            # Apply spectral subtraction with safety checks
            enhanced_magnitude = magnitude - alpha * noise_spectrum
            enhanced_magnitude = np.maximum(enhanced_magnitude, beta * magnitude)
            
            # Reconstruct audio
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft, hop_length=512, length=len(audio))
            
            return enhanced_audio.astype(np.float32)
            
        except Exception as e:
            print(f"Spectral subtraction failed: {e}")
            return audio
    
    def wiener_filter(self, audio: np.ndarray, noise_factor: float = 0.1) -> np.ndarray:
        """Apply Wiener filtering for noise reduction - FIXED"""
        try:
            # Compute power spectral density
            f, psd = signal.welch(audio, self.sample_rate, nperseg=min(1024, len(audio)//4))
            
            # Estimate noise PSD (assuming first 0.5s is noise)
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
            
            return filtered_audio.astype(np.float32)
            
        except Exception as e:
            print(f"Wiener filter failed: {e}")
            return audio
    
    def advanced_bandpass_filter(self, audio: np.ndarray) -> np.ndarray:
        """Apply advanced bandpass filtering - FIXED"""
        try:
            # High-pass filter to remove low-frequency noise
            sos_hp = signal.butter(4, self.high_pass_cutoff, btype='high', 
                                  fs=self.sample_rate, output='sos')
            audio = signal.sosfilt(sos_hp, audio)
            
            # Low-pass filter to remove high-frequency noise
            sos_lp = signal.butter(4, self.low_pass_cutoff, btype='low', 
                                  fs=self.sample_rate, output='sos')
            audio = signal.sosfilt(sos_lp, audio)
            
            # Notch filters for power line noise - FIXED implementation
            for freq in self.notch_freq:
                if freq < self.sample_rate / 2:  # Only apply if frequency is valid
                    try:
                        # Create notch filter and convert to SOS format
                        b, a = signal.iirnotch(freq, Q=30, fs=self.sample_rate)
                        sos_notch = signal.tf2sos(b, a)
                        audio = signal.sosfilt(sos_notch, audio)
                    except Exception as e:
                        print(f"Notch filter at {freq}Hz failed: {e}")
                        continue
            
            return audio.astype(np.float32)
            
        except Exception as e:
            print(f"Bandpass filter failed: {e}")
            return audio
    
    def adaptive_noise_reduction(self, audio: np.ndarray) -> np.ndarray:
        """Apply adaptive noise reduction using noisereduce library - FIXED"""
        try:
            # Use noisereduce for initial denoising
            reduced_noise = nr.reduce_noise(
                y=audio, 
                sr=self.sample_rate, 
                stationary=False, 
                prop_decrease=0.8,
                n_std_thresh_stationary=1.5,
                n_std_thresh_nonstationary=2.0
            )
            return reduced_noise.astype(np.float32)
        except Exception as e:
            print(f"Adaptive noise reduction failed: {e}")
            return audio
    
    def dynamic_range_compression(self, audio: np.ndarray, 
                                threshold: float = 0.3, ratio: float = 4.0) -> np.ndarray:
        """Apply dynamic range compression - FIXED"""
        try:
            # Simple compressor implementation
            compressed = np.copy(audio)
            mask = np.abs(audio) > threshold
            compressed[mask] = np.sign(audio[mask]) * (threshold + (np.abs(audio[mask]) - threshold) / ratio)
            
            return compressed.astype(np.float32)
        except Exception as e:
            print(f"Dynamic range compression failed: {e}")
            return audio
    
    def enhance_audio(self, audio: np.ndarray, enhancement_level: str = "moderate") -> np.ndarray:
        """Main enhancement pipeline - FIXED with better error handling"""
        original_audio = audio.copy()
        
        try:
            # Normalize input
            if len(audio) == 0:
                return original_audio
                
            # Step 1: Adaptive noise reduction
            print("Applying adaptive noise reduction...")
            audio = self.adaptive_noise_reduction(audio)
            
            # Step 2: Bandpass filtering
            print("Applying bandpass filtering...")
            audio = self.advanced_bandpass_filter(audio)
            
            # Step 3: Spectral subtraction
            if enhancement_level in ["moderate", "aggressive"]:
                print("Applying spectral subtraction...")
                alpha_val = 2.5 if enhancement_level == "aggressive" else 2.0
                audio = self.spectral_subtraction(audio, alpha=alpha_val, beta=0.05)
            
            # Step 4: Wiener filtering
            if enhancement_level == "aggressive":
                print("Applying Wiener filtering...")
                audio = self.wiener_filter(audio, noise_factor=0.05)
            
            # Step 5: Dynamic range compression
            print("Applying dynamic range compression...")
            audio = self.dynamic_range_compression(audio)
            
            # Step 6: Final normalization
            audio = librosa.util.normalize(audio)
            
            print("Audio enhancement completed successfully")
            return audio.astype(np.float32)
            
        except Exception as e:
            print(f"Enhancement pipeline failed: {e}")
            return original_audio.astype(np.float32)

class AudioProcessor:
    """Audio chunking and preprocessing system - FIXED VERSION"""
    
    def __init__(self, config: Config):
        self.config = config
        self.enhancer = AudioEnhancer(config.SAMPLE_RATE)
    
    def load_and_preprocess_audio(self, audio_path: str, 
                                enhancement_level: str = "moderate") -> Tuple[np.ndarray, np.ndarray, int]:
        """Load and preprocess audio file - FIXED to return both original and enhanced"""
        try:
            # Load audio
            original_audio, sr = librosa.load(audio_path, sr=self.config.SAMPLE_RATE, mono=True)
            
            # Limit audio length
            max_samples = self.config.MAX_AUDIO_LENGTH * self.config.SAMPLE_RATE
            if len(original_audio) > max_samples:
                original_audio = original_audio[:max_samples]
                print(f"Audio truncated to {self.config.MAX_AUDIO_LENGTH} seconds")
            
            # Enhance audio
            enhanced_audio = self.enhancer.enhance_audio(original_audio, enhancement_level)
            
            return original_audio, enhanced_audio, sr
            
        except Exception as e:
            raise Exception(f"Failed to load audio: {e}")
    
    def chunk_audio_with_overlap(self, audio: np.ndarray, sr: int) -> List[Tuple[np.ndarray, float, float]]:
        """Split audio into overlapping chunks - FIXED"""
        chunk_samples = int(self.config.CHUNK_LENGTH * sr)
        overlap_samples = int(self.config.OVERLAP_LENGTH * sr)
        stride = chunk_samples - overlap_samples
        
        if stride <= 0:
            stride = chunk_samples // 2  # Fallback to 50% overlap
        
        chunks = []
        for start in range(0, len(audio), stride):
            end = min(start + chunk_samples, len(audio))
            if end - start < sr:  # Skip chunks shorter than 1 second
                break
                
            chunk = audio[start:end]
            
            # Calculate timestamps
            start_time = start / sr
            end_time = end / sr
            
            chunks.append((chunk, start_time, end_time))
            
            if end >= len(audio):
                break
        
        return chunks

class Gemma3nTranscriber:
    """Gemma3n-based audio transcription system - FIXED VERSION"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.processor = None
        self.load_model()
    
    def load_model(self):
        """Load Gemma3n model and processor - FIXED"""
        try:
            print("Loading Gemma3n model...")
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.config.PROCESSOR_PATH,
                local_files_only=False,  # Allow downloading if not found locally
                trust_remote_code=True
            )
            
            # Load model with memory optimization - FIXED model class
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.MODEL_PATH,
                torch_dtype=self.config.TORCH_DTYPE,
                device_map="auto",
                max_memory={0: self.config.MAX_MEMORY} if torch.cuda.is_available() else None,
                local_files_only=False,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).eval()
            
            print(f"Model loaded successfully on {self.config.DEVICE}")
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Note: Make sure you have the correct model path and internet connection for first-time setup")
            self.model = None
            self.processor = None
    
    def transcribe_chunk(self, audio_chunk: np.ndarray, language: str = "auto") -> str:
        """Transcribe a single audio chunk - FIXED"""
        if self.model is None or self.processor is None:
            return "[MODEL_NOT_LOADED]"
            
        try:
            # Save chunk to temporary file for processing
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                sf.write(temp_file.name, audio_chunk, self.config.SAMPLE_RATE)
                temp_path = temp_file.name
            
            # Prepare messages for the model
            prompt = f"Transcribe this audio accurately. Language: {language if language != 'auto' else 'detect automatically'}. Provide only the transcription text without any additional commentary."
            
            # Process audio file
            inputs = self.processor(
                text=prompt,
                audio=temp_path,
                return_tensors="pt"
            ).to(self.model.device)
            
            # Generate transcription
            with torch.inference_mode():
                generation = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode output
            input_len = inputs["input_ids"].shape[-1]
            generated_ids = generation[0][input_len:]
            transcription = self.processor.decode(generated_ids, skip_special_tokens=True)
            
            # Cleanup
            os.unlink(temp_path)
            
            return transcription.strip()
            
        except Exception as e:
            print(f"Transcription error for chunk: {e}")
            return "[TRANSCRIPTION_ERROR]"
    
    def merge_transcriptions(self, transcriptions: List[Tuple[str, float, float]]) -> str:
        """Merge overlapping transcriptions intelligently - FIXED"""
        if not transcriptions:
            return ""
        
        merged_text = ""
        prev_words = []
        
        for i, (text, start_time, end_time) in enumerate(transcriptions):
            if text in ["[MODEL_NOT_LOADED]", "[TRANSCRIPTION_ERROR]"]:
                continue
                
            current_words = text.split()
            
            if i == 0:
                merged_text = text
                prev_words = current_words
            else:
                # Find overlap in last few words
                overlap_found = False
                max_overlap = min(10, len(prev_words), len(current_words))
                
                for overlap_len in range(max_overlap, 0, -1):
                    if (len(prev_words) >= overlap_len and 
                        len(current_words) >= overlap_len and
                        prev_words[-overlap_len:] == current_words[:overlap_len]):
                        
                        # Found overlap, merge without duplication
                        remaining_words = current_words[overlap_len:]
                        if remaining_words:
                            merged_text += " " + " ".join(remaining_words)
                        overlap_found = True
                        prev_words = current_words
                        break
                
                if not overlap_found:
                    merged_text += " " + text
                    prev_words = current_words
        
        return merged_text.strip()

class TranscriptionSystem:
    """Main transcription system orchestrator - FIXED VERSION"""
    
    def __init__(self):
        self.config = Config()
        self.audio_processor = AudioProcessor(self.config)
        self.transcriber = None
        self.initialize_transcriber()
    
    def initialize_transcriber(self):
        """Initialize the transcriber"""
        try:
            self.transcriber = Gemma3nTranscriber(self.config)
        except Exception as e:
            print(f"Failed to initialize transcriber: {e}")
            self.transcriber = None
    
    def transcribe_audio(self, audio_path: str, language: str = "auto", 
                        enhancement_level: str = "moderate") -> Tuple[str, str, str, str]:
        """Main transcription function - FIXED to return enhanced audio"""
        if not self.transcriber or not self.transcriber.model:
            return "Error: Transcriber not initialized. Please check model paths and internet connection.", "", "", ""
        
        try:
            # Load and preprocess audio
            print("Loading and enhancing audio...")
            original_audio, enhanced_audio, sr = self.audio_processor.load_and_preprocess_audio(
                audio_path, enhancement_level
            )
            
            # Save enhanced audio to temporary file for UI display
            with tempfile.NamedTemporaryFile(suffix="_enhanced.wav", delete=False) as temp_file:
                sf.write(temp_file.name, enhanced_audio, sr)
                enhanced_audio_path = temp_file.name
            
            # Save original audio to temporary file for comparison
            with tempfile.NamedTemporaryFile(suffix="_original.wav", delete=False) as temp_file:
                sf.write(temp_file.name, original_audio, sr)
                original_audio_path = temp_file.name
            
            # Chunk enhanced audio
            print("Chunking audio...")
            chunks = self.audio_processor.chunk_audio_with_overlap(enhanced_audio, sr)
            
            if not chunks:
                return "Error: No valid audio chunks created.", "", enhanced_audio_path, original_audio_path
            
            # Transcribe chunks
            transcriptions = []
            for i, (chunk, start_time, end_time) in enumerate(chunks):
                print(f"Transcribing chunk {i+1}/{len(chunks)}...")
                
                # Transcribe
                transcription = self.transcriber.transcribe_chunk(chunk, language)
                transcriptions.append((transcription, start_time, end_time))
                
                # Memory cleanup
                if i % 3 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
            
            # Merge transcriptions
            print("Merging transcriptions...")
            final_transcription = self.transcriber.merge_transcriptions(transcriptions)
            
            # Create detailed report
            duration = len(enhanced_audio) / sr
            report = f"""
Transcription Report:
===================
Audio Duration: {duration:.2f} seconds
Number of Chunks: {len(chunks)}
Enhancement Level: {enhancement_level}
Language: {language}
Model: Gemma3n-e4b-it

Enhancement Details:
- Original audio length: {len(original_audio)/sr:.2f}s
- Enhanced audio length: {len(enhanced_audio)/sr:.2f}s
- Sample rate: {sr} Hz
- Chunk size: {self.config.CHUNK_LENGTH}s
- Overlap: {self.config.OVERLAP_LENGTH}s

Chunk Details:
"""
            for i, (trans, start, end) in enumerate(transcriptions):
                if trans not in ["[MODEL_NOT_LOADED]", "[TRANSCRIPTION_ERROR]"]:
                    report += f"Chunk {i+1}: {start:.1f}s - {end:.1f}s\n"
                    report += f"Text: {trans[:100]}{'...' if len(trans) > 100 else ''}\n\n"
            
            return final_transcription, report, enhanced_audio_path, original_audio_path
            
        except Exception as e:
            return f"Error: {str(e)}", "", "", ""

# Gradio Interface - FIXED VERSION with Enhanced Audio Preview
def create_gradio_interface():
    """Create professional Gradio interface with audio comparison"""
    
    # Initialize system
    transcription_system = TranscriptionSystem()
    
    # Language options
    languages = [
        "auto", "english", "spanish", "french", "german", "italian", "portuguese",
        "russian", "chinese", "japanese", "korean", "arabic", "hindi", "dutch",
        "polish", "turkish", "swedish", "danish", "norwegian", "finnish",
        "bengali", "tamil", "urdu", "gujarati", "marathi", "telugu", "kannada"
    ]
    
    def transcribe_interface(audio_file, language, enhancement_level, manual_language):
        """Interface function for Gradio - FIXED"""
        if audio_file is None:
            return "Please upload an audio file.", "", None, None
        
        # Use manual language if provided
        selected_language = manual_language.strip() if manual_language.strip() else language
        
        try:
            transcription, report, enhanced_path, original_path = transcription_system.transcribe_audio(
                audio_file, selected_language, enhancement_level
            )
            
            # Return transcription, report, and audio files for comparison
            return transcription, report, enhanced_path, original_path
            
        except Exception as e:
            return f"Error: {str(e)}", "", None, None
    
    # Create Gradio interface
    with gr.Blocks(
        title="Professional Audio Transcription System with Enhancement Preview",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            font-family: 'Arial', sans-serif;
            max-width: 1400px;
            margin: 0 auto;
        }
        .main-header {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 30px;
        }
        .feature-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .audio-comparison {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
        """
    ) as interface:
        
        gr.HTML("""
        <div class="main-header">
            <h1>🎙️ Professional Audio Transcription System</h1>
            <p><strong>Powered by Gemma3n-e4b-it with Advanced Speech Enhancement</strong></p>
            <p><em>Now with Enhanced Audio Preview!</em></p>
        </div>
        """)
        
        gr.HTML("""
        <div class="feature-box">
            <h3>🚀 Key Features:</h3>
            <ul>
                <li>✨ Advanced noise reduction and speech enhancement</li>
                <li>🔊 Before/After audio comparison preview</li>
                <li>🌍 Multi-language support with manual language entry</li>
                <li>⚡ Intelligent audio chunking for long recordings</li>
                <li>🎯 Optimized for call recordings with noise and distortion</li>
                <li>🔧 Professional-grade audio preprocessing</li>
            </ul>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("<h3>📁 Input Configuration</h3>")
                
                audio_input = gr.Audio(
                    label="Upload Audio File",
                    type="filepath",
                    format="wav"
                )
                
                language_dropdown = gr.Dropdown(
                    choices=languages,
                    value="auto",
                    label="Select Language",
                    info="Choose the primary language of the audio"
                )
                
                manual_language = gr.Textbox(
                    label="Manual Language Entry",
                    placeholder="e.g., malayalam, punjabi, sindhi, nepali...",
                    info="Enter any language name if not in dropdown"
                )
                
                enhancement_level = gr.Radio(
                    choices=["light", "moderate", "aggressive"],
                    value="moderate",
                    label="Enhancement Level",
                    info="Higher levels provide more noise reduction"
                )
                
                transcribe_btn = gr.Button(
                    "🎯 Start Transcription",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=2):
                gr.HTML("<h3>📄 Transcription Results</h3>")
                
                transcription_output = gr.Textbox(
                    label="Transcribed Text",
                    lines=12,
                    max_lines=20,
                    placeholder="Your transcription will appear here...",
                    show_copy_button=True
                )
                
                # NEW: Audio Enhancement Comparison Section
                gr.HTML("""
                <div class="audio-comparison">
                    <h3>🔊 Audio Enhancement Comparison</h3>
                    <p>Compare the original and enhanced audio to hear the improvement:</p>
                </div>
                """)
                
                with gr.Row():
                    with gr.Column():
                        gr.HTML("<h4>📥 Original Audio</h4>")
                        original_audio_output = gr.Audio(
                            label="Original Audio (Before Enhancement)",
                            interactive=False
                        )
                    
                    with gr.Column():
                        gr.HTML("<h4>✨ Enhanced Audio</h4>")
                        enhanced_audio_output = gr.Audio(
                            label="Enhanced Audio (After Processing)",
                            interactive=False
                        )
                
                report_output = gr.Accordion(
                    label="📊 Detailed Processing Report",
                    open=False
                )
                
                with report_output:
                    report_text = gr.Textbox(
                        label="Processing Details",
                        lines=12,
                        show_copy_button=True
                    )
        
        # Event handling - FIXED
        transcribe_btn.click(
            fn=transcribe_interface,
            inputs=[audio_input, language_dropdown, enhancement_level, manual_language],
            outputs=[transcription_output, report_text, enhanced_audio_output, original_audio_output]
        )
        
        # Examples and tips section
        gr.HTML("""
        <div style="margin-top: 30px; padding: 20px; background-color: #f8f9fa; border-radius: 10px;">
            <h3>💡 Usage Tips & Fixes Applied:</h3>
            <ul>
                <li><strong>✅ FIXED:</strong> Filter frequency error - Low-pass cutoff now correctly set below Nyquist frequency</li>
                <li><strong>✅ FIXED:</strong> Notch filter implementation using proper SOS format</li>
                <li><strong>✅ NEW:</strong> Enhanced audio preview to compare before/after processing</li>
                <li><strong>✅ FIXED:</strong> Model loading with proper AutoModelForCausalLM class</li>
                <li><strong>File Formats:</strong> Supports WAV, MP3, FLAC, M4A, and other common formats</li>
                <li><strong>File Size:</strong> Optimized for files up to 1 hour in length</li>
                <li><strong>Enhancement Levels:</strong>
                    <ul>
                        <li><em>Light:</em> Basic noise reduction, preserves original audio character</li>
                        <li><em>Moderate:</em> Balanced enhancement, good for most call recordings</li>
                        <li><em>Aggressive:</em> Maximum noise reduction, best for very noisy audio</li>
                    </ul>
                </li>
                <li><strong>Languages:</strong> Supports 100+ languages. Use manual entry for specialized languages.</li>
                <li><strong>Audio Comparison:</strong> Listen to both original and enhanced audio to verify improvement quality</li>
            </ul>
        </div>
        """)
    
    return interface

# Main execution - FIXED
if __name__ == "__main__":
    # Check configuration
    print("🎙️ Professional Audio Transcription System - FIXED VERSION")
    print("=" * 60)
    print(f"Device: {Config.DEVICE}")
    print(f"Model Path: {Config.MODEL_PATH}")
    print(f"Sample Rate: {Config.SAMPLE_RATE}")
    print("✅ All critical fixes applied:")
    print("  - Filter frequency limits corrected")
    print("  - Notch filter implementation fixed")
    print("  - Enhanced audio preview added")
    print("  - Model loading class updated")
    print("  - Error handling improved")
    print("=" * 60)
    
    # Verify dependencies
    try:
        import librosa
        import noisereduce
        import scipy
        import transformers
        print("✅ All dependencies verified")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install missing packages and try again")
        exit(1)
    
    # Launch interface
    try:
        interface = create_gradio_interface()
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=True,
            show_error=True
        )
    except Exception as e:
        print(f"Failed to launch interface: {e}")
