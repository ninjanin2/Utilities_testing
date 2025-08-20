"""
Professional Audio Transcription System with Gemma3n-e4b-it
Author: Advanced AI Audio Processing System
Features: Speech Enhancement, Noise Removal, Long Audio Chunking, Multi-language Support
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
warnings.filterwarnings("ignore")

# Audio processing libraries
from scipy import signal
from scipy.fft import fft, ifft
import noisereduce as nr

# Transformers for Gemma3n
from transformers import (
    AutoProcessor, 
    Gemma3nForConditionalGeneration,
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
    """Advanced audio enhancement and noise removal system"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.setup_filters()
    
    def setup_filters(self):
        """Initialize filter parameters"""
        self.high_pass_cutoff = 80  # Hz
        self.low_pass_cutoff = 8000  # Hz
        self.notch_freq = [50, 60]  # Power line noise frequencies
    
    def spectral_subtraction(self, audio: np.ndarray, alpha: float = 2.0, beta: float = 0.01) -> np.ndarray:
        """Advanced spectral subtraction for noise reduction"""
        # Convert to frequency domain
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimate noise from first 0.5 seconds
        noise_frames = int(0.5 * self.sample_rate / 512)
        noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
        
        # Apply spectral subtraction
        enhanced_magnitude = magnitude - alpha * noise_spectrum
        enhanced_magnitude = np.maximum(enhanced_magnitude, beta * magnitude)
        
        # Reconstruct audio
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        enhanced_audio = librosa.istft(enhanced_stft, hop_length=512)
        
        return enhanced_audio
    
    def wiener_filter(self, audio: np.ndarray, noise_factor: float = 0.1) -> np.ndarray:
        """Apply Wiener filtering for noise reduction"""
        # Compute power spectral density
        f, psd = signal.welch(audio, self.sample_rate, nperseg=1024)
        
        # Estimate noise PSD (assuming first 0.5s is noise)
        noise_samples = int(0.5 * self.sample_rate)
        noise_psd = np.mean(np.abs(fft(audio[:noise_samples]))**2)
        
        # Apply Wiener filter in frequency domain
        audio_fft = fft(audio)
        wiener_filter = np.abs(audio_fft)**2 / (np.abs(audio_fft)**2 + noise_factor * noise_psd)
        filtered_fft = audio_fft * wiener_filter
        filtered_audio = np.real(ifft(filtered_fft))
        
        return filtered_audio.astype(np.float32)
    
    def advanced_bandpass_filter(self, audio: np.ndarray) -> np.ndarray:
        """Apply advanced bandpass filtering"""
        # High-pass filter to remove low-frequency noise
        sos_hp = signal.butter(4, self.high_pass_cutoff, btype='high', 
                              fs=self.sample_rate, output='sos')
        audio = signal.sosfilt(sos_hp, audio)
        
        # Low-pass filter to remove high-frequency noise
        sos_lp = signal.butter(4, self.low_pass_cutoff, btype='low', 
                              fs=self.sample_rate, output='sos')
        audio = signal.sosfilt(sos_lp, audio)
        
        # Notch filters for power line noise
        for freq in self.notch_freq:
            sos_notch = signal.iirnotch(freq, Q=30, fs=self.sample_rate)
            audio = signal.sosfilt(sos_notch, audio)
        
        return audio
    
    def adaptive_noise_reduction(self, audio: np.ndarray) -> np.ndarray:
        """Apply adaptive noise reduction using noisereduce library"""
        try:
            # Use noisereduce for initial denoising
            reduced_noise = nr.reduce_noise(y=audio, sr=self.sample_rate, 
                                          stationary=False, prop_decrease=0.8)
            return reduced_noise
        except Exception as e:
            print(f"Adaptive noise reduction failed: {e}")
            return audio
    
    def dynamic_range_compression(self, audio: np.ndarray, 
                                threshold: float = 0.3, ratio: float = 4.0) -> np.ndarray:
        """Apply dynamic range compression"""
        # Simple compressor implementation
        compressed = np.copy(audio)
        mask = np.abs(audio) > threshold
        compressed[mask] = threshold + (audio[mask] - threshold) / ratio
        compressed[~mask] = audio[~mask]
        
        return compressed
    
    def enhance_audio(self, audio: np.ndarray, enhancement_level: str = "aggressive") -> np.ndarray:
        """Main enhancement pipeline"""
        original_audio = audio.copy()
        
        try:
            # Step 1: Adaptive noise reduction
            audio = self.adaptive_noise_reduction(audio)
            
            # Step 2: Bandpass filtering
            audio = self.advanced_bandpass_filter(audio)
            
            # Step 3: Spectral subtraction
            if enhancement_level in ["moderate", "aggressive"]:
                audio = self.spectral_subtraction(audio, alpha=2.5, beta=0.05)
            
            # Step 4: Wiener filtering
            if enhancement_level == "aggressive":
                audio = self.wiener_filter(audio, noise_factor=0.05)
            
            # Step 5: Dynamic range compression
            audio = self.dynamic_range_compression(audio)
            
            # Step 6: Normalize
            audio = librosa.util.normalize(audio)
            
            return audio.astype(np.float32)
            
        except Exception as e:
            print(f"Enhancement failed: {e}")
            return original_audio

class AudioProcessor:
    """Audio chunking and preprocessing system"""
    
    def __init__(self, config: Config):
        self.config = config
        self.enhancer = AudioEnhancer(config.SAMPLE_RATE)
    
    def load_and_preprocess_audio(self, audio_path: str, 
                                enhancement_level: str = "moderate") -> Tuple[np.ndarray, int]:
        """Load and preprocess audio file"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.config.SAMPLE_RATE, mono=True)
            
            # Limit audio length
            max_samples = self.config.MAX_AUDIO_LENGTH * self.config.SAMPLE_RATE
            if len(audio) > max_samples:
                audio = audio[:max_samples]
                print(f"Audio truncated to {self.config.MAX_AUDIO_LENGTH} seconds")
            
            # Enhance audio
            audio = self.enhancer.enhance_audio(audio, enhancement_level)
            
            return audio, sr
            
        except Exception as e:
            raise Exception(f"Failed to load audio: {e}")
    
    def chunk_audio_with_overlap(self, audio: np.ndarray, sr: int) -> List[Tuple[np.ndarray, float, float]]:
        """Split audio into overlapping chunks"""
        chunk_samples = int(self.config.CHUNK_LENGTH * sr)
        overlap_samples = int(self.config.OVERLAP_LENGTH * sr)
        stride = chunk_samples - overlap_samples
        
        chunks = []
        for start in range(0, len(audio) - overlap_samples, stride):
            end = min(start + chunk_samples, len(audio))
            chunk = audio[start:end]
            
            # Calculate timestamps
            start_time = start / sr
            end_time = end / sr
            
            chunks.append((chunk, start_time, end_time))
            
            if end >= len(audio):
                break
        
        return chunks

class Gemma3nTranscriber:
    """Gemma3n-based audio transcription system"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.processor = None
        self.load_model()
    
    def load_model(self):
        """Load Gemma3n model and processor"""
        try:
            print("Loading Gemma3n model...")
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.config.PROCESSOR_PATH,
                local_files_only=True
            )
            
            # Load model with memory optimization
            self.model = Gemma3nForConditionalGeneration.from_pretrained(
                self.config.MODEL_PATH,
                torch_dtype=self.config.TORCH_DTYPE,
                device_map="auto",
                max_memory={0: self.config.MAX_MEMORY},
                local_files_only=True,
                low_cpu_mem_usage=True
            ).eval()
            
            print(f"Model loaded successfully on {self.config.DEVICE}")
            
        except Exception as e:
            raise Exception(f"Failed to load model: {e}")
    
    def transcribe_chunk(self, audio_chunk: np.ndarray, language: str = "auto") -> str:
        """Transcribe a single audio chunk"""
        try:
            # Prepare messages for the model
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio", 
                            "audio": audio_chunk.tolist()
                        },
                        {
                            "type": "text", 
                            "text": f"Transcribe this audio accurately. Language: {language if language != 'auto' else 'detect automatically'}"
                        }
                    ]
                }
            ]
            
            # Process inputs
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.model.device)
            
            # Generate transcription
            with torch.inference_mode():
                generation = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode output
            input_len = inputs["input_ids"].shape[-1]
            generated_ids = generation[0][input_len:]
            transcription = self.processor.decode(generated_ids, skip_special_tokens=True)
            
            return transcription.strip()
            
        except Exception as e:
            print(f"Transcription error for chunk: {e}")
            return "[TRANSCRIPTION_ERROR]"
    
    def merge_transcriptions(self, transcriptions: List[Tuple[str, float, float]]) -> str:
        """Merge overlapping transcriptions intelligently"""
        if not transcriptions:
            return ""
        
        merged_text = ""
        prev_text = ""
        
        for i, (text, start_time, end_time) in enumerate(transcriptions):
            if i == 0:
                merged_text = text
                prev_text = text
            else:
                # Simple overlap handling - look for common phrases
                words = text.split()
                prev_words = prev_text.split()
                
                # Find overlap
                overlap_found = False
                for j in range(min(10, len(prev_words))):  # Check last 10 words
                    for k in range(min(10, len(words))):   # Check first 10 words
                        if prev_words[-(j+1):] == words[:k+1] and j > 0:
                            # Found overlap, merge without duplication
                            merged_text += " " + " ".join(words[k+1:])
                            overlap_found = True
                            break
                    if overlap_found:
                        break
                
                if not overlap_found:
                    merged_text += " " + text
                
                prev_text = text
        
        return merged_text.strip()

class TranscriptionSystem:
    """Main transcription system orchestrator"""
    
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
                        enhancement_level: str = "moderate") -> Tuple[str, str]:
        """Main transcription function"""
        if not self.transcriber:
            return "Error: Transcriber not initialized. Please check model paths.", ""
        
        try:
            # Load and preprocess audio
            status = "Loading and enhancing audio..."
            audio, sr = self.audio_processor.load_and_preprocess_audio(
                audio_path, enhancement_level
            )
            
            # Chunk audio
            status = "Chunking audio..."
            chunks = self.audio_processor.chunk_audio_with_overlap(audio, sr)
            
            # Transcribe chunks
            transcriptions = []
            for i, (chunk, start_time, end_time) in enumerate(chunks):
                status = f"Transcribing chunk {i+1}/{len(chunks)}..."
                
                # Save chunk temporarily for model input
                chunk_path = f"/tmp/chunk_{i}.wav"
                sf.write(chunk_path, chunk, sr)
                
                # Transcribe
                transcription = self.transcriber.transcribe_chunk(chunk, language)
                transcriptions.append((transcription, start_time, end_time))
                
                # Cleanup
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
                
                # Memory cleanup
                if i % 5 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
            
            # Merge transcriptions
            status = "Merging transcriptions..."
            final_transcription = self.transcriber.merge_transcriptions(transcriptions)
            
            # Create detailed report
            report = f"""
Transcription Report:
===================
Audio Duration: {len(audio)/sr:.2f} seconds
Number of Chunks: {len(chunks)}
Enhancement Level: {enhancement_level}
Language: {language}
Model: Gemma3n-e4b-it

Chunk Details:
"""
            for i, (trans, start, end) in enumerate(transcriptions):
                report += f"Chunk {i+1}: {start:.1f}s - {end:.1f}s\n"
                report += f"Text: {trans[:100]}{'...' if len(trans) > 100 else ''}\n\n"
            
            return final_transcription, report
            
        except Exception as e:
            return f"Error: {str(e)}", ""

# Gradio Interface
def create_gradio_interface():
    """Create professional Gradio interface"""
    
    # Initialize system
    transcription_system = TranscriptionSystem()
    
    # Language options
    languages = [
        "auto", "english", "spanish", "french", "german", "italian", "portuguese",
        "russian", "chinese", "japanese", "korean", "arabic", "hindi", "dutch",
        "polish", "turkish", "swedish", "danish", "norwegian", "finnish"
    ]
    
    def transcribe_interface(audio_file, language, enhancement_level, manual_language):
        """Interface function for Gradio"""
        if audio_file is None:
            return "Please upload an audio file.", ""
        
        # Use manual language if provided
        selected_language = manual_language.strip() if manual_language.strip() else language
        
        try:
            transcription, report = transcription_system.transcribe_audio(
                audio_file, selected_language, enhancement_level
            )
            return transcription, report
        except Exception as e:
            return f"Error: {str(e)}", ""
    
    # Create Gradio interface
    with gr.Blocks(
        title="Professional Audio Transcription System",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            font-family: 'Arial', sans-serif;
            max-width: 1200px;
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
        """
    ) as interface:
        
        gr.HTML("""
        <div class="main-header">
            <h1>🎙️ Professional Audio Transcription System</h1>
            <p><strong>Powered by Gemma3n-e4b-it with Advanced Speech Enhancement</strong></p>
        </div>
        """)
        
        gr.HTML("""
        <div class="feature-box">
            <h3>🚀 Key Features:</h3>
            <ul>
                <li>✨ Advanced noise reduction and speech enhancement</li>
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
                    placeholder="e.g., bengali, tamil, urdu, swahili...",
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
                    lines=15,
                    max_lines=25,
                    placeholder="Your transcription will appear here...",
                    show_copy_button=True
                )
                
                report_output = gr.Accordion(
                    label="📊 Detailed Processing Report",
                    open=False
                )
                
                with report_output:
                    report_text = gr.Textbox(
                        label="Processing Details",
                        lines=10,
                        show_copy_button=True
                    )
        
        # Event handling
        transcribe_btn.click(
            fn=transcribe_interface,
            inputs=[audio_input, language_dropdown, enhancement_level, manual_language],
            outputs=[transcription_output, report_text]
        )
        
        # Examples section
        gr.HTML("""
        <div style="margin-top: 30px; padding: 20px; background-color: #f8f9fa; border-radius: 10px;">
            <h3>💡 Usage Tips:</h3>
            <ul>
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
            </ul>
        </div>
        """)
    
    return interface

# Main execution
if __name__ == "__main__":
    # Check configuration
    print("🎙️ Professional Audio Transcription System")
    print("=" * 50)
    print(f"Device: {Config.DEVICE}")
    print(f"Model Path: {Config.MODEL_PATH}")
    print(f"Sample Rate: {Config.SAMPLE_RATE}")
    print("=" * 50)
    
    # Verify model paths
    if not os.path.exists(Config.MODEL_PATH):
        print("⚠️ WARNING: Model path not found!")
        print("Please update Config.MODEL_PATH to point to your local Gemma3n-e4b-it model")
        print("Download from: https://huggingface.co/google/gemma-3n-e4b-it")
    
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
