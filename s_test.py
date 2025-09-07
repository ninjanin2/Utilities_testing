import os
import warnings
import torch
import torchaudio
import numpy as np
import gradio as gr
from pathlib import Path
import tempfile
from typing import Tuple, Optional
import logging

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

# Global variable for model directory path
MODEL_DIR = "/path/to/your/sgmse/model"  # Change this to your local model directory

class SGMSEEnhancer:
    """
    SGMSE Speech Enhancement Model Wrapper
    Based on the sp-uhh/speech-enhancement-sgmse implementation
    """
    
    def __init__(self, model_dir: str):
        """
        Initialize the SGMSE model from local directory
        
        Args:
            model_dir: Path to local SGMSE model directory
        """
        self.model_dir = Path(model_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.target_sr = 16000  # Standard sampling rate for ASR
        self.n_fft = 1024
        self.hop_length = 256
        self.win_length = 1024
        
        # Model parameters based on SGMSE+ configuration
        self.N = 30  # Number of diffusion steps
        self.corrector_steps = 1
        self.snr = 0.5
        self.predictor = 'euler_maruyama'
        self.corrector = 'ald'
        
        logging.info(f"Using device: {self.device}")
        self.load_model()
    
    def load_model(self):
        """Load the SGMSE model from local directory"""
        try:
            # Add the model directory to Python path
            import sys
            sys.path.append(str(self.model_dir))
            
            # Import SGMSE modules
            from sgmse.model import ScoreModel
            from sgmse import sampling
            
            # Find checkpoint file
            checkpoint_files = list(self.model_dir.glob("*.ckpt"))
            if not checkpoint_files:
                raise FileNotFoundError("No checkpoint files found in model directory")
            
            checkpoint_path = checkpoint_files[0]  # Use first checkpoint found
            logging.info(f"Loading checkpoint from: {checkpoint_path}")
            
            # Load the model
            self.model = ScoreModel.load_from_checkpoint(
                checkpoint_path, 
                map_location=self.device
            )
            self.model.eval()
            self.model.to(self.device)
            
            logging.info("SGMSE model loaded successfully")
            
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise
    
    def preprocess_audio(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        """
        Preprocess audio for SGMSE model
        
        Args:
            audio: Input audio tensor
            sr: Original sampling rate
            
        Returns:
            Preprocessed audio tensor
        """
        # Resample to target sampling rate if needed
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            audio = resampler(audio)
        
        # Convert to mono if stereo
        if audio.dim() > 1 and audio.size(0) > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        # Ensure proper shape
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            
        return audio
    
    def audio_to_spec(self, audio: torch.Tensor) -> torch.Tensor:
        """Convert audio to complex spectrogram"""
        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length).to(audio.device),
            return_complex=True
        )
        return spec
    
    def spec_to_audio(self, spec: torch.Tensor) -> torch.Tensor:
        """Convert complex spectrogram back to audio"""
        audio = torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length).to(spec.device)
        )
        return audio
    
    def enhance_chunk(self, noisy_spec: torch.Tensor) -> torch.Tensor:
        """
        Enhance a single chunk of spectrogram
        
        Args:
            noisy_spec: Noisy complex spectrogram
            
        Returns:
            Enhanced complex spectrogram
        """
        with torch.no_grad():
            # Convert to model input format
            noisy_spec = noisy_spec.to(self.device)
            
            # Add batch dimension if needed
            if noisy_spec.dim() == 2:
                noisy_spec = noisy_spec.unsqueeze(0)
            
            # Get sampler
            sampler = self.model.get_pc_sampler(
                self.predictor, 
                self.corrector,
                y=noisy_spec,
                N=self.N,
                corrector_steps=self.corrector_steps,
                snr=self.snr
            )
            
            # Sample enhanced spectrogram
            enhanced_spec, _ = sampler()
            
            return enhanced_spec.squeeze(0)
    
    def enhance_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """
        Enhance speech from audio file
        
        Args:
            audio_path: Path to input audio file
            
        Returns:
            Tuple of (enhanced_audio, sampling_rate)
        """
        try:
            # Load audio
            audio, sr = torchaudio.load(audio_path)
            logging.info(f"Loaded audio: shape={audio.shape}, sr={sr}")
            
            # Preprocess
            audio = self.preprocess_audio(audio, sr)
            audio = audio.to(self.device)
            
            # Convert to spectrogram
            noisy_spec = self.audio_to_spec(audio)
            
            # Handle long audio by chunking if necessary
            max_length = 512  # Max spectrogram length for memory efficiency
            
            if noisy_spec.size(-1) > max_length:
                # Process in chunks
                enhanced_specs = []
                for i in range(0, noisy_spec.size(-1), max_length):
                    chunk = noisy_spec[..., i:i+max_length]
                    enhanced_chunk = self.enhance_chunk(chunk)
                    enhanced_specs.append(enhanced_chunk)
                
                enhanced_spec = torch.cat(enhanced_specs, dim=-1)
            else:
                enhanced_spec = self.enhance_chunk(noisy_spec)
            
            # Convert back to audio
            enhanced_audio = self.spec_to_audio(enhanced_spec)
            
            # Post-process for ASR readiness
            enhanced_audio = self.post_process_for_asr(enhanced_audio)
            
            logging.info(f"Enhanced audio: shape={enhanced_audio.shape}")
            return enhanced_audio.cpu(), self.target_sr
            
        except Exception as e:
            logging.error(f"Error enhancing audio: {e}")
            raise
    
    def post_process_for_asr(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Post-process enhanced audio to be ASR ready
        
        Args:
            audio: Enhanced audio tensor
            
        Returns:
            ASR-ready audio tensor
        """
        # Normalize audio
        audio = audio / (torch.max(torch.abs(audio)) + 1e-8)
        
        # Apply gentle compression to reduce dynamic range
        audio = torch.sign(audio) * torch.pow(torch.abs(audio), 0.8)
        
        # Final normalization
        audio = audio * 0.9  # Leave some headroom
        
        return audio

# Global model instance
enhancer = None

def initialize_model():
    """Initialize the SGMSE model"""
    global enhancer
    try:
        enhancer = SGMSEEnhancer(MODEL_DIR)
        return "Model initialized successfully!"
    except Exception as e:
        return f"Error initializing model: {str(e)}"

def enhance_speech(audio_file) -> Tuple[int, np.ndarray]:
    """
    Gradio interface function for speech enhancement
    
    Args:
        audio_file: Input audio file from Gradio
        
    Returns:
        Tuple of (sampling_rate, enhanced_audio_array)
    """
    global enhancer
    
    if enhancer is None:
        raise gr.Error("Model not initialized. Please check MODEL_DIR path.")
    
    if audio_file is None:
        raise gr.Error("Please upload an audio file.")
    
    try:
        # Enhance the audio
        enhanced_audio, sr = enhancer.enhance_audio(audio_file)
        
        # Convert to numpy for Gradio
        enhanced_audio_np = enhanced_audio.numpy()
        
        # Ensure proper shape for Gradio (flatten if needed)
        if enhanced_audio_np.ndim > 1:
            enhanced_audio_np = enhanced_audio_np.flatten()
            
        return sr, enhanced_audio_np
        
    except Exception as e:
        raise gr.Error(f"Enhancement failed: {str(e)}")

def create_gradio_interface():
    """Create and configure the Gradio interface"""
    
    with gr.Blocks(
        title="SGMSE Speech Enhancement", 
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 800px;
            margin: auto;
        }
        """
    ) as interface:
        
        gr.Markdown(
            """
            # SGMSE Speech Enhancement System
            
            This system uses the SGMSE+ diffusion-based model for speech enhancement.
            Upload a noisy audio file to get an ASR-ready enhanced version.
            
            **Features:**
            - Handles audio of any length
            - Optimized for ASR systems
            - Removes noise and reverberation
            - Preserves speech quality
            """
        )
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Input")
                audio_input = gr.Audio(
                    label="Upload Noisy Audio",
                    type="filepath",
                    sources=["upload", "microphone"]
                )
                
                enhance_btn = gr.Button(
                    "Enhance Speech", 
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column():
                gr.Markdown("### Output")
                audio_output = gr.Audio(
                    label="Enhanced Audio (ASR Ready)",
                    type="numpy"
                )
        
        # Model status
        with gr.Row():
            status_text = gr.Textbox(
                label="Model Status",
                value="Click 'Initialize Model' to start",
                interactive=False
            )
            init_btn = gr.Button("Initialize Model")
        
        # Event handlers
        init_btn.click(
            initialize_model,
            outputs=status_text
        )
        
        enhance_btn.click(
            enhance_speech,
            inputs=audio_input,
            outputs=audio_output
        )
        
        # Examples section
        gr.Markdown(
            """
            ### Usage Instructions
            
            1. **Initialize Model**: Click "Initialize Model" first to load the SGMSE model
            2. **Upload Audio**: Upload a noisy speech file or record using microphone
            3. **Enhance**: Click "Enhance Speech" to process the audio
            4. **Download**: Use the download button to save the enhanced audio
            
            **Supported Formats**: WAV, MP3, FLAC, M4A
            **Output**: 16kHz WAV optimized for ASR systems
            """
        )
    
    return interface

def main():
    """Main function to run the Gradio interface"""
    
    # Verify model directory exists
    if not os.path.exists(MODEL_DIR):
        print(f"ERROR: Model directory not found: {MODEL_DIR}")
        print("Please update the MODEL_DIR variable with the correct path to your SGMSE model.")
        return
    
    print("Starting SGMSE Speech Enhancement Interface...")
    print(f"Model directory: {MODEL_DIR}")
    print(f"Using device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Create and launch interface
    interface = create_gradio_interface()
    
    interface.launch(
        server_name="127.0.0.1",  # Local only for offline use
        server_port=7860,
        share=False,  # No sharing for offline use
        show_error=True,
        inbrowser=True
    )

if __name__ == "__main__":
    main()
