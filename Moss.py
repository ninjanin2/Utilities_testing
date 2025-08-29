import os
import torch
import torchaudio
import numpy as np
import gradio as gr
from torch import nn
import torch.nn.functional as F
from typing import Tuple, Optional
import warnings
import tempfile
warnings.filterwarnings("ignore")

# Global model path - Update this to your local model directory
MODEL_PATH = "path/to/your/mossformer2_model/checkpoint.pt"

class MossFormer2Block(nn.Module):
    """MossFormer2 transformer block implementation"""
    def __init__(self, d_model=256, nhead=8, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

class MossFormer2SE(nn.Module):
    """MossFormer2 Speech Enhancement Model"""
    def __init__(self, 
                 n_fft=512, 
                 hop_length=256, 
                 win_length=512,
                 d_model=256,
                 nhead=8,
                 num_layers=6,
                 dim_feedforward=1024,
                 dropout=0.1):
        super().__init__()
        
        # STFT parameters
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        
        # Feature dimensions
        self.freq_bins = n_fft // 2 + 1
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(self.freq_bins * 2, d_model)  # *2 for real and imag
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            MossFormer2Block(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(d_model, self.freq_bins * 2)
        
        # Magnitude and phase estimation
        self.magnitude_head = nn.Sequential(
            nn.Linear(self.freq_bins * 2, self.freq_bins),
            nn.Sigmoid()
        )
        
        self.phase_head = nn.Sequential(
            nn.Linear(self.freq_bins * 2, self.freq_bins),
            nn.Tanh()
        )

    def forward(self, waveform):
        """
        Args:
            waveform: Input waveform tensor of shape (batch_size, time)
        Returns:
            enhanced_waveform: Enhanced waveform tensor of shape (batch_size, time)
        """
        batch_size, time_len = waveform.shape
        
        # Create window tensor (FIXED: Explicit window specification required in PyTorch 2.1+)
        window = torch.hann_window(self.win_length, device=waveform.device, dtype=waveform.dtype)
        
        # STFT (FIXED: return_complex=True is now required)
        stft = torch.stft(waveform, 
                         n_fft=self.n_fft,
                         hop_length=self.hop_length,
                         win_length=self.win_length,
                         window=window,
                         return_complex=True,  # Required parameter
                         center=True,
                         pad_mode='reflect',
                         normalized=False,
                         onesided=True)
        
        # stft shape: (batch_size, freq_bins, time_frames)
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)
        
        # Prepare input features
        real_part = stft.real
        imag_part = stft.imag
        features = torch.cat([real_part, imag_part], dim=1)  # (batch_size, freq_bins*2, time_frames)
        
        # Transpose for transformer: (batch_size, time_frames, freq_bins*2)
        features = features.transpose(1, 2)
        
        # Project to model dimension
        x = self.input_proj(features)  # (batch_size, time_frames, d_model)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Project back to frequency domain
        x = self.output_proj(x)  # (batch_size, time_frames, freq_bins*2)
        
        # Estimate enhanced magnitude and phase
        enhanced_magnitude = self.magnitude_head(x) * magnitude.transpose(1, 2)
        phase_residual = self.phase_head(x) * np.pi
        enhanced_phase = phase.transpose(1, 2) + phase_residual
        
        # Reconstruct complex spectrogram
        enhanced_real = enhanced_magnitude * torch.cos(enhanced_phase)
        enhanced_imag = enhanced_magnitude * torch.sin(enhanced_phase)
        enhanced_stft = torch.complex(enhanced_real, enhanced_imag)
        
        # Transpose back: (batch_size, freq_bins, time_frames)
        enhanced_stft = enhanced_stft.transpose(1, 2)
        
        # ISTFT (FIXED: Explicit window specification)
        enhanced_waveform = torch.istft(enhanced_stft,
                                      n_fft=self.n_fft,
                                      hop_length=self.hop_length,
                                      win_length=self.win_length,
                                      window=window,
                                      center=True,
                                      normalized=False,
                                      onesided=True,
                                      length=time_len)
        
        return enhanced_waveform

class AudioEnhancer:
    """Audio Enhancement Pipeline"""
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        self.model = None
        self.sample_rate = 48000  # MossFormer2_SE_48K uses 48kHz
        self.max_audio_length = 16 * self.sample_rate  # Limit to 16 seconds to prevent memory issues
        self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load the MossFormer2 model from checkpoint"""
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
            
            # Load checkpoint with proper error handling
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            
            # Initialize model
            self.model = MossFormer2SE()
            
            # Load state dict with error handling
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Model loaded successfully from {model_path}")
            print(f"🔧 Using device: {self.device}")
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            raise e
    
    def preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess audio file"""
        try:
            # Load audio (FIXED: Handle upcoming torchaudio.load changes)
            waveform, sr = torchaudio.load(audio_path, normalize=True, channels_first=True)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Limit audio length to prevent memory issues
            if waveform.shape[1] > self.max_audio_length:
                print(f"⚠️ Audio too long ({waveform.shape[1]/self.sample_rate:.1f}s), truncating to 16s")
                waveform = waveform[:, :self.max_audio_length]
            
            # Resample to target sample rate if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
                waveform = resampler(waveform)
            
            # Normalize to prevent clipping
            max_val = torch.max(torch.abs(waveform))
            if max_val > 0:
                waveform = waveform / max_val
            
            return waveform.to(self.device)
            
        except Exception as e:
            print(f"❌ Error preprocessing audio: {str(e)}")
            raise e
    
    def enhance_audio(self, input_audio_path: str, output_audio_path: str) -> str:
        """Enhance audio file"""
        try:
            if self.model is None:
                raise RuntimeError("Model not loaded")
            
            # Preprocess audio
            waveform = self.preprocess_audio(input_audio_path)
            
            # Add batch dimension
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            elif waveform.dim() == 2 and waveform.shape[0] == 1:
                pass  # Already has batch dimension
            else:
                waveform = waveform.squeeze()
                waveform = waveform.unsqueeze(0)
            
            # Enhance audio with memory optimization
            with torch.no_grad():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # Clear GPU cache
                enhanced_waveform = self.model(waveform)
            
            # Move to CPU and convert to proper format for saving
            enhanced_waveform = enhanced_waveform.cpu().float()
            
            # Ensure proper shape for saving
            if enhanced_waveform.dim() == 1:
                enhanced_waveform = enhanced_waveform.unsqueeze(0)
            
            # Save enhanced audio (FIXED: Proper backend handling)
            torchaudio.save(output_audio_path, enhanced_waveform, self.sample_rate, encoding="PCM_S", bits_per_sample=16)
            
            print(f"✅ Audio enhanced and saved to: {output_audio_path}")
            return output_audio_path
            
        except Exception as e:
            print(f"❌ Error enhancing audio: {str(e)}")
            raise e

# Global enhancer instance
enhancer = None

def initialize_enhancer():
    """Initialize the audio enhancer"""
    global enhancer
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        enhancer = AudioEnhancer(MODEL_PATH, device)
        return f"✅ Model loaded successfully on {device}!"
    except Exception as e:
        return f"❌ Error loading model: {str(e)}"

def process_audio(input_audio):
    """Process audio through Gradio interface"""
    global enhancer
    
    if enhancer is None:
        return None, "❌ Model not loaded. Please check the model path."
    
    if input_audio is None:
        return None, "❌ Please upload an audio file."
    
    try:
        # Create temporary output file with proper extension
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            output_path = tmp_file.name
        
        # Enhance audio
        enhanced_path = enhancer.enhance_audio(input_audio, output_path)
        
        return enhanced_path, "✅ Audio enhanced successfully!"
        
    except Exception as e:
        return None, f"❌ Error processing audio: {str(e)}"

def create_gradio_interface():
    """Create Gradio interface (UPDATED for Gradio 5)"""
    
    # Custom CSS for Gradio 5 compatibility
    css = """
    .gradio-container {
        max-width: 900px !important;
        margin: auto !important;
    }
    .main-content {
        padding: 2rem;
    }
    """
    
    # FIXED: Updated for Gradio 5 syntax
    with gr.Blocks(css=css, title="MossFormer2 Audio Enhancement", theme=gr.themes.Soft()) as interface:
        
        gr.Markdown("""
        # 🎵 MossFormer2 Audio Speech Enhancement
        
        Upload an audio file to enhance its quality using the MossFormer2_SE_48K model.
        The model will remove noise and improve speech clarity.
        
        **Supported formats:** WAV, MP3, FLAC, OGG
        **Max duration:** 16 seconds (longer files will be truncated)
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Model status
                status_text = gr.Textbox(
                    label="🔧 Model Status",
                    value=initialize_enhancer(),
                    interactive=False,
                    container=True
                )
                
                # Audio input (FIXED: Updated for Gradio 5)
                audio_input = gr.Audio(
                    label="📤 Upload Audio File",
                    type="filepath",
                    sources=["upload"],
                    format="wav"
                )
                
                # Process button (FIXED: Updated styling for Gradio 5)
                process_btn = gr.Button(
                    "🚀 Enhance Audio",
                    variant="primary",
                    size="lg",
                    scale=1
                )
            
            with gr.Column(scale=1):
                # Audio output (FIXED: Updated for Gradio 5)
                audio_output = gr.Audio(
                    label="📥 Enhanced Audio",
                    type="filepath",
                    interactive=False
                )
                
                # Status message
                message_output = gr.Textbox(
                    label="📊 Processing Status",
                    interactive=False,
                    container=True
                )
        
        # Connect components (FIXED: Proper event handling for Gradio 5)
        process_btn.click(
            fn=process_audio,
            inputs=[audio_input],
            outputs=[audio_output, message_output],
            show_progress="full"
        )
        
        with gr.Accordion("📖 Instructions & Technical Details", open=False):
            gr.Markdown("""
            ## 📝 How to Use:
            1. **Model Setup:** Ensure the model checkpoint is available at the specified path
            2. **Upload Audio:** Click "Upload Audio File" and select your audio file
            3. **Process:** Click "Enhance Audio" to start processing
            4. **Download:** Once complete, play or download the enhanced audio
            
            ## ⚙️ Technical Specifications:
            - **Model Architecture:** MossFormer2 Transformer-based Speech Enhancement
            - **Sample Rate:** 48 kHz
            - **Processing:** Real-time STFT-based frequency domain enhancement
            - **Input:** Noisy/distorted speech audio
            - **Output:** Clean, enhanced speech with noise reduction
            
            ## ⚠️ Limitations:
            - Maximum audio length: 16 seconds (to prevent memory issues)
            - Best results with speech audio (not music)
            - Requires sufficient system RAM for processing
            """)
    
    return interface

def main():
    """Main function to run the application"""
    print("🎵 MossFormer2 Audio Enhancement System")
    print("=" * 50)
    print(f"PyTorch version: {torch.__version__}")
    print(f"TorchAudio version: {torchaudio.__version__}")
    print(f"Gradio version: {gr.__version__}")
    
    # Check if model path exists
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️  Warning: Model not found at {MODEL_PATH}")
        print("Please update the MODEL_PATH variable with the correct path to your checkpoint.pt file")
    
    # Check system resources
    if torch.cuda.is_available():
        print(f"🚀 CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("💻 Using CPU for inference")
    
    # Create and launch interface
    interface = create_gradio_interface()
    
    print("🚀 Starting Gradio interface...")
    interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )

if __name__ == "__main__":
    main()
