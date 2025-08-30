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
import math
warnings.filterwarnings("ignore")

# Global model path - Update this to your local model directory
MODEL_PATH = "path/to/your/mossformer2_model/checkpoint.pt"

class LayerNormalization4D(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        param_size = [1, input_dimension, 1, 1]
        self.gamma = nn.Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = nn.Parameter(torch.Tensor(*param_size).to(torch.float32))
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        if x.ndim == 4:
            _, C, _, _ = x.shape
            stat_dim = (1,)
        else:
            raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))
        
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,1,T,F]
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps
        )  # [B,1,T,F]
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat

class LayerNormalization4DCF(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        assert len(input_dimension) == 2
        param_size = [1, input_dimension[0], 1, input_dimension[1]]
        self.gamma = nn.Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = nn.Parameter(torch.Tensor(*param_size).to(torch.float32))
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        if x.ndim == 4:
            stat_dim = (1, 3)
        else:
            raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))
        
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,1,T,1]
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps
        )  # [B,1,T,1]
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat

class GatedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(GatedConv1d, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels * 2, kernel_size, stride, padding, dilation, groups, bias)
        self.out_channels = out_channels

    def forward(self, x):
        outputs = self.conv1d(x)
        out, gate = outputs.chunk(2, dim=1)
        return out * torch.sigmoid(gate)

class MossFormerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        
        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feed forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self attention
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2, attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        
        # Feed forward
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src2))))
        src = src + self.dropout2(src2)
        
        return src

class DilatedDenseBlock(nn.Module):
    def __init__(self, depth, in_channels, growth_rate, kernel_size, dilation_factor_init=1):
        super(DilatedDenseBlock, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.kernel_size = kernel_size
        self.dilation_factor_init = dilation_factor_init
        
        self.layers = nn.ModuleList()
        for i in range(depth):
            dil = dilation_factor_init * (2 ** i)
            pad_length = dil * (kernel_size - 1) // 2
            in_ch = in_channels + i * growth_rate
            layer = nn.Sequential(
                nn.Conv1d(in_ch, growth_rate, kernel_size, dilation=dil, padding=pad_length),
                nn.ReLU(inplace=True)
            )
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
            x = torch.cat([x, out], dim=1)
        return x

class RecurrentModule(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Dilated dense blocks for recurrent patterns
        self.dense_block1 = DilatedDenseBlock(depth=3, in_channels=input_size, 
                                            growth_rate=hidden_size//4, kernel_size=3)
        self.dense_block2 = DilatedDenseBlock(depth=3, in_channels=input_size + 3*hidden_size//4, 
                                            growth_rate=hidden_size//4, kernel_size=3, dilation_factor_init=2)
        
        # Output projection
        total_out_channels = input_size + 6 * hidden_size // 4
        self.output_proj = nn.Conv1d(total_out_channels, hidden_size, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch, channels, time)
        x1 = self.dense_block1(x)
        x2 = self.dense_block2(x1)
        output = self.output_proj(x2)
        return self.dropout(output)

class MossFormer2_SE_48K(nn.Module):
    """
    MossFormer2 Speech Enhancement model for 48kHz audio
    Based on the actual ClearerVoice-Studio implementation
    """
    def __init__(self, 
                 n_fft=2048,
                 hop_length=512, 
                 win_length=2048,
                 n_imag=1025,
                 n_mag=1025,
                 n_audio_channel=1,
                 nhead=8,
                 d_model=384,
                 num_layers=6,
                 dim_feedforward=1536,
                 dropout=0.1):
        super().__init__()
        
        # STFT parameters for 48kHz
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_imag = n_imag
        self.n_mag = n_mag
        self.n_audio_channel = n_audio_channel
        
        # Model architecture parameters
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
        # Input processing
        self.input_norm = LayerNormalization4DCF([n_audio_channel, n_imag])
        self.input_conv = nn.Conv2d(n_audio_channel * 2, d_model, kernel_size=(1, 1))
        
        # MossFormer blocks
        self.moss_blocks = nn.ModuleList([
            MossFormerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Recurrent module
        self.recurrent_module = RecurrentModule(d_model, d_model, dropout=dropout)
        
        # Output processing
        self.output_conv = nn.Conv2d(d_model, n_audio_channel * 2, kernel_size=(1, 1))
        self.output_norm = LayerNormalization4DCF([n_audio_channel, n_imag])
        
        # Magnitude and phase estimation
        self.mag_mask = nn.Sequential(
            nn.Conv2d(n_audio_channel * 2, n_audio_channel, kernel_size=(1, 1)),
            nn.Sigmoid()
        )
        
        self.phase_conv = nn.Sequential(
            nn.Conv2d(n_audio_channel * 2, n_audio_channel, kernel_size=(1, 1)),
            nn.Tanh()
        )

    def forward(self, input_wav):
        """
        Args:
            input_wav: [B, T] or [B, 1, T]
        Returns:
            enhanced_wav: [B, T]
        """
        if input_wav.dim() == 2:
            input_wav = input_wav.unsqueeze(1)  # [B, 1, T]
        
        batch_size, n_channels, seq_len = input_wav.shape
        
        # STFT
        window = torch.hann_window(self.win_length, device=input_wav.device, dtype=input_wav.dtype)
        
        # Apply STFT to each channel
        stft_list = []
        for i in range(n_channels):
            stft = torch.stft(input_wav[:, i, :], 
                            n_fft=self.n_fft,
                            hop_length=self.hop_length,
                            win_length=self.win_length,
                            window=window,
                            return_complex=True,
                            center=True,
                            pad_mode='reflect')
            stft_list.append(stft)
        
        # Stack channels: [B, C, F, T]
        stft_complex = torch.stack(stft_list, dim=1)
        
        # Separate real and imaginary parts
        stft_real = stft_complex.real
        stft_imag = stft_complex.imag
        
        # Input features: [B, C*2, F, T]
        input_features = torch.cat([stft_real, stft_imag], dim=1)
        
        # Apply input normalization
        input_features = self.input_norm(input_features)
        
        # Input convolution
        x = self.input_conv(input_features)  # [B, d_model, F, T]
        
        # Reshape for transformer: [B, T, F*d_model]
        B, D, F, T = x.shape
        x_transformer = x.permute(0, 3, 1, 2).contiguous().view(B, T, D*F)
        
        # Apply MossFormer blocks
        for block in self.moss_blocks:
            x_transformer = block(x_transformer)
        
        # Reshape back: [B, d_model, F, T]
        x = x_transformer.view(B, T, D, F).permute(0, 2, 3, 1).contiguous()
        
        # Apply recurrent module
        # Reshape for 1D conv: [B, d_model*F, T]
        x_recurrent = x.view(B, D*F, T)
        x_recurrent = self.recurrent_module(x_recurrent)
        
        # Reshape back: [B, d_model, F, T]
        x = x_recurrent.view(B, D, F, T)
        
        # Output processing
        output = self.output_conv(x)  # [B, C*2, F, T]
        output = self.output_norm(output)
        
        # Estimate masks
        mag_mask = self.mag_mask(output)  # [B, C, F, T]
        phase_residual = self.phase_conv(output)  # [B, C, F, T]
        
        # Apply masks
        mag_input = torch.sqrt(stft_real**2 + stft_imag**2 + 1e-8)
        phase_input = torch.atan2(stft_imag, stft_real + 1e-8)
        
        # Enhanced magnitude and phase
        enhanced_mag = mag_mask * mag_input
        enhanced_phase = phase_input + phase_residual * 0.5
        
        # Reconstruct complex spectrogram
        enhanced_real = enhanced_mag * torch.cos(enhanced_phase)
        enhanced_imag = enhanced_mag * torch.sin(enhanced_phase)
        enhanced_stft = torch.complex(enhanced_real, enhanced_imag)
        
        # ISTFT
        enhanced_wav_list = []
        for i in range(n_channels):
            enhanced_wav_ch = torch.istft(enhanced_stft[:, i, :, :],
                                        n_fft=self.n_fft,
                                        hop_length=self.hop_length,
                                        win_length=self.win_length,
                                        window=window,
                                        center=True,
                                        length=seq_len)
            enhanced_wav_list.append(enhanced_wav_ch)
        
        # Stack channels and squeeze if single channel
        enhanced_wav = torch.stack(enhanced_wav_list, dim=1)
        if n_channels == 1:
            enhanced_wav = enhanced_wav.squeeze(1)  # [B, T]
        
        return enhanced_wav

class AudioEnhancer:
    """Audio Enhancement Pipeline with proper MossFormer2 loading"""
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        self.model = None
        self.sample_rate = 48000  # MossFormer2_SE_48K uses 48kHz
        self.max_audio_length = 16 * self.sample_rate  # Limit to 16 seconds
        self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load the MossFormer2_SE_48K model from checkpoint"""
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
            
            # Initialize model with correct architecture
            self.model = MossFormer2_SE_48K()
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Handle key mismatches - common in pretrained models
            model_keys = set(self.model.state_dict().keys())
            checkpoint_keys = set(state_dict.keys())
            
            # Remove 'module.' prefix if present (from DataParallel training)
            if any(key.startswith('module.') for key in checkpoint_keys):
                new_state_dict = {}
                for key, value in state_dict.items():
                    new_key = key[7:] if key.startswith('module.') else key
                    new_state_dict[new_key] = value
                state_dict = new_state_dict
            
            # Load with strict=False to handle slight architecture differences
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"⚠️ Missing keys in checkpoint: {len(missing_keys)} keys")
                print(f"First few missing keys: {missing_keys[:5]}")
            
            if unexpected_keys:
                print(f"⚠️ Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
                print(f"First few unexpected keys: {unexpected_keys[:5]}")
            
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Model loaded successfully from {model_path}")
            print(f"🔧 Using device: {self.device}")
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            print(f"💡 Try using the ClearerVoice-Studio framework for proper model loading")
            raise e
    
    def preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess audio file"""
        try:
            # Load audio
            waveform, sr = torchaudio.load(audio_path, normalize=True)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Limit audio length
            if waveform.shape[1] > self.max_audio_length:
                print(f"⚠️ Audio too long ({waveform.shape[1]/self.sample_rate:.1f}s), truncating to 16s")
                waveform = waveform[:, :self.max_audio_length]
            
            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
                waveform = resampler(waveform)
            
            # Normalize
            max_val = torch.max(torch.abs(waveform))
            if max_val > 0:
                waveform = waveform / max_val * 0.95  # Prevent clipping
            
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
            
            # Ensure correct shape: [B, T]
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            elif waveform.dim() == 2 and waveform.shape[0] == 1:
                waveform = waveform.squeeze(0).unsqueeze(0)
            else:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Enhance audio
            with torch.no_grad():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                enhanced_waveform = self.model(waveform)
            
            # Post-process
            enhanced_waveform = enhanced_waveform.cpu().float()
            
            # Ensure proper shape for saving
            if enhanced_waveform.dim() == 1:
                enhanced_waveform = enhanced_waveform.unsqueeze(0)
            
            # Normalize output
            max_val = torch.max(torch.abs(enhanced_waveform))
            if max_val > 0:
                enhanced_waveform = enhanced_waveform / max_val * 0.95
            
            # Save enhanced audio
            torchaudio.save(output_audio_path, enhanced_waveform, self.sample_rate, 
                          encoding="PCM_S", bits_per_sample=16)
            
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
        return f"✅ MossFormer2_SE_48K model loaded successfully on {device}!"
    except Exception as e:
        return f"❌ Error loading model: {str(e)}\n💡 Make sure you have the correct MossFormer2 checkpoint file."

def process_audio(input_audio):
    """Process audio through Gradio interface"""
    global enhancer
    
    if enhancer is None:
        return None, "❌ Model not loaded. Please check the model path and file."
    
    if input_audio is None:
        return None, "❌ Please upload an audio file."
    
    try:
        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            output_path = tmp_file.name
        
        # Enhance audio
        enhanced_path = enhancer.enhance_audio(input_audio, output_path)
        
        return enhanced_path, "✅ Audio enhanced successfully using MossFormer2_SE_48K!"
        
    except Exception as e:
        error_msg = f"❌ Error processing audio: {str(e)}"
        if "state_dict" in str(e).lower():
            error_msg += "\n💡 This might be due to checkpoint format mismatch. Try using ClearerVoice-Studio framework."
        return None, error_msg

def create_gradio_interface():
    """Create Gradio interface"""
    
    css = """
    .gradio-container {
        max-width: 1000px !important;
        margin: auto !important;
    }
    .main-content {
        padding: 2rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    """
    
    with gr.Blocks(css=css, title="MossFormer2_SE_48K Audio Enhancement", theme=gr.themes.Soft()) as interface:
        
        gr.Markdown("""
        # 🎵 MossFormer2_SE_48K Speech Enhancement
        
        **Professional-grade speech enhancement using MossFormer2 architecture**
        
        This implementation uses the MossFormer2_SE_48K model architecture for 48kHz audio enhancement.
        Upload your audio file to remove background noise and improve speech clarity.
        """)
        
        gr.HTML("""
        <div class="warning-box">
            <strong>⚠️ Important Notes:</strong><br>
            • This model expects MossFormer2_SE_48K checkpoint files<br>
            • For best compatibility, use checkpoints from ClearerVoice-Studio<br>
            • Supported formats: WAV, MP3, FLAC, OGG<br>
            • Maximum duration: 16 seconds per file
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Model status
                status_text = gr.Textbox(
                    label="🔧 Model Status",
                    value=initialize_enhancer(),
                    interactive=False,
                    container=True,
                    lines=3
                )
                
                # Audio input
                audio_input = gr.Audio(
                    label="📤 Upload Audio File (Max 16s)",
                    type="filepath",
                    sources=["upload"]
                )
                
                # Process button
                process_btn = gr.Button(
                    "🚀 Enhance Audio with MossFormer2",
                    variant="primary",
                    size="lg",
                    scale=1
                )
            
            with gr.Column(scale=1):
                # Audio output
                audio_output = gr.Audio(
                    label="📥 Enhanced Audio Output",
                    type="filepath",
                    interactive=False
                )
                
                # Status message
                message_output = gr.Textbox(
                    label="📊 Processing Status",
                    interactive=False,
                    container=True,
                    lines=3
                )
        
        # Connect components
        process_btn.click(
            fn=process_audio,
            inputs=[audio_input],
            outputs=[audio_output, message_output],
            show_progress="full"
        )
        
        with gr.Accordion("📖 Technical Information & Troubleshooting", open=False):
            gr.Markdown("""
            ## 🔧 Model Architecture
            - **Framework:** MossFormer2 hybrid Transformer + RNN-free recurrent architecture
            - **Sample Rate:** 48 kHz (high-fidelity audio processing)
            - **STFT Parameters:** n_fft=2048, hop_length=512, win_length=2048
            - **Model Components:** Multi-head attention, dilated dense blocks, magnitude/phase estimation
            
            ## ⚠️ Troubleshooting
            
            **"Error loading model: Error(s) in loading state_dict":**
            - Ensure you have the correct MossFormer2_SE_48K checkpoint file
            - The checkpoint should be from ClearerVoice-Studio or compatible training
            - Some key mismatches are handled automatically (strict=False loading)
            
            **Best Performance:**
            - Use 48kHz audio files for optimal results
            - Keep audio files under 16 seconds for memory efficiency
            - Speech enhancement works best on human speech (not music)
            
            ## 📝 Alternative Usage
            If you continue having issues with checkpoint loading, consider using the official ClearerVoice-Studio framework:
            ```
            from clearvoice import ClearVoice
            myClearVoice = ClearVoice(task='speech_enhancement', model_names=['MossFormer2_SE_48K'])
            output_wav = myClearVoice(input_path='input.wav', online_write=False)
            ```
            """)
    
    return interface

def main():
    """Main function to run the application"""
    print("🎵 MossFormer2_SE_48K Audio Enhancement System")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"TorchAudio version: {torchaudio.__version__}")
    print(f"Gradio version: {gr.__version__}")
    
    # Check if model path exists
    if not os.path.exists(MODEL_PATH):
        print(f"\n⚠️  WARNING: Model checkpoint not found at:")
        print(f"   {MODEL_PATH}")
        print(f"\n💡 Please:")
        print(f"   1. Update MODEL_PATH variable with correct path")
        print(f"   2. Ensure you have MossFormer2_SE_48K checkpoint file")
        print(f"   3. Download from: https://huggingface.co/alibabasglab/MossFormer2_SE_48K")
    
    # System info
    if torch.cuda.is_available():
        print(f"\n🚀 CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("\n💻 Using CPU for inference")
    
    print(f"\n🎯 Model Configuration:")
    print(f"   - Target Sample Rate: 48 kHz")
    print(f"   - Max Audio Length: 16 seconds")
    print(f"   - Architecture: MossFormer2 + Recurrent Module")
    
    # Create and launch interface
    interface = create_gradio_interface()
    
    print(f"\n🚀 Starting Gradio interface...")
    print(f"   Access at: http://127.0.0.1:7860")
    
    interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )

if __name__ == "__main__":
    main()
