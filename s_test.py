import os
import sys
import warnings
import torch
import torchaudio
import numpy as np
import gradio as gr
from pathlib import Path
import tempfile
from typing import Tuple, Optional
import logging
import subprocess

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

# Global variable for model directory path
MODEL_DIR = "/path/to/your/sgmse/model"  # Change this to your local model directory

def find_cuda_home():
    """Find CUDA installation directory automatically"""
    cuda_paths = [
        os.environ.get('CUDA_HOME'),
        os.environ.get('CUDA_PATH'),
        '/usr/local/cuda',
        '/opt/cuda',
        '/usr/cuda',
    ]
    
    # Check conda environment
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        cuda_paths.extend([
            conda_prefix,
            os.path.join(conda_prefix, 'pkgs', 'cuda-toolkit'),
        ])
    
    # Check for versioned CUDA installations
    for base_path in ['/usr/local', '/opt']:
        try:
            if os.path.exists(base_path):
                cuda_dirs = [d for d in os.listdir(base_path) if d.startswith('cuda')]
                cuda_paths.extend([os.path.join(base_path, d) for d in cuda_dirs])
        except (OSError, PermissionError):
            continue
    
    # Find valid CUDA installation
    for path in cuda_paths:
        if path and os.path.exists(path):
            nvcc_path = os.path.join(path, 'bin', 'nvcc')
            if os.path.exists(nvcc_path):
                logging.info(f"Found CUDA at: {path}")
                return path
    
    return None

def setup_cuda_environment():
    """Setup CUDA environment variables"""
    # First, try to get CUDA_HOME from PyTorch if available
    try:
        import torch.utils.cpp_extension
        pytorch_cuda_home = getattr(torch.utils.cpp_extension, 'CUDA_HOME', None)
        if pytorch_cuda_home:
            os.environ['CUDA_HOME'] = pytorch_cuda_home
            logging.info(f"Using PyTorch CUDA_HOME: {pytorch_cuda_home}")
            return pytorch_cuda_home
    except (ImportError, AttributeError):
        pass
    
    # If not set, try to find it automatically
    cuda_home = find_cuda_home()
    if cuda_home:
        os.environ['CUDA_HOME'] = cuda_home
        os.environ['CUDA_PATH'] = cuda_home
        
        # Add to PATH if not already there
        cuda_bin = os.path.join(cuda_home, 'bin')
        if cuda_bin not in os.environ.get('PATH', ''):
            os.environ['PATH'] = f"{cuda_bin}:{os.environ.get('PATH', '')}"
        
        # Add to LD_LIBRARY_PATH
        cuda_lib = os.path.join(cuda_home, 'lib64')
        if os.path.exists(cuda_lib):
            ld_path = os.environ.get('LD_LIBRARY_PATH', '')
            if cuda_lib not in ld_path:
                os.environ['LD_LIBRARY_PATH'] = f"{cuda_lib}:{ld_path}"
        
        logging.info(f"Set CUDA_HOME to: {cuda_home}")
        return cuda_home
    
    return None

def check_cuda_installation():
    """Comprehensive CUDA installation check"""
    try:
        # Check if PyTorch detects CUDA
        if not torch.cuda.is_available():
            logging.error("PyTorch cannot detect CUDA")
            return False, "PyTorch cannot detect CUDA. Please install CUDA-enabled PyTorch."
        
        # Check CUDA_HOME
        cuda_home = setup_cuda_environment()
        if not cuda_home:
            return False, "CUDA installation not found. Please install CUDA toolkit."
        
        # Verify CUDA compiler
        nvcc_path = os.path.join(cuda_home, 'bin', 'nvcc')
        if not os.path.exists(nvcc_path):
            return False, f"NVCC compiler not found at {nvcc_path}"
        
        # Test CUDA compilation capability
        try:
            from torch.utils.cpp_extension import CUDA_HOME
            if CUDA_HOME is None:
                return False, "PyTorch cpp_extension cannot find CUDA_HOME"
        except ImportError:
            return False, "PyTorch cpp_extension not available"
        
        logging.info(f"CUDA setup verified: {cuda_home}")
        return True, f"CUDA ready at {cuda_home}"
        
    except Exception as e:
        return False, f"CUDA check failed: {str(e)}"

def check_ninja_installation():
    """Check if Ninja is properly installed"""
    try:
        result = subprocess.run(['ninja', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            logging.info(f"Ninja version: {result.stdout.strip()}")
            return True
        else:
            logging.error("Ninja is installed but not working properly")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        logging.error("Ninja is not installed or not found in PATH")
        return False

class SGMSEEnhancer:
    """
    SGMSE Speech Enhancement Model Wrapper with proper CUDA setup
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
        self.target_sr = 16000
        self.n_fft = 1024
        self.hop_length = 256
        self.win_length = 1024
        
        # Model parameters
        self.N = 30
        self.corrector_steps = 1
        self.snr = 0.5
        self.predictor = 'euler_maruyama'
        self.corrector = 'ald'
        
        logging.info(f"Using device: {self.device}")
        
        # Setup CUDA environment first
        cuda_ok, cuda_msg = check_cuda_installation()
        if not cuda_ok:
            if self.device.type == 'cuda':
                raise RuntimeError(f"CUDA setup failed: {cuda_msg}")
            else:
                logging.warning(f"CUDA not available, using CPU: {cuda_msg}")
        
        # Check Ninja
        if not check_ninja_installation():
            raise RuntimeError("Ninja is not properly installed. Please run: pip install ninja")
        
        self.load_model()
    
    def load_model(self):
        """Load the SGMSE model from local directory"""
        try:
            # Add the model directory to Python path
            sys.path.insert(0, str(self.model_dir))
            
            # Import SGMSE modules
            from sgmse.model import ScoreModel
            
            # Find checkpoint file
            checkpoint_files = list(self.model_dir.glob("*.ckpt"))
            if not checkpoint_files:
                raise FileNotFoundError("No checkpoint files found in model directory")
            
            checkpoint_path = checkpoint_files
            logging.info(f"Loading checkpoint from: {checkpoint_path}")
            
            # Load the model with error handling
            try:
                self.model = ScoreModel.load_from_checkpoint(
                    checkpoint_path, 
                    map_location=self.device,
                    strict=False
                )
            except Exception as e:
                logging.error(f"Error loading checkpoint: {e}")
                # Try CPU loading first
                self.model = ScoreModel.load_from_checkpoint(
                    checkpoint_path, 
                    map_location='cpu'
                )
                self.model.to(self.device)
            
            self.model.eval()
            logging.info("SGMSE model loaded successfully")
            
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise
    
    # ... (rest of the methods remain the same as previous version)
    def preprocess_audio(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            audio = resampler(audio)
        
        if audio.dim() > 1 and audio.size(0) > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            
        return audio
    
    def audio_to_spec(self, audio: torch.Tensor) -> torch.Tensor:
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
        audio = torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length).to(spec.device)
        )
        return audio
    
    def enhance_chunk(self, noisy_spec: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            noisy_spec = noisy_spec.to(self.device)
            
            if noisy_spec.dim() == 2:
                noisy_spec = noisy_spec.unsqueeze(0)
            
            sampler = self.model.get_pc_sampler(
                self.predictor, 
                self.corrector,
                y=noisy_spec,
                N=self.N,
                corrector_steps=self.corrector_steps,
                snr=self.snr
            )
            
            enhanced_spec, _ = sampler()
            return enhanced_spec.squeeze(0)
    
    def enhance_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        try:
            audio, sr = torchaudio.load(audio_path)
            logging.info(f"Loaded audio: shape={audio.shape}, sr={sr}")
            
            audio = self.preprocess_audio(audio, sr)
            audio = audio.to(self.device)
            
            noisy_spec = self.audio_to_spec(audio)
            
            max_length = 512
            if noisy_spec.size(-1) > max_length:
                enhanced_specs = []
                for i in range(0, noisy_spec.size(-1), max_length):
                    chunk = noisy_spec[..., i:i+max_length]
                    enhanced_chunk = self.enhance_chunk(chunk)
                    enhanced_specs.append(enhanced_chunk)
                enhanced_spec = torch.cat(enhanced_specs, dim=-1)
            else:
                enhanced_spec = self.enhance_chunk(noisy_spec)
            
            enhanced_audio = self.spec_to_audio(enhanced_spec)
            enhanced_audio = self.post_process_for_asr(enhanced_audio)
            
            logging.info(f"Enhanced audio: shape={enhanced_audio.shape}")
            return enhanced_audio.cpu(), self.target_sr
            
        except Exception as e:
            logging.error(f"Error enhancing audio: {e}")
            raise
    
    def post_process_for_asr(self, audio: torch.Tensor) -> torch.Tensor:
        audio = audio / (torch.max(torch.abs(audio)) + 1e-8)
        audio = torch.sign(audio) * torch.pow(torch.abs(audio), 0.8)
        audio = audio * 0.9
        return audio

# Global model instance
enhancer = None

def initialize_model():
    """Initialize the SGMSE model with comprehensive checks"""
    global enhancer
    try:
        # Check if model directory exists
        if not os.path.exists(MODEL_DIR):
            return f"Error: Model directory not found: {MODEL_DIR}"
        
        # Check CUDA setup
        cuda_ok, cuda_msg = check_cuda_installation()
        if not cuda_ok and torch.cuda.is_available():
            return f"CUDA Error: {cuda_msg}"
        
        # Check Ninja installation
        if not check_ninja_installation():
            return "Error: Ninja is not installed. Please run: pip install ninja"
        
        enhancer = SGMSEEnhancer(MODEL_DIR)
        return f"Model initialized successfully! {cuda_msg}"
    except Exception as e:
        return f"Error initializing model: {str(e)}"

def enhance_speech(audio_file) -> Tuple[int, np.ndarray]:
    """Gradio interface function for speech enhancement"""
    global enhancer
    
    if enhancer is None:
        raise gr.Error("Model not initialized. Please click 'Initialize Model' first.")
    
    if audio_file is None:
        raise gr.Error("Please upload an audio file.")
    
    try:
        enhanced_audio, sr = enhancer.enhance_audio(audio_file)
        enhanced_audio_np = enhanced_audio.numpy()
        
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
            
            **Requirements:**
            - CUDA toolkit installed
            - Ninja build system: `pip install ninja`
            - CUDA_HOME environment variable set
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
                enhance_btn = gr.Button("Enhance Speech", variant="primary", size="lg")
            
            with gr.Column():
                gr.Markdown("### Output")
                audio_output = gr.Audio(label="Enhanced Audio (ASR Ready)", type="numpy")
        
        with gr.Row():
            status_text = gr.Textbox(
                label="System Status",
                value="Click 'Initialize Model' to start",
                interactive=False
            )
            init_btn = gr.Button("Initialize Model")
        
        # System diagnostics
        cuda_ok, cuda_msg = check_cuda_installation()
        ninja_ok = check_ninja_installation()
        
        with gr.Row():
            gr.Markdown(
                f"""
                ### System Diagnostics
                - **PyTorch Version**: {torch.__version__}
                - **CUDA Available**: {torch.cuda.is_available()}
                - **CUDA Status**: {'✅' if cuda_ok else '❌'} {cuda_msg}
                - **Ninja Installed**: {'✅' if ninja_ok else '❌'}
                - **CUDA_HOME**: {os.environ.get('CUDA_HOME', 'Not set')}
                """
            )
        
        # Event handlers
        init_btn.click(initialize_model, outputs=status_text)
        enhance_btn.click(enhance_speech, inputs=audio_input, outputs=audio_output)
        
        gr.Markdown(
            """
            ### CUDA Setup Instructions
            
            If you see CUDA_HOME errors:
            
            **Linux/Ubuntu:**
            ```
            # Find CUDA installation
            ls /usr/local/cuda*
            
            # Set CUDA_HOME
            export CUDA_HOME=/usr/local/cuda
            export PATH=$CUDA_HOME/bin:$PATH
            export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
            
            # Make permanent
            echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
            source ~/.bashrc
            ```
            
            **For Conda:**
            ```
            export CUDA_HOME=$CONDA_PREFIX
            ```
            
            **Verify Setup:**
            ```
            python -c "import torch; from torch.utils.cpp_extension import CUDA_HOME; print(torch.cuda.is_available(), CUDA_HOME)"
            ```
            
            This should print `(True, '/path/to/cuda')`
            """
        )
    
    return interface

def main():
    """Main function with comprehensive system checks"""
    print("SGMSE Speech Enhancement - System Check")
    print("=" * 50)
    
    # Check PyTorch
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    # Check CUDA setup
    cuda_ok, cuda_msg = check_cuda_installation()
    print(f"CUDA Setup: {'✅' if cuda_ok else '❌'} {cuda_msg}")
    
    # Check Ninja
    ninja_ok = check_ninja_installation()
    print(f"Ninja: {'✅' if ninja_ok else '❌'}")
    
    # Check model directory
    model_ok = os.path.exists(MODEL_DIR)
    print(f"Model Directory: {'✅' if model_ok else '❌'} {MODEL_DIR}")
    
    if not model_ok:
        print(f"\nERROR: Update MODEL_DIR variable to point to your SGMSE model directory")
        return
    
    if not ninja_ok:
        print(f"\nERROR: Install Ninja: pip install ninja")
        return
    
    if not cuda_ok and torch.cuda.is_available():
        print(f"\nWARNING: {cuda_msg}")
        print("Please set CUDA_HOME environment variable")
    
    print("\nStarting Gradio interface...")
    interface = create_gradio_interface()
    
    interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True
    )

if __name__ == "__main__":
    main()
