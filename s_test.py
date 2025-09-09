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
import glob

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

# Global variable for model directory path
MODEL_DIR = "/path/to/your/sgmse/model"  # Change this to your local model directory

def auto_detect_cuda_home():
    """
    Automatically detect CUDA installation without requiring admin privileges
    This works similar to how Whisper/Gemma models auto-detect CUDA
    """
    
    # Method 1: Check if PyTorch already knows CUDA location
    try:
        import torch.utils.cpp_extension
        pytorch_cuda_home = getattr(torch.utils.cpp_extension, 'CUDA_HOME', None)
        if pytorch_cuda_home and os.path.exists(pytorch_cuda_home):
            logging.info(f"Found CUDA via PyTorch: {pytorch_cuda_home}")
            return pytorch_cuda_home
    except (ImportError, AttributeError):
        pass
    
    # Method 2: Use nvcc to find CUDA installation
    try:
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            # Get nvcc path and derive CUDA_HOME
            nvcc_path = subprocess.run(['which', 'nvcc'], 
                                     capture_output=True, text=True, timeout=10)
            if nvcc_path.returncode == 0:
                nvcc_location = nvcc_path.stdout.strip()
                # CUDA_HOME is typically two levels up from nvcc (bin/nvcc -> ..)
                cuda_home = os.path.dirname(os.path.dirname(nvcc_location))
                if os.path.exists(cuda_home):
                    logging.info(f"Found CUDA via nvcc: {cuda_home}")
                    return cuda_home
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    # Method 3: Check standard CUDA installation paths
    standard_paths = [
        '/usr/local/cuda-11.8',
        '/usr/local/cuda-11.7',
        '/usr/local/cuda-11.6',
        '/usr/local/cuda-11.5',
        '/usr/local/cuda-11.4',
        '/usr/local/cuda-11.3',
        '/usr/local/cuda-11.2',
        '/usr/local/cuda-11.1',
        '/usr/local/cuda-11.0',
        '/usr/local/cuda',
        '/opt/cuda',
        '/usr/cuda',
    ]
    
    for path in standard_paths:
        if os.path.exists(path) and os.path.exists(os.path.join(path, 'bin', 'nvcc')):
            logging.info(f"Found CUDA at standard path: {path}")
            return path
    
    # Method 4: Search for CUDA installations using glob
    try:
        cuda_patterns = [
            '/usr/local/cuda*',
            '/opt/cuda*',
            '/usr/cuda*',
        ]
        
        for pattern in cuda_patterns:
            matches = glob.glob(pattern)
            for match in matches:
                if os.path.exists(os.path.join(match, 'bin', 'nvcc')):
                    logging.info(f"Found CUDA via glob search: {match}")
                    return match
    except Exception:
        pass
    
    # Method 5: Check conda environment
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        conda_cuda = os.path.join(conda_prefix, 'bin', 'nvcc')
        if os.path.exists(conda_cuda):
            logging.info(f"Found CUDA in conda environment: {conda_prefix}")
            return conda_prefix
    
    # Method 6: Use torch.cuda location as hint
    if torch.cuda.is_available():
        try:
            # Get CUDA runtime library location
            import ctypes
            cuda_lib = ctypes.CDLL('libcudart.so.11.0')  # For CUDA 11.x
            # This is a fallback - try common paths based on CUDA availability
            fallback_paths = ['/usr/local/cuda-11.8', '/usr/local/cuda']
            for path in fallback_paths:
                if os.path.exists(path):
                    logging.info(f"Using fallback CUDA path: {path}")
                    return path
        except:
            pass
    
    logging.warning("Could not automatically detect CUDA installation")
    return None

def setup_cuda_environment_runtime():
    """
    Setup CUDA environment variables at runtime without admin privileges
    """
    # Auto-detect CUDA installation
    cuda_home = auto_detect_cuda_home()
    
    if cuda_home:
        # Set environment variables for current process only
        os.environ['CUDA_HOME'] = cuda_home
        os.environ['CUDA_PATH'] = cuda_home
        
        # Add CUDA bin to PATH
        cuda_bin = os.path.join(cuda_home, 'bin')
        current_path = os.environ.get('PATH', '')
        if cuda_bin not in current_path:
            os.environ['PATH'] = f"{cuda_bin}:{current_path}"
        
        # Add CUDA lib to LD_LIBRARY_PATH
        cuda_lib64 = os.path.join(cuda_home, 'lib64')
        cuda_lib = os.path.join(cuda_home, 'lib')
        
        ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
        
        if os.path.exists(cuda_lib64) and cuda_lib64 not in ld_library_path:
            os.environ['LD_LIBRARY_PATH'] = f"{cuda_lib64}:{ld_library_path}"
        elif os.path.exists(cuda_lib) and cuda_lib not in ld_library_path:
            os.environ['LD_LIBRARY_PATH'] = f"{cuda_lib}:{ld_library_path}"
        
        # Set additional CUDA-related environment variables
        os.environ['CUDA_TOOLKIT_ROOT_DIR'] = cuda_home
        
        logging.info(f"CUDA environment setup completed: {cuda_home}")
        return True, cuda_home
    else:
        logging.error("Failed to setup CUDA environment - CUDA installation not found")
        return False, "CUDA installation not detected"

def check_ninja_installation():
    """Check if Ninja is properly installed"""
    try:
        result = subprocess.run(['ninja', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            logging.info(f"Ninja version: {result.stdout.strip()}")
            return True
        else:
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

def validate_cuda_setup():
    """Validate that CUDA setup is working properly"""
    try:
        # Check PyTorch CUDA availability
        if not torch.cuda.is_available():
            return False, "PyTorch cannot detect CUDA"
        
        # Check if we can create a CUDA tensor (basic CUDA functionality)
        try:
            test_tensor = torch.randn(10, device='cuda')
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            return False, f"CUDA tensor creation failed: {str(e)}"
        
        # Check CUDA_HOME environment variable
        cuda_home = os.environ.get('CUDA_HOME')
        if not cuda_home:
            return False, "CUDA_HOME not set"
        
        # Verify CUDA_HOME points to valid installation
        nvcc_path = os.path.join(cuda_home, 'bin', 'nvcc')
        if not os.path.exists(nvcc_path):
            return False, f"NVCC not found at {nvcc_path}"
        
        # Check if PyTorch cpp_extension can find CUDA
        try:
            from torch.utils.cpp_extension import CUDA_HOME
            if CUDA_HOME is None:
                return False, "PyTorch cpp_extension cannot find CUDA_HOME"
        except ImportError:
            return False, "PyTorch cpp_extension not available"
        
        return True, f"CUDA setup validated: {cuda_home}"
        
    except Exception as e:
        return False, f"CUDA validation failed: {str(e)}"

class SGMSEEnhancer:
    """
    SGMSE Speech Enhancement Model Wrapper with automatic CUDA detection
    """
    
    def __init__(self, model_dir: str):
        """
        Initialize the SGMSE model from local directory
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
        
        # Setup CUDA environment automatically
        if self.device.type == 'cuda':
            cuda_ok, cuda_msg = setup_cuda_environment_runtime()
            if not cuda_ok:
                logging.warning(f"CUDA setup issue: {cuda_msg}, trying to continue...")
            
            # Validate CUDA setup
            valid_ok, valid_msg = validate_cuda_setup()
            if not valid_ok:
                logging.warning(f"CUDA validation issue: {valid_msg}")
        
        # Check Ninja
        if not check_ninja_installation():
            raise RuntimeError("Ninja is not installed. Please run: pip install ninja")
        
        self.load_model()
    
    def load_model(self):
        """Load the SGMSE model from local directory with error handling"""
        try:
            # Ensure CUDA environment is set before importing model
            if self.device.type == 'cuda':
                setup_cuda_environment_runtime()
            
            # Add the model directory to Python path
            sys.path.insert(0, str(self.model_dir))
            
            # Import SGMSE modules
            from sgmse.model import ScoreModel
            
            # Find checkpoint file
            checkpoint_files = list(self.model_dir.glob("*.ckpt"))
            if not checkpoint_files:
                raise FileNotFoundError("No checkpoint files found in model directory")
            
            checkpoint_path = checkpoint_files[0]
            logging.info(f"Loading checkpoint from: {checkpoint_path}")
            
            # Try different loading strategies
            loading_strategies = [
                # Strategy 1: Direct loading to target device
                lambda: ScoreModel.load_from_checkpoint(
                    checkpoint_path, 
                    map_location=self.device,
                    strict=False
                ),
                # Strategy 2: Load to CPU first, then move to device
                lambda: self._load_to_cpu_then_move(checkpoint_path),
                # Strategy 3: Force CPU loading if CUDA fails
                lambda: ScoreModel.load_from_checkpoint(
                    checkpoint_path, 
                    map_location='cpu',
                    strict=False
                )
            ]
            
            for i, strategy in enumerate(loading_strategies):
                try:
                    logging.info(f"Trying loading strategy {i+1}...")
                    self.model = strategy()
                    
                    # Move to target device if loaded on CPU
                    if self.device.type == 'cuda' and next(self.model.parameters()).device.type == 'cpu':
                        self.model.to(self.device)
                    
                    self.model.eval()
                    logging.info(f"SGMSE model loaded successfully using strategy {i+1}")
                    return
                    
                except Exception as e:
                    logging.warning(f"Loading strategy {i+1} failed: {str(e)}")
                    if i == len(loading_strategies) - 1:
                        raise e
                    continue
            
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise
    
    def _load_to_cpu_then_move(self, checkpoint_path):
        """Load model to CPU first, then move to target device"""
        model = ScoreModel.load_from_checkpoint(
            checkpoint_path, 
            map_location='cpu',
            strict=False
        )
        return model
    
    def preprocess_audio(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        """Preprocess audio for SGMSE model"""
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            audio = resampler(audio)
        
        if audio.dim() > 1 and audio.size(0) > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
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
        """Enhance a single chunk of spectrogram"""
        with torch.no_grad():
            noisy_spec = noisy_spec.to(self.device)
            
            if noisy_spec.dim() == 2:
                noisy_spec = noisy_spec.unsqueeze(0)
            
            try:
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
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # Handle CUDA OOM by falling back to CPU
                    logging.warning("CUDA OOM detected, falling back to CPU processing")
                    torch.cuda.empty_cache()
                    
                    # Move to CPU
                    noisy_spec_cpu = noisy_spec.cpu()
                    model_device = next(self.model.parameters()).device
                    self.model.cpu()
                    
                    # Process on CPU
                    sampler = self.model.get_pc_sampler(
                        self.predictor, 
                        self.corrector,
                        y=noisy_spec_cpu,
                        N=self.N,
                        corrector_steps=self.corrector_steps,
                        snr=self.snr
                    )
                    
                    enhanced_spec, _ = sampler()
                    
                    # Move model back to original device
                    self.model.to(model_device)
                    
                    return enhanced_spec.squeeze(0).to(self.device)
                else:
                    raise e
    
    def enhance_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """Enhance speech from audio file"""
        try:
            audio, sr = torchaudio.load(audio_path)
            logging.info(f"Loaded audio: shape={audio.shape}, sr={sr}")
            
            audio = self.preprocess_audio(audio, sr)
            audio = audio.to(self.device)
            
            noisy_spec = self.audio_to_spec(audio)
            
            # Adaptive chunk size based on available memory
            if self.device.type == 'cuda':
                # Get available GPU memory
                try:
                    free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                    # Adjust chunk size based on available memory
                    if free_memory > 8 * 1024**3:  # 8GB
                        max_length = 1024
                    elif free_memory > 4 * 1024**3:  # 4GB
                        max_length = 512
                    else:
                        max_length = 256
                except:
                    max_length = 512
            else:
                max_length = 256  # Conservative for CPU
            
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
            
            enhanced_audio = self.spec_to_audio(enhanced_spec)
            enhanced_audio = self.post_process_for_asr(enhanced_audio)
            
            logging.info(f"Enhanced audio: shape={enhanced_audio.shape}")
            return enhanced_audio.cpu(), self.target_sr
            
        except Exception as e:
            logging.error(f"Error enhancing audio: {e}")
            raise
    
    def post_process_for_asr(self, audio: torch.Tensor) -> torch.Tensor:
        """Post-process enhanced audio to be ASR ready"""
        audio = audio / (torch.max(torch.abs(audio)) + 1e-8)
        audio = torch.sign(audio) * torch.pow(torch.abs(audio), 0.8)
        audio = audio * 0.9
        return audio

# Global model instance
enhancer = None

def initialize_model():
    """Initialize the SGMSE model with automatic CUDA setup"""
    global enhancer
    try:
        # Check if model directory exists
        if not os.path.exists(MODEL_DIR):
            return f"❌ Error: Model directory not found: {MODEL_DIR}"
        
        # Setup CUDA environment automatically (like Whisper/Gemma do)
        if torch.cuda.is_available():
            cuda_ok, cuda_msg = setup_cuda_environment_runtime()
            if not cuda_ok:
                return f"⚠️ CUDA Warning: {cuda_msg} - Will attempt CPU fallback"
        
        # Check Ninja installation
        if not check_ninja_installation():
            return "❌ Error: Ninja is not installed. Please run: pip install ninja"
        
        enhancer = SGMSEEnhancer(MODEL_DIR)
        device_info = f"Using {enhancer.device}"
        if enhancer.device.type == 'cuda':
            device_info += f" - {torch.cuda.get_device_name(0)}"
        
        return f"✅ Model initialized successfully! {device_info}"
        
    except Exception as e:
        return f"❌ Error initializing model: {str(e)}"

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
            **Auto-detects CUDA like Whisper/Gemma models - no admin privileges required!**
            
            **Features:**
            - ✅ Automatic CUDA detection and setup
            - ✅ Works without admin privileges
            - ✅ GPU memory management and fallbacks
            - ✅ ASR-optimized output (16kHz)
            - ✅ Handles any audio length
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
            init_btn = gr.Button("Initialize Model", variant="secondary")
        
        # System information (auto-updated)
        cuda_available = torch.cuda.is_available()
        ninja_available = check_ninja_installation()
        
        # Auto-detect CUDA for display
        cuda_detected, cuda_path = setup_cuda_environment_runtime()
        
        with gr.Row():
            gr.Markdown(
                f"""
                ### System Status
                - **PyTorch**: {torch.__version__} 
                - **CUDA Available**: {'✅' if cuda_available else '❌'} 
                - **CUDA Auto-Detection**: {'✅' if cuda_detected else '❌'} {cuda_path if cuda_detected else 'Not found'}
                - **Ninja Build System**: {'✅' if ninja_available else '❌ Run: pip install ninja'}
                - **Admin Rights Required**: ❌ No (auto-detection used)
                
                {f"**GPU**: {torch.cuda.get_device_name(0)}" if cuda_available else "**Device**: CPU only"}
                """
            )
        
        # Event handlers
        init_btn.click(initialize_model, outputs=status_text)
        enhance_btn.click(enhance_speech, inputs=audio_input, outputs=audio_output)
        
        gr.Markdown(
            """
            ### Usage Notes
            
            **No Setup Required!** This script automatically:
            - Detects your CUDA 11.8 installation
            - Sets required environment variables at runtime  
            - Works like Whisper/Gemma models (plug-and-play)
            - Handles GPU memory management automatically
            
            **If Issues Occur:**
            1. Ensure Ninja is installed: `pip install ninja`
            2. Restart the application if CUDA detection fails
            3. Check that your SGMSE model directory is correct
            
            **Supported Formats**: WAV, MP3, FLAC, M4A → 16kHz WAV output
            """
        )
    
    return interface

def main():
    """Main function with automatic system setup"""
    print("🎤 SGMSE Speech Enhancement")
    print("=" * 50)
    
    # Auto-setup like Whisper/Gemma models
    print("🔍 Auto-detecting system configuration...")
    
    # Check PyTorch and CUDA
    print(f"📦 PyTorch: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"🎮 CUDA Available: {'✅ Yes' if cuda_available else '❌ No'}")
    
    if cuda_available:
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"🎮 CUDA Version: {torch.version.cuda}")
    
    # Auto-detect CUDA installation
    cuda_detected, cuda_info = setup_cuda_environment_runtime()
    print(f"🔧 CUDA Auto-Detection: {'✅ Success' if cuda_detected else '❌ Failed'}")
    if cuda_detected:
        print(f"📂 CUDA Path: {cuda_info}")
    
    # Check Ninja
    ninja_ok = check_ninja_installation()
    print(f"🥷 Ninja Build System: {'✅ Ready' if ninja_ok else '❌ Missing'}")
    
    # Check model directory
    model_ok = os.path.exists(MODEL_DIR)
    print(f"🤖 Model Directory: {'✅ Found' if model_ok else '❌ Not Found'}")
    
    if not model_ok:
        print(f"\n❌ Please update MODEL_DIR variable to point to your SGMSE model")
        print(f"   Current: {MODEL_DIR}")
        return
    
    if not ninja_ok:
        print(f"\n❌ Please install Ninja: pip install ninja")
        return
    
    print(f"\n🚀 Starting Gradio interface...")
    print(f"   No admin privileges required!")
    print(f"   Auto-CUDA setup: {'Enabled' if cuda_detected else 'Disabled'}")
    
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
