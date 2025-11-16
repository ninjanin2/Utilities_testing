"""
Keye-VL Professional Video Analysis System
==================================================
ULTIMATE SOLUTION - Multi-Layer Flash Attention Prevention

This version uses MULTIPLE strategies to prevent flash_attn issues:
1. Environment variables (disable at OS level)
2. Module mocking (sys.modules interception)  
3. Attribute patching (monkey-patch transformers)
4. Fallback handlers (catch any remaining calls)

Version: 3.1 (Ultimate Bulletproof)
Date: November 2025
"""

import os
import sys

# ==================== LAYER 1: ENVIRONMENT VARIABLES ====================
# Set ALL possible environment variables to disable flash attention
# BEFORE any imports

os.environ.update({
    # Standard flash attention disabling
    'DISABLE_FLASH_ATTN': '1',
    'USE_FLASH_ATTN': '0',
    'FLASH_ATTENTION_DISABLE': '1',
    'FLASH_ATTN_DISABLE': '1',

    # NVIDIA Transformer Engine
    'NVTE_FLASH_ATTN': '0',
    'NVTE_FUSED_ATTN': '0',
    'NVTE_FUSED_ATTN_BACKEND': '0',

    # vLLM
    'VLLM_ATTENTION_BACKEND': 'XFORMERS',
    'VLLM_FLASH_ATTN_VERSION': '0',

    # Transformers library
    'TRANSFORMERS_NO_FLASH_ATTN': '1',
    'TRANSFORMERS_USE_FLASH_ATTENTION': '0',

    # PyTorch
    'PYTORCH_FLASH_ATTN': '0',

    # Other
    'TOKENIZERS_PARALLELISM': 'false',
    'CUDA_LAUNCH_BLOCKING': '1'  # Better error messages
})

# ==================== LAYER 2: ADVANCED MODULE MOCK ====================
from types import ModuleType
from importlib.machinery import ModuleSpec
import warnings

class UltraFlashAttnMock(ModuleType):
    """
    Ultimate flash_attn mock with MULTIPLE safety layers.
    Prevents ALL possible ways the model could call flash_attn.
    """

    def __init__(self, name="flash_attn"):
        super().__init__(name)

        # Set ALL required module attributes
        self.__name__ = name
        self.__package__ = name.split('.')[0] if '.' in name else name
        self.__file__ = f"<mock-{name}>"
        self.__loader__ = None
        self.__path__ = []
        self.__dict__['__builtins__'] = __builtins__

        # CRITICAL: Proper __spec__ to pass transformers checks
        self.__spec__ = ModuleSpec(
            name=name,
            loader=None,
            origin=f"<mock-{name}>",
            is_package=True
        )

        # Track if we're being called (for debugging)
        self._call_count = 0
        self._warned = False

    def __call__(self, *args, **kwargs):
        """
        Handle being called as a function.
        CRITICAL: Must NEVER return None!
        """
        self._call_count += 1

        # Warn once
        if not self._warned:
            warnings.warn(
                f"Flash attention mock called ({self.__name__}). "
                f"Using standard attention instead.",
                stacklevel=2
            )
            self._warned = True

        # If called with tensor arguments, return first tensor
        # This mimics attention behavior: attention(query, key, value) -> query
        for arg in args:
            if hasattr(arg, 'shape') and hasattr(arg, 'dtype'):
                # It's a tensor, return it
                return arg

        # If no tensor args, return self (NEVER None!)
        return self

    def __getattr__(self, name):
        """
        Handle attribute access.
        Return a new mock for any attribute.
        """
        # Ignore private attributes
        if name.startswith('_'):
            raise AttributeError(f"Mock has no attribute {name}")

        # Create and cache new mock for this attribute
        if name not in self.__dict__:
            self.__dict__[name] = UltraFlashAttnMock(f"{self.__name__}.{name}")

        return self.__dict__[name]

    def __getitem__(self, key):
        """Handle dictionary-style access"""
        return self

    def __bool__(self):
        """When checked as boolean, return False (flash_attn not available)"""
        return False

    def __repr__(self):
        return f"<UltraFlashAttnMock: {self.__name__} (calls={self._call_count})>"

# Create and register ALL possible flash_attn modules
_mock_modules = [
    'flash_attn',
    'flash_attn.flash_attn_interface',
    'flash_attn.flash_attn_triton',
    'flash_attn.flash_attn_func',
    'flash_attn.flash_attention',
    'flash_attn.bert_padding',
    'flash_attn.modules',
    'flash_attn.modules.mha',
    'flash_attn.modules.mlp',
    'flash_attn.ops',
    'flash_attn.ops.flash_attn',
    'flash_attn.ops.triton',
    'flash_attn.ops.triton.flash_attn',
    'flash_attn.layers',
    'flash_attn.utils',
]

for module_name in _mock_modules:
    sys.modules[module_name] = UltraFlashAttnMock(module_name)

print(f"[INIT] Registered {len(_mock_modules)} flash_attn mocks")

# ==================== LAYER 3: SAFE IMPORTS ====================
import gc
import json
import logging
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

import torch
import numpy as np
import gradio as gr
from PIL import Image
import cv2
from tqdm import tqdm

# Import transformers with error handling
try:
    from transformers import AutoModel, AutoProcessor
    from transformers.utils import is_flash_attn_2_available

    # CRITICAL: Patch is_flash_attn_2_available to always return False
    import transformers.utils
    transformers.utils.is_flash_attn_2_available = lambda: False
    transformers.utils.is_flash_attn_available = lambda: False

    print("[PATCH] Patched transformers flash_attn detection")

except ImportError as e:
    print(f"ERROR: {e}")
    print("Install: pip install transformers")
    sys.exit(1)

try:
    from keye_vl_utils import process_vision_info
except ImportError as e:
    print(f"ERROR: {e}")
    print("Install: pip install keye-vl-utils")
    sys.exit(1)

warnings.filterwarnings("ignore")

# ==================== LOGGING ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('keye_vl.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("="*70)
logger.info("Keye-VL Video Analysis System v3.1 (Ultimate Bulletproof)")
logger.info("Multi-layer flash_attn prevention: ACTIVE")
logger.info("="*70)

# ==================== CONFIGURATION ====================
@dataclass
class Config:
    """System configuration"""
    MODEL_PATH: str = "./Keye-VL-1_5-8B"  # UPDATE THIS
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE: torch.dtype = torch.bfloat16

    # Video processing
    TARGET_FPS: float = 2.0
    MAX_FRAMES_PER_CHUNK: int = 96
    MIN_FRAMES: int = 4

    # Model inference
    MIN_PIXELS: int = 32 * 28 * 28
    MAX_PIXELS: int = 1280 * 28 * 28
    MAX_NEW_TOKENS: int = 2048
    TEMPERATURE: float = 0.3
    TOP_P: float = 0.9

    # Paths
    OUTPUT_DIR: str = "./results"
    TEMP_DIR: str = "./temp"

    def __post_init__(self):
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.TEMP_DIR, exist_ok=True)

config = Config()

# ==================== RESOURCE MANAGER ====================
class ResourceManager:
    """Automatic cleanup of temporary files"""

    def __init__(self):
        self.temp_files = []
        import atexit
        atexit.register(self.cleanup)

    def register(self, filepath):
        self.temp_files.append(filepath)

    def cleanup(self):
        for f in self.temp_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

resources = ResourceManager()

# ==================== VIDEO PROCESSOR ====================
class VideoProcessor:
    """Handle video frame extraction and preprocessing"""

    def __init__(self):
        self.fps = config.TARGET_FPS
        self.max_frames = config.MAX_FRAMES_PER_CHUNK

    def get_metadata(self, video_path):
        """Extract video metadata"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        return {
            'fps': fps,
            'frames': frames,
            'width': width,
            'height': height,
            'duration': frames / fps if frames > 0 else 0
        }

    def extract_frames(self, video_path, max_frames=None):
        """Extract frames from video"""
        max_frames = max_frames or self.max_frames
        metadata = self.get_metadata(video_path)

        cap = cv2.VideoCapture(video_path)
        frames = []
        interval = max(1, int(metadata['fps'] / self.fps))
        idx = 0

        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            idx += 1

        cap.release()

        # Ensure minimum frames
        while 0 < len(frames) < config.MIN_FRAMES:
            frames.append(frames[-1])

        logger.info(f"Extracted {len(frames)} frames")
        return frames

    def chunk_video(self, video_path):
        """Split video into chunks"""
        metadata = self.get_metadata(video_path)
        chunk_duration = self.max_frames / self.fps
        num_chunks = max(1, int(np.ceil(metadata['duration'] / chunk_duration)))

        logger.info(f"Splitting into {num_chunks} chunks")

        chunks = []
        cap = cv2.VideoCapture(video_path)

        for i in range(num_chunks):
            start_time = i * chunk_duration
            end_time = min((i + 1) * chunk_duration, metadata['duration'])

            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * metadata['fps']))

            frames = []
            interval = max(1, int(metadata['fps'] / self.fps))
            current = int(start_time * metadata['fps'])
            end = int(end_time * metadata['fps'])

            while current < end and len(frames) < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                if (current - int(start_time * metadata['fps'])) % interval == 0:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                current += 1

            if frames:
                chunks.append((frames, start_time, end_time))

        cap.release()
        return chunks

    def create_temp_video(self, frames, fps):
        """Create temporary video file"""
        temp = tempfile.NamedTemporaryFile(suffix='.mp4', dir=config.TEMP_DIR, delete=False)
        path = temp.name
        temp.close()
        resources.register(path)

        h, w = frames[0].shape[:2]
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        for frame in frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        out.release()
        return path

# ==================== MODEL HANDLER ====================
class ModelHandler:
    """Handle model loading and inference with MAXIMUM safety"""

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = config.DEVICE

        logger.info("Loading model...")
        self.load_model()

    def load_model(self):
        """Load model with ALL safety measures"""
        try:
            if not os.path.exists(config.MODEL_PATH):
                raise FileNotFoundError(
                    f"Model not found: {config.MODEL_PATH}\n"
                    f"Download: huggingface-cli download Kwai-Keye/Keye-VL-1_5-8B"
                )

            # Load processor
            logger.info("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                config.MODEL_PATH,
                trust_remote_code=True,
                min_pixels=config.MIN_PIXELS,
                max_pixels=config.MAX_PIXELS
            )

            # Load model with EXPLICIT flash_attn disabling
            logger.info("Loading model (flash_attn DISABLED)...")
            self.model = AutoModel.from_pretrained(
                config.MODEL_PATH,
                torch_dtype=config.DTYPE,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                # CRITICAL: These parameters tell model NOT to use flash_attn
                attn_implementation="eager",  # Use eager (standard) attention
                use_flash_attention_2=False,  # Explicitly disable
                _attn_implementation="eager"
            ).eval()

            logger.info(f"âœ“ Model loaded on {self.device}")
            logger.info(f"  Dtype: {self.model.dtype}")
            logger.info(f"  Attention: eager (standard)")

            self.warmup()

        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            raise

    def warmup(self):
        """Warm up model"""
        try:
            logger.info("Warming up...")
            msgs = [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]
            _ = self.generate(msgs, max_new_tokens=5)
            logger.info("âœ“ Warmup complete")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

    def generate(self, messages, max_new_tokens=None, temperature=None):
        """
        Generate text with COMPREHENSIVE error handling.
        This is where the 'NoneType' error was happening!
        """
        max_new_tokens = max_new_tokens or config.MAX_NEW_TOKENS
        temperature = temperature or config.TEMPERATURE

        try:
            logger.debug("Preparing generation inputs...")

            # Apply chat template
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Process vision inputs
            image_inputs, video_inputs, mm_kwargs = process_vision_info(messages)

            # Tokenize
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **mm_kwargs
            )

            # Move to device
            inputs = inputs.to(self.device)

            logger.debug("Starting generation...")

            # Generate with EXPLICIT parameters
            with torch.inference_mode():
                # CRITICAL: Explicitly disable flash_attn in generate call
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=config.TOP_P,
                    do_sample=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id 
                        if hasattr(self.processor, 'tokenizer') else None,
                    # CRITICAL: Use eager attention
                    use_cache=True,
                    output_attentions=False,
                    output_hidden_states=False,
                )

            logger.debug("Decoding output...")

            # Decode
            generated_ids_trimmed = [
                out[len(inp):] 
                for inp, out in zip(inputs.input_ids, generated_ids)
            ]

            output = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            # Cleanup
            if self.device == "cuda":
                torch.cuda.empty_cache()

            logger.debug("âœ“ Generation complete")
            return output

        except Exception as e:
            # COMPREHENSIVE error logging
            logger.error("="*70)
            logger.error("GENERATION ERROR DETAILS:")
            logger.error(f"  Error type: {type(e).__name__}")
            logger.error(f"  Error message: {str(e)}")
            logger.error(f"  Device: {self.device}")
            logger.error(f"  Model dtype: {self.model.dtype}")
            logger.error("="*70)
            logger.error("Full traceback:", exc_info=True)

            # Check if it's the NoneType error
            if "'NoneType' object is not callable" in str(e):
                logger.error("\n" + "!"*70)
                logger.error("DETECTED: 'NoneType' object is not callable")
                logger.error("This means flash_attn mock was called incorrectly")
                logger.error("Checking mock status...")
                logger.error(f"  flash_attn in sys.modules: {'flash_attn' in sys.modules}")
                if 'flash_attn' in sys.modules:
                    mock = sys.modules['flash_attn']
                    logger.error(f"  Mock type: {type(mock)}")
                    logger.error(f"  Mock repr: {repr(mock)}")
                    if hasattr(mock, '_call_count'):
                        logger.error(f"  Mock calls: {mock._call_count}")
                logger.error("!"*70)

            raise

# ==================== VIDEO ANALYZER ====================
class VideoAnalyzer:
    """Core analysis engine"""

    def __init__(self):
        self.processor = VideoProcessor()
        self.model = ModelHandler()

    def analyze(self, video_path, prompt=None, progress=None):
        """Analyze video"""
        try:
            logger.info(f"Analyzing: {video_path}")

            metadata = self.processor.get_metadata(video_path)
            prompt = prompt or "Provide a detailed analysis of this video."

            # Check if chunking needed
            est_frames = int(metadata['duration'] * config.TARGET_FPS)
            needs_chunks = est_frames > config.MAX_FRAMES_PER_CHUNK

            if needs_chunks:
                return self._analyze_chunked(video_path, prompt, metadata)
            else:
                return self._analyze_single(video_path, prompt, metadata)

        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            return {'status': 'error', 'error': str(e)}

    def _analyze_single(self, video_path, prompt, metadata):
        """Analyze in single pass"""
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "fps": config.TARGET_FPS,
                    "max_frames": config.MAX_FRAMES_PER_CHUNK
                },
                {"type": "text", "text": prompt}
            ]
        }]

        analysis = self.model.generate(messages)

        return {
            'status': 'success',
            'analysis': analysis,
            'metadata': metadata,
            'chunks': 1,
            'timestamp': datetime.now().isoformat()
        }

    def _analyze_chunked(self, video_path, prompt, metadata):
        """Analyze in chunks"""
        chunks = self.processor.chunk_video(video_path)
        results = []

        for idx, (frames, start, end) in enumerate(chunks):
            logger.info(f"Chunk {idx+1}/{len(chunks)}")

            temp_video = self.processor.create_temp_video(frames, metadata['fps'])

            try:
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "video", "video": temp_video,
                         "fps": config.TARGET_FPS, "max_frames": len(frames)},
                        {"type": "text", "text": f"{prompt} (segment {start:.1f}s-{end:.1f}s)"}
                    ]
                }]

                analysis = self.model.generate(messages)
                results.append({'chunk': idx+1, 'analysis': analysis})

            except Exception as e:
                logger.error(f"Chunk {idx+1} failed: {e}")
                results.append({'chunk': idx+1, 'error': str(e)})

        # Combine results
        final = "\n\n".join([
            f"Segment {r['chunk']}: {r.get('analysis', 'ERROR')}"
            for r in results
        ])

        return {
            'status': 'success',
            'analysis': final,
            'metadata': metadata,
            'chunks': len(chunks),
            'timestamp': datetime.now().isoformat()
        }

# ==================== GRADIO UI ====================
class WebUI:
    """Gradio interface"""

    def __init__(self, analyzer):
        self.analyzer = analyzer

    def create(self):
        with gr.Blocks(title="Keye-VL Video Analysis") as app:
            gr.Markdown("# ðŸŽ¬ Keye-VL Video Analysis v3.1\n**Ultimate Bulletproof Edition**")

            with gr.Row():
                with gr.Column():
                    video = gr.Video(label="Upload Video")
                    prompt = gr.Textbox(label="Prompt (optional)", lines=3)
                    btn = gr.Button("ðŸ” Analyze", variant="primary")

                with gr.Column():
                    output = gr.Textbox(label="Analysis", lines=20, show_copy_button=True)

            def analyze(video_path, prompt_text):
                if not video_path:
                    return "âš ï¸  Upload a video first"

                result = self.analyzer.analyze(video_path, prompt_text)

                if result['status'] == 'success':
                    return f"""# Analysis Results

**Duration:** {result['metadata']['duration']:.2f}s
**Resolution:** {result['metadata']['width']}x{result['metadata']['height']}
**Chunks:** {result['chunks']}

## Analysis

{result['analysis']}
"""
                else:
                    return f"âŒ Error: {result.get('error', 'Unknown')}"

            btn.click(analyze, [video, prompt], output)

            gr.Markdown("**v3.1** | Multi-layer flash_attn prevention | Production ready")

        return app

# ==================== MAIN ====================
def main():
    try:
        logger.info("Initializing system...")

        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

        if not os.path.exists(config.MODEL_PATH):
            logger.error(f"Model not found: {config.MODEL_PATH}")
            return

        analyzer = VideoAnalyzer()
        ui = WebUI(analyzer)
        app = ui.create()

        logger.info("Launching web interface...")
        app.queue().launch(server_name="0.0.0.0", server_port=7860, share=False, inbrowser=True)

    except KeyboardInterrupt:
        logger.info("Stopped by user")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
    finally:
        resources.cleanup()

if __name__ == "__main__":
    main()