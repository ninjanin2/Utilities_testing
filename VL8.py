#!/usr/bin/env python3
"""
Keye-VL Professional Video Analysis Platform
============================================
Industry-grade video understanding system powered by Keye-VL-1.5-8B

Features:
- Single & batch video analysis
- Automatic chunking for long videos  
- Advanced preprocessing pipeline
- Production-ready architecture
- Gradio web interface
- Offline operation

Version: 4.0 (Production - No Flash Attention Required)
Author: AI Assistant
Date: November 2025
License: MIT
"""

import os
import sys
import gc
import json
import logging
import warnings
import atexit
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict

# Suppress warnings before imports
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import core libraries
import torch
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

# Import ML libraries
try:
    from transformers import AutoModel, AutoProcessor
    import transformers
    # Suppress transformers warnings
    transformers.logging.set_verbosity_error()
except ImportError:
    print("ERROR: transformers not installed. Run: pip install transformers")
    sys.exit(1)

try:
    from keye_vl_utils import process_vision_info
except ImportError:
    print("ERROR: keye-vl-utils not installed. Run: pip install keye-vl-utils")
    sys.exit(1)

# Import Gradio
try:
    import gradio as gr
except ImportError:
    print("ERROR: gradio not installed. Run: pip install gradio")
    sys.exit(1)

# ==================== LOGGING CONFIGURATION ====================
def setup_logging(log_dir: str = "./logs") -> logging.Logger:
    """Configure professional logging system"""
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("KeyeVL")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, "keye_vl.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(levelname)s | %(message)s'))

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

logger = setup_logging()
logger.info("="*70)
logger.info("Keye-VL Professional Video Analysis Platform v4.0")
logger.info("="*70)

# ==================== CONFIGURATION ====================
@dataclass
class SystemConfig:
    """
    Central configuration for the entire system.
    Modify these values to customize behavior.
    """

    # Model settings
    MODEL_PATH: str = "./Keye-VL-1_5-8B"  # <<<< UPDATE THIS PATH
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE: torch.dtype = torch.bfloat16

    # Video processing
    TARGET_FPS: float = 2.0
    MAX_FRAMES_PER_CHUNK: int = 96
    MIN_FRAMES: int = 4
    MAX_VIDEO_DURATION: int = 3600  # 1 hour max

    # Model inference
    MIN_PIXELS: int = 32 * 28 * 28
    MAX_PIXELS: int = 1280 * 28 * 28
    MAX_NEW_TOKENS: int = 2048
    TEMPERATURE: float = 0.3
    TOP_P: float = 0.9
    TOP_K: int = 50
    DO_SAMPLE: bool = True

    # Preprocessing
    ENABLE_PREPROCESSING: bool = True
    GAUSSIAN_KERNEL: Tuple[int, int] = (3, 3)
    CLAHE_CLIP: float = 2.0
    CLAHE_GRID: Tuple[int, int] = (8, 8)

    # Paths
    OUTPUT_DIR: str = "./analysis_results"
    TEMP_DIR: str = "./temp"
    CACHE_DIR: str = "./cache"
    LOG_DIR: str = "./logs"

    # Batch processing
    MAX_BATCH_SIZE: int = 10

    def __post_init__(self):
        """Create directories"""
        for directory in [self.OUTPUT_DIR, self.TEMP_DIR, self.CACHE_DIR, self.LOG_DIR]:
            os.makedirs(directory, exist_ok=True)

        logger.info("System Configuration:")
        logger.info(f"  Device: {self.DEVICE}")
        logger.info(f"  Model Path: {self.MODEL_PATH}")
        logger.info(f"  Output Dir: {self.OUTPUT_DIR}")

config = SystemConfig()

# ==================== RESOURCE MANAGER ====================
class ResourceManager:
    """
    Manages system resources, temporary files, and cleanup.
    Ensures no resource leaks in production environment.
    """

    def __init__(self):
        self.temp_files: List[str] = []
        self.temp_dirs: List[str] = []
        atexit.register(self.cleanup)
        logger.debug("ResourceManager initialized")

    def register_file(self, filepath: str):
        """Register temporary file for cleanup"""
        self.temp_files.append(filepath)
        logger.debug(f"Registered temp file: {filepath}")

    def register_dir(self, dirpath: str):
        """Register temporary directory for cleanup"""
        self.temp_dirs.append(dirpath)
        logger.debug(f"Registered temp dir: {dirpath}")

    def cleanup(self):
        """Clean up all registered resources"""
        logger.info("Cleaning up resources...")

        # Remove files
        for filepath in self.temp_files:
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    logger.debug(f"Removed: {filepath}")
            except Exception as e:
                logger.warning(f"Failed to remove {filepath}: {e}")

        # Remove directories
        for dirpath in self.temp_dirs:
            try:
                if os.path.exists(dirpath):
                    shutil.rmtree(dirpath)
                    logger.debug(f"Removed directory: {dirpath}")
            except Exception as e:
                logger.warning(f"Failed to remove {dirpath}: {e}")

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("Cleared CUDA cache")

        # Garbage collection
        gc.collect()
        logger.info("Resource cleanup complete")

resource_manager = ResourceManager()

# ==================== VIDEO PREPROCESSOR ====================
class VideoPreprocessor:
    """
    Advanced video preprocessing with multiple strategies.
    Handles frame extraction, enhancement, and optimization.
    """

    def __init__(self, config: SystemConfig):
        self.config = config
        logger.info("VideoPreprocessor initialized")

    def extract_metadata(self, video_path: str) -> Dict[str, Any]:
        """
        Extract comprehensive video metadata.

        Args:
            video_path: Path to video file

        Returns:
            Dictionary containing video metadata
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)

        try:
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                logger.warning(f"Invalid FPS: {fps}, using default 30.0")
                fps = 30.0

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

            duration = frame_count / fps if frame_count > 0 else 0

            metadata = {
                'path': video_path,
                'filename': Path(video_path).name,
                'fps': fps,
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'duration': duration,
                'fourcc': fourcc,
                'size_bytes': os.path.getsize(video_path)
            }

            logger.info(f"Video: {metadata['filename']}, {duration:.2f}s, {width}x{height}")
            return metadata

        finally:
            cap.release()

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply advanced preprocessing to a frame.

        Args:
            frame: Input frame in RGB format

        Returns:
            Preprocessed frame
        """
        if not self.config.ENABLE_PREPROCESSING:
            return frame

        try:
            # Gaussian blur for noise reduction
            frame = cv2.GaussianBlur(frame, self.config.GAUSSIAN_KERNEL, 0)

            # CLAHE for contrast enhancement
            lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)

            clahe = cv2.createCLAHE(
                clipLimit=self.config.CLAHE_CLIP,
                tileGridSize=self.config.CLAHE_GRID
            )
            l = clahe.apply(l)

            frame = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

            return frame

        except Exception as e:
            logger.warning(f"Preprocessing failed: {e}, using original frame")
            return frame

    def extract_frames(self, video_path: str, max_frames: Optional[int] = None) -> List[np.ndarray]:
        """
        Extract frames from video with uniform sampling.

        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract

        Returns:
            List of preprocessed frames
        """
        metadata = self.extract_metadata(video_path)
        max_frames = max_frames or self.config.MAX_FRAMES_PER_CHUNK

        cap = cv2.VideoCapture(video_path)
        frames = []

        try:
            # Calculate sampling interval
            interval = max(1, int(metadata['fps'] / self.config.TARGET_FPS))
            frame_idx = 0

            with tqdm(total=max_frames, desc="Extracting frames", leave=False) as pbar:
                while len(frames) < max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_idx % interval == 0:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_processed = self.preprocess_frame(frame_rgb)
                        frames.append(frame_processed)
                        pbar.update(1)

                    frame_idx += 1

            # Ensure minimum frames
            if 0 < len(frames) < self.config.MIN_FRAMES:
                logger.warning(f"Only {len(frames)} frames extracted, duplicating")
                while len(frames) < self.config.MIN_FRAMES:
                    frames.append(frames[-1].copy())

            if not frames:
                raise ValueError("No frames extracted from video")

            logger.info(f"Extracted {len(frames)} frames")
            return frames

        finally:
            cap.release()

    def chunk_video(self, video_path: str) -> List[Tuple[List[np.ndarray], float, float]]:
        """
        Split long video into manageable chunks.

        Args:
            video_path: Path to video file

        Returns:
            List of (frames, start_time, end_time) tuples
        """
        metadata = self.extract_metadata(video_path)
        duration = metadata['duration']

        chunk_duration = self.config.MAX_FRAMES_PER_CHUNK / self.config.TARGET_FPS
        num_chunks = max(1, int(np.ceil(duration / chunk_duration)))

        logger.info(f"Splitting {duration:.2f}s video into {num_chunks} chunks")

        chunks = []
        cap = cv2.VideoCapture(video_path)

        try:
            for chunk_idx in range(num_chunks):
                start_time = chunk_idx * chunk_duration
                end_time = min((chunk_idx + 1) * chunk_duration, duration)

                # Seek to start position
                start_frame = int(start_time * metadata['fps'])
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

                chunk_frames = []
                interval = max(1, int(metadata['fps'] / self.config.TARGET_FPS))
                current_frame = start_frame
                end_frame = int(end_time * metadata['fps'])

                while current_frame < end_frame and len(chunk_frames) < self.config.MAX_FRAMES_PER_CHUNK:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if (current_frame - start_frame) % interval == 0:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_processed = self.preprocess_frame(frame_rgb)
                        chunk_frames.append(frame_processed)

                    current_frame += 1

                if chunk_frames:
                    chunks.append((chunk_frames, start_time, end_time))
                    logger.info(f"Chunk {chunk_idx+1}: {len(chunk_frames)} frames, {start_time:.1f}s-{end_time:.1f}s")

            return chunks

        finally:
            cap.release()

    def create_temp_video(self, frames: List[np.ndarray], fps: float) -> str:
        """
        Create temporary video file from frames.

        Args:
            frames: List of frames
            fps: Output FPS

        Returns:
            Path to temporary video file
        """
        temp_file = tempfile.NamedTemporaryFile(
            suffix='.mp4',
            dir=self.config.TEMP_DIR,
            delete=False
        )
        temp_path = temp_file.name
        temp_file.close()

        resource_manager.register_file(temp_path)

        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_path, fourcc, fps, (w, h))

        try:
            if not out.isOpened():
                raise RuntimeError("Failed to create video writer")

            for frame in frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)

            logger.debug(f"Created temp video: {temp_path}")
            return temp_path

        finally:
            out.release()

# ==================== MODEL HANDLER ====================
class ModelHandler:
    """
    Handles Keye-VL model loading and inference.
    No flash_attn required - model automatically uses standard attention.
    """

    def __init__(self, config: SystemConfig):
        self.config = config
        self.model = None
        self.processor = None
        self.device = config.DEVICE

        logger.info("Initializing ModelHandler...")
        self.load_model()

    def load_model(self):
        """
        Load model and processor.
        CRITICAL: Do NOT specify attn_implementation - let model use defaults.
        """
        try:
            if not os.path.exists(self.config.MODEL_PATH):
                raise FileNotFoundError(
                    f"Model not found: {self.config.MODEL_PATH}\n"
                    f"Download with: huggingface-cli download Kwai-Keye/Keye-VL-1_5-8B --local-dir ./Keye-VL-1_5-8B"
                )

            logger.info("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                self.config.MODEL_PATH,
                trust_remote_code=True,
                min_pixels=self.config.MIN_PIXELS,
                max_pixels=self.config.MAX_PIXELS
            )

            logger.info("Loading model...")
            # CRITICAL: Load model WITHOUT specifying attn_implementation
            # This allows the model to use its default attention mechanism
            self.model = AutoModel.from_pretrained(
                self.config.MODEL_PATH,
                torch_dtype=self.config.DTYPE,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            ).eval()

            logger.info(f"âœ“ Model loaded successfully on {self.device}")
            logger.info(f"  Model dtype: {self.model.dtype}")

            self.warmup()

        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            raise

    def warmup(self):
        """Warm up model with dummy input"""
        try:
            logger.info("Warming up model...")
            dummy_messages = [{
                "role": "user",
                "content": [{"type": "text", "text": "Hello"}]
            }]
            _ = self.generate(dummy_messages, max_new_tokens=5)
            logger.info("âœ“ Warmup complete")
        except Exception as e:
            logger.warning(f"Warmup failed (non-critical): {e}")

    def generate(self, messages: List[Dict], max_new_tokens: Optional[int] = None,
                temperature: Optional[float] = None) -> str:
        """
        Generate text from messages.

        Args:
            messages: List of message dictionaries
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        max_new_tokens = max_new_tokens or self.config.MAX_NEW_TOKENS
        temperature = temperature or self.config.TEMPERATURE

        try:
            # Prepare input
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Process vision information
            image_inputs, video_inputs, mm_kwargs = process_vision_info(messages)

            # Tokenize
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **mm_kwargs
            ).to(self.device)

            # Generate
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=self.config.TOP_P,
                    top_k=self.config.TOP_K,
                    do_sample=self.config.DO_SAMPLE,
                    pad_token_id=self.processor.tokenizer.pad_token_id
                        if hasattr(self.processor, 'tokenizer') else None
                )

            # Decode
            generated_ids_trimmed = [
                out[len(inp):]
                for inp, out in zip(inputs.input_ids, generated_ids)
            ]

            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            # Cleanup
            if self.device == "cuda":
                torch.cuda.empty_cache()

            return output_text

        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            raise

# ==================== VIDEO ANALYZER ====================
class VideoAnalyzer:
    """
    Core video analysis engine.
    Orchestrates preprocessing, model inference, and result synthesis.
    """

    def __init__(self, model_handler: ModelHandler, preprocessor: VideoPreprocessor):
        self.model = model_handler
        self.preprocessor = preprocessor
        logger.info("VideoAnalyzer initialized")

    def analyze_video(self, video_path: str, prompt: Optional[str] = None,
                     progress_callback: Optional[Any] = None) -> Dict[str, Any]:
        """
        Analyze a single video.

        Args:
            video_path: Path to video file
            prompt: Custom analysis prompt
            progress_callback: Optional progress callback

        Returns:
            Analysis results dictionary
        """
        try:
            logger.info(f"Starting analysis: {video_path}")

            # Extract metadata
            metadata = self.preprocessor.extract_metadata(video_path)

            # Use default prompt if none provided
            prompt = prompt or "Provide a comprehensive and detailed analysis of this video, including all visual elements, actions, scenes, objects, and events."

            # Determine if chunking is needed
            estimated_frames = int(metadata['duration'] * config.TARGET_FPS)
            needs_chunking = estimated_frames > config.MAX_FRAMES_PER_CHUNK

            if needs_chunking:
                logger.info(f"Video requires chunking ({estimated_frames} estimated frames)")
                result = self._analyze_chunked(video_path, prompt, metadata, progress_callback)
            else:
                logger.info("Analyzing in single pass")
                result = self._analyze_single(video_path, prompt, metadata, progress_callback)

            # Save results
            self._save_results(result, video_path)

            logger.info("âœ“ Analysis complete")
            return result

        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            return {
                'status': 'error',
                'error': str(e),
                'video_path': video_path
            }

    def _analyze_single(self, video_path: str, prompt: str,
                       metadata: Dict, progress_callback: Any) -> Dict:
        """Analyze video in single pass"""
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
            'video_path': video_path,
            'metadata': metadata,
            'prompt': prompt,
            'analysis': analysis,
            'chunks': 1,
            'timestamp': datetime.now().isoformat()
        }

    def _analyze_chunked(self, video_path: str, prompt: str,
                        metadata: Dict, progress_callback: Any) -> Dict:
        """Analyze video in multiple chunks"""
        chunks = self.preprocessor.chunk_video(video_path)
        chunk_results = []

        for idx, (frames, start, end) in enumerate(chunks):
            logger.info(f"Processing chunk {idx+1}/{len(chunks)}")

            # Create temp video
            temp_video = self.preprocessor.create_temp_video(frames, metadata['fps'])

            try:
                messages = [{
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": temp_video,
                            "fps": config.TARGET_FPS,
                            "max_frames": len(frames)
                        },
                        {"type": "text", "text": f"{prompt}\nAnalyze this segment from {start:.1f}s to {end:.1f}s."}
                    ]
                }]

                analysis = self.model.generate(messages)

                chunk_results.append({
                    'chunk_id': idx + 1,
                    'start_time': start,
                    'end_time': end,
                    'analysis': analysis
                })

            except Exception as e:
                logger.error(f"Chunk {idx+1} failed: {e}")
                chunk_results.append({
                    'chunk_id': idx + 1,
                    'error': str(e)
                })

        # Synthesize results
        final_analysis = self._synthesize_results(chunk_results)

        return {
            'status': 'success',
            'video_path': video_path,
            'metadata': metadata,
            'prompt': prompt,
            'analysis': final_analysis,
            'chunk_results': chunk_results,
            'chunks': len(chunks),
            'timestamp': datetime.now().isoformat()
        }

    def _synthesize_results(self, chunk_results: List[Dict]) -> str:
        """Combine chunk analyses into final result"""
        valid_chunks = [c for c in chunk_results if 'error' not in c]

        if not valid_chunks:
            return "Error: All chunks failed to process"

        parts = []
        for chunk in valid_chunks:
            parts.append(
                f"**Segment {chunk['chunk_id']} "
                f"({chunk['start_time']:.1f}s-{chunk['end_time']:.1f}s)**\n\n"
                f"{chunk['analysis']}"
            )

        return "\n\n---\n\n".join(parts)

    def _save_results(self, result: Dict, video_path: str):
        """Save analysis results to JSON file"""
        try:
            video_name = Path(video_path).stem
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(
                config.OUTPUT_DIR,
                f"{video_name}_analysis_{timestamp}.json"
            )

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            logger.info(f"Results saved: {output_path}")

        except Exception as e:
            logger.error(f"Failed to save results: {e}")

    def batch_analyze(self, folder_path: str, prompt: Optional[str] = None) -> List[Dict]:
        """
        Analyze multiple videos in a folder.

        Args:
            folder_path: Path to folder containing videos
            prompt: Custom prompt to use for all videos

        Returns:
            List of analysis results
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        # Find video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm']
        video_files = []

        for ext in video_extensions:
            video_files.extend(Path(folder_path).glob(f'*{ext}'))
            video_files.extend(Path(folder_path).glob(f'*{ext.upper()}'))

        video_files = sorted(list(set(video_files)))
        logger.info(f"Found {len(video_files)} videos to analyze")

        results = []
        for idx, video_file in enumerate(video_files):
            logger.info(f"Processing video {idx+1}/{len(video_files)}: {video_file.name}")

            result = self.analyze_video(str(video_file), prompt)
            results.append(result)

        return results

# ==================== GRADIO INTERFACE ====================
class WebInterface:
    """Professional Gradio web interface"""

    def __init__(self, analyzer: VideoAnalyzer):
        self.analyzer = analyzer
        logger.info("WebInterface initialized")

    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface"""

        with gr.Blocks(
            title="Keye-VL Video Analysis",
            theme=gr.themes.Soft()
        ) as interface:

            gr.Markdown("""
            # ðŸŽ¬ Keye-VL Professional Video Analysis Platform

            **Industry-grade video understanding powered by Keye-VL-1.5-8B**

            Production-ready â€¢ Automatic chunking â€¢ Advanced preprocessing â€¢ Offline capable
            """)

            with gr.Tabs():
                # Single video analysis
                with gr.Tab("ðŸ“¹ Single Video Analysis"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            video_input = gr.Video(label="Upload Video")
                            prompt_input = gr.Textbox(
                                label="Custom Prompt (Optional)",
                                placeholder="Leave empty for comprehensive analysis...",
                                lines=3
                            )
                            analyze_btn = gr.Button(
                                "ðŸ” Analyze Video",
                                variant="primary",
                                size="lg"
                            )

                            gr.Markdown("""
                            **Supported:** MP4, AVI, MOV, MKV, WEBM  
                            **Features:** Automatic chunking, preprocessing  
                            **Output:** Detailed analysis + JSON export
                            """)

                        with gr.Column(scale=1):
                            output_text = gr.Textbox(
                                label="Analysis Results",
                                lines=20,
                                show_copy_button=True
                            )
                            output_json = gr.JSON(
                                label="Detailed Results",
                                visible=False
                            )
                            show_json_btn = gr.Button("ðŸ“Š Show JSON Details")

                # Batch analysis
                with gr.Tab("ðŸ“‚ Batch Video Analysis"):
                    with gr.Row():
                        with gr.Column():
                            folder_input = gr.Textbox(
                                label="Folder Path",
                                placeholder="/path/to/video/folder"
                            )
                            batch_prompt = gr.Textbox(
                                label="Prompt for All Videos",
                                lines=2
                            )
                            batch_btn = gr.Button(
                                "ðŸ” Analyze Folder",
                                variant="primary"
                            )

                        with gr.Column():
                            batch_output = gr.Textbox(
                                label="Batch Results",
                                lines=20,
                                show_copy_button=True
                            )

                # System info
                with gr.Tab("â„¹ï¸ System Info"):
                    gr.Markdown(f"""
                    ## Configuration

                    **Model:** Keye-VL-1.5-8B  
                    **Device:** {config.DEVICE}  
                    **Dtype:** {config.DTYPE}  
                    **Max Frames/Chunk:** {config.MAX_FRAMES_PER_CHUNK}  
                    **Target FPS:** {config.TARGET_FPS}  

                    **Flash Attention:** Not required âœ“  
                    **Offline Mode:** Supported âœ“  
                    **Production Ready:** Yes âœ“

                    ## Version
                    **Version:** 4.0  
                    **Status:** Production  
                    **Date:** November 2025
                    """)

            # Event handlers
            analyze_btn.click(
                fn=self.analyze_handler,
                inputs=[video_input, prompt_input],
                outputs=[output_text, output_json]
            )

            show_json_btn.click(
                fn=lambda: gr.update(visible=True),
                outputs=output_json
            )

            batch_btn.click(
                fn=self.batch_handler,
                inputs=[folder_input, batch_prompt],
                outputs=batch_output
            )

            gr.Markdown("""
            ---
            <div style="text-align: center; color: #666;">
                <p>Keye-VL Professional Video Analysis Platform v4.0</p>
                <p>No flash_attn required â€¢ Works out of the box â€¢ Production ready</p>
            </div>
            """)

        return interface

    def analyze_handler(self, video_path: str, prompt: str,
                       progress=gr.Progress()) -> Tuple[str, Dict]:
        """Handle single video analysis"""

        if not video_path:
            return "âš ï¸  Please upload a video first", {}

        try:
            result = self.analyzer.analyze_video(video_path, prompt, progress)

            if result['status'] == 'success':
                output = f"""# Analysis Results

**Video:** {result['metadata']['filename']}  
**Duration:** {result['metadata']['duration']:.2f}s  
**Resolution:** {result['metadata']['width']}x{result['metadata']['height']}  
**Chunks:** {result['chunks']}  
**Timestamp:** {result['timestamp']}

---

## Analysis

{result['analysis']}
"""
                return output, result
            else:
                return f"âŒ Error: {result.get('error', 'Unknown error')}", result

        except Exception as e:
            logger.error(f"Handler error: {e}", exc_info=True)
            return f"âŒ Error: {str(e)}", {'error': str(e)}

    def batch_handler(self, folder_path: str, prompt: str) -> str:
        """Handle batch analysis"""

        if not folder_path:
            return "âš ï¸  Please provide a folder path"

        try:
            results = self.analyzer.batch_analyze(folder_path, prompt)

            # Format output
            output = f"# Batch Analysis Results\n\n"
            output += f"**Folder:** {folder_path}\n"
            output += f"**Total Videos:** {len(results)}\n"
            output += f"**Successful:** {sum(1 for r in results if r['status'] == 'success')}\n\n"
            output += "---\n\n"

            for idx, result in enumerate(results, 1):
                if result['status'] == 'success':
                    output += f"## {idx}. {result['metadata']['filename']}\n\n"
                    output += f"{result['analysis'][:500]}...\n\n"
                    output += "---\n\n"
                else:
                    output += f"## {idx}. Error\n{result.get('error')}\n\n"

            return output

        except Exception as e:
            logger.error(f"Batch handler error: {e}", exc_info=True)
            return f"âŒ Error: {str(e)}"

# ==================== MAIN APPLICATION ====================
def main():
    """Main application entry point"""

    try:
        logger.info("Starting application...")
        logger.info(f"Python: {sys.version.split()[0]}")
        logger.info(f"PyTorch: {torch.__version__}")
        logger.info(f"CUDA Available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Version: {torch.version.cuda}")

        # Verify model exists
        if not os.path.exists(config.MODEL_PATH):
            logger.error(f"Model not found: {config.MODEL_PATH}")
            logger.error("Download with:")
            logger.error("  huggingface-cli download Kwai-Keye/Keye-VL-1_5-8B --local-dir ./Keye-VL-1_5-8B")
            return

        # Initialize components
        logger.info("Initializing components...")

        preprocessor = VideoPreprocessor(config)
        model_handler = ModelHandler(config)
        analyzer = VideoAnalyzer(model_handler, preprocessor)
        interface = WebInterface(analyzer)

        # Create and launch interface
        logger.info("Creating web interface...")
        app = interface.create_interface()

        logger.info("Launching application...")
        logger.info("="*70)

        app.queue(max_size=20).launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
            inbrowser=True
        )

    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        raise
    finally:
        logger.info("Shutting down...")
        resource_manager.cleanup()
        logger.info("Shutdown complete")

if __name__ == "__main__":
    main()