"""
Keye-VL-1.5-8B Video Analysis System
Professional-grade video analysis software with single and batch processing capabilities
Optimized for RTX A4000 16GB GPU with 32GB RAM

Version: 2.2 (Flash Attention Removed)
Last Updated: November 2025
"""

import os
import gc
import json
import logging
import warnings
import atexit
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from contextlib import contextmanager

import torch
import numpy as np
import gradio as gr
from PIL import Image
import cv2
from tqdm import tqdm

# Transformers imports
try:
    from transformers import AutoModel, AutoProcessor
    from keye_vl_utils import process_vision_info
except ImportError as e:
    raise ImportError(
        f"Required packages not installed: {e}\n"
        "Please run: pip install transformers keye-vl-utils"
    )

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('video_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== GLOBAL CONFIGURATION ====================
@dataclass
class Config:
    """Global configuration for the video analysis system"""

    # Model configuration
    MODEL_PATH: str = "/path/to/your/Keye-VL-1_5-8B"  # <<< CHANGE THIS TO YOUR MODEL PATH

    # GPU and memory settings
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    TORCH_DTYPE: torch.dtype = torch.bfloat16  # Use bfloat16 for better performance
    MAX_MEMORY: Dict[str, str] = field(default_factory=lambda: {"cuda:0": "15GB"})  # Reserve 1GB for system

    # Video processing parameters
    DEFAULT_FPS: float = 2.0  # Optimal FPS based on research
    MAX_FRAMES_PER_CHUNK: int = 96  # Optimal frame count per chunk
    MIN_FRAMES: int = 4
    MAX_TOTAL_FRAMES: int = 1024  # Maximum frames for entire video

    # Image/Video preprocessing
    MIN_PIXELS: int = 32 * 28 * 28  # For dynamic resolution
    MAX_PIXELS: int = 1280 * 28 * 28

    # Generation parameters
    MAX_NEW_TOKENS: int = 2048
    TEMPERATURE: float = 0.3
    TOP_P: float = 0.9
    DO_SAMPLE: bool = True

    # Batch processing
    BATCH_SIZE: int = 1  # Process one video at a time to optimize memory
    SIMILARITY_THRESHOLD: float = 0.5  # For search matching

    # Output settings
    OUTPUT_DIR: str = "./analysis_results"
    CACHE_DIR: str = "./cache"
    TEMP_DIR: str = "./temp"

    def __post_init__(self):
        """Create necessary directories"""
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        os.makedirs(self.TEMP_DIR, exist_ok=True)

config = Config()

# Cleanup handler for temporary files
temp_files_to_cleanup = []

def cleanup_temp_files():
    """Clean up temporary files on exit"""
    global temp_files_to_cleanup
    for temp_file in temp_files_to_cleanup:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                logger.debug(f"Cleaned up temp file: {temp_file}")
        except Exception as e:
            logger.warning(f"Could not clean up {temp_file}: {e}")
    temp_files_to_cleanup.clear()

atexit.register(cleanup_temp_files)


# ==================== VIDEO PREPROCESSING ====================
class VideoPreprocessor:
    """Advanced video preprocessing with optimal frame sampling strategies"""

    def __init__(self, target_fps: float = config.DEFAULT_FPS, 
                 max_frames: int = config.MAX_FRAMES_PER_CHUNK):
        self.target_fps = target_fps
        self.max_frames = max_frames
        logger.info(f"VideoPreprocessor initialized: FPS={target_fps}, max_frames={max_frames}")

    def extract_video_info(self, video_path: str) -> Dict[str, Any]:
        """Extract comprehensive video metadata with error handling"""
        cap = None
        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")

            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Validate video properties
            if fps <= 0:
                logger.warning(f"Invalid FPS ({fps}), using default 30")
                fps = 30.0

            if total_frames <= 0:
                logger.warning("Cannot determine total frames, will read until end")
                total_frames = -1

            duration = total_frames / fps if total_frames > 0 and fps > 0 else 0

            info = {
                'fps': fps,
                'total_frames': total_frames,
                'width': width,
                'height': height,
                'duration': duration,
                'path': video_path,
                'codec': int(cap.get(cv2.CAP_PROP_FOURCC))
            }

            logger.info(f"Video info: {duration:.2f}s, {total_frames} frames, {width}x{height}")
            return info

        except Exception as e:
            logger.error(f"Error extracting video info: {e}")
            raise
        finally:
            if cap is not None:
                cap.release()

    def uniform_fps_sampling(self, video_path: str, target_fps: Optional[float] = None) -> List[np.ndarray]:
        """
        Uniform FPS sampling - optimal for most video understanding tasks
        Based on: "Frame Sampling Strategies Matter" (2025)
        """
        if target_fps is None:
            target_fps = self.target_fps

        cap = None
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")

            original_fps = cap.get(cv2.CAP_PROP_FPS)
            if original_fps <= 0:
                original_fps = 30.0

            # Calculate frame interval for uniform sampling
            frame_interval = max(1, int(original_fps / target_fps))

            frames = []
            frame_idx = 0

            with tqdm(total=self.max_frames, desc="Sampling frames", leave=False) as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_idx % frame_interval == 0:
                        # Convert BGR to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        # Apply preprocessing
                        frame_processed = self.preprocess_frame(frame_rgb)
                        frames.append(frame_processed)
                        pbar.update(1)

                        if len(frames) >= self.max_frames:
                            break

                    frame_idx += 1

            # Ensure minimum frames
            if 0 < len(frames) < config.MIN_FRAMES:
                logger.warning(f"Only {len(frames)} frames sampled, duplicating to meet minimum")
                while len(frames) < config.MIN_FRAMES:
                    frames.append(frames[-1].copy())

            if len(frames) == 0:
                raise ValueError("No frames could be extracted from video")

            logger.info(f"Sampled {len(frames)} frames using uniform FPS sampling")
            return frames

        except Exception as e:
            logger.error(f"Error in uniform FPS sampling: {e}")
            raise
        finally:
            if cap is not None:
                cap.release()

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Advanced frame preprocessing for optimal model performance
        Includes: noise reduction, contrast enhancement, and normalization
        """
        try:
            # Validate input
            if frame is None or frame.size == 0:
                raise ValueError("Invalid frame")

            # Apply Gaussian blur for noise reduction
            frame = cv2.GaussianBlur(frame, (3, 3), 0)

            # Contrast enhancement using CLAHE
            lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            frame = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

            return frame

        except Exception as e:
            logger.warning(f"Frame preprocessing error: {e}, returning original")
            return frame

    def chunk_video(self, video_path: str) -> List[Tuple[List[np.ndarray], float, float]]:
        """
        Chunk long videos into manageable segments with temporal information
        Returns: List of (frames, start_time, end_time) tuples
        """
        cap = None
        try:
            video_info = self.extract_video_info(video_path)
            duration = video_info['duration']

            if duration <= 0:
                raise ValueError("Invalid video duration")

            # Calculate number of chunks needed
            frames_per_chunk = self.max_frames
            chunk_duration = frames_per_chunk / self.target_fps
            num_chunks = max(1, int(np.ceil(duration / chunk_duration)))

            logger.info(f"Splitting video into {num_chunks} chunks")

            chunks = []
            cap = cv2.VideoCapture(video_path)
            original_fps = video_info['fps']

            for chunk_idx in range(num_chunks):
                start_time = chunk_idx * chunk_duration
                end_time = min((chunk_idx + 1) * chunk_duration, duration)

                # Seek to start position
                start_frame = int(start_time * original_fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

                chunk_frames = []
                frame_interval = max(1, int(original_fps / self.target_fps))

                current_frame_idx = start_frame
                end_frame_idx = int(end_time * original_fps)

                while current_frame_idx < end_frame_idx and len(chunk_frames) < frames_per_chunk:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if (current_frame_idx - start_frame) % frame_interval == 0:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_processed = self.preprocess_frame(frame_rgb)
                        chunk_frames.append(frame_processed)

                    current_frame_idx += 1

                if chunk_frames:
                    chunks.append((chunk_frames, start_time, end_time))
                    logger.info(f"Chunk {chunk_idx + 1}/{num_chunks}: {len(chunk_frames)} frames, {start_time:.2f}s-{end_time:.2f}s")
                else:
                    logger.warning(f"Chunk {chunk_idx + 1} has no frames, skipping")

            if not chunks:
                raise ValueError("No valid chunks could be created from video")

            return chunks

        except Exception as e:
            logger.error(f"Error chunking video: {e}")
            raise
        finally:
            if cap is not None:
                cap.release()


# ==================== MODEL WRAPPER ====================
class KeyeVLModel:
    """Wrapper for Keye-VL-1.5-8B model with optimized inference"""

    def __init__(self, model_path: str = config.MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.device = config.DEVICE
        logger.info(f"Initializing Keye-VL model from {model_path}")
        self.load_model()

    def load_model(self):
        """Load model with optimal settings for RTX A4000"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"Model path does not exist: {self.model_path}\n"
                    f"Please download the model and update MODEL_PATH in the Config class."
                )

            logger.info("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                min_pixels=config.MIN_PIXELS,
                max_pixels=config.MAX_PIXELS
            )

            logger.info("Loading model (without Flash Attention)...")
            self.model = AutoModel.from_pretrained(
                self.model_path,
                torch_dtype=config.TORCH_DTYPE,
                device_map="auto",
                trust_remote_code=True,
                max_memory=config.MAX_MEMORY
            ).eval()

            logger.info(f"Model loaded successfully on {self.device}")
            logger.info(f"Model dtype: {self.model.dtype}")

            # Warm-up inference
            self._warmup()

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def _warmup(self):
        """Warm-up inference to initialize CUDA kernels"""
        try:
            logger.info("Performing warm-up inference...")
            dummy_messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"}
                ]
            }]
            _ = self.generate(dummy_messages, max_new_tokens=10)
            logger.info("Warm-up completed successfully")
        except Exception as e:
            logger.warning(f"Warm-up failed (this is usually ok): {e}")

    def generate(self, messages: List[Dict[str, Any]], max_new_tokens: int = config.MAX_NEW_TOKENS,
                 temperature: float = config.TEMPERATURE) -> str:
        """Generate text from messages with optimized parameters"""
        try:
            # Prepare input
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Process vision information
            image_inputs, video_inputs, mm_processor_kwargs = process_vision_info(messages)

            # Process inputs
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **mm_processor_kwargs
            )

            # Move to device
            inputs = inputs.to(self.device)

            # Generate with optimized parameters
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=config.TOP_P,
                    do_sample=config.DO_SAMPLE,
                    pad_token_id=self.processor.tokenizer.pad_token_id if hasattr(self.processor, 'tokenizer') else None
                )

            # Decode output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            # Clear cache
            if self.device == "cuda":
                torch.cuda.empty_cache()

            return output_text

        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise


# ==================== VIDEO ANALYZER ====================
class VideoAnalyzer:
    """High-level video analysis with chunking support"""

    def __init__(self, model: KeyeVLModel, preprocessor: VideoPreprocessor):
        self.model = model
        self.preprocessor = preprocessor
        logger.info("VideoAnalyzer initialized")

    def analyze_single_video(self, video_path: str, prompt: Optional[str] = None,
                            progress: Any = None) -> Dict[str, Any]:
        """
        Analyze a single video with optional custom prompt
        Handles videos of any length through intelligent chunking
        """
        try:
            if not video_path or not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")

            logger.info(f"Analyzing video: {video_path}")

            if progress is not None:
                progress(0, desc="Extracting video information...")

            # Extract video info
            video_info = self.preprocessor.extract_video_info(video_path)

            # Default prompt if none provided
            if not prompt or prompt.strip() == "":
                prompt = "Provide a comprehensive and detailed analysis of this video, including all visual elements, actions, scenes, objects, people, and any notable events or patterns."

            # Determine if chunking is needed
            estimated_frames = int(video_info['duration'] * self.preprocessor.target_fps)
            needs_chunking = estimated_frames > config.MAX_FRAMES_PER_CHUNK

            if needs_chunking:
                logger.info(f"Video needs chunking: {estimated_frames} estimated frames")
                result = self._analyze_chunked_video(video_path, prompt, video_info, progress)
            else:
                logger.info("Video can be analyzed in single pass")
                result = self._analyze_single_chunk(video_path, prompt, video_info, progress)

            # Save result
            self._save_result(result, video_path)

            return result

        except Exception as e:
            logger.error(f"Error analyzing video: {e}", exc_info=True)
            return {
                'status': 'error',
                'error': str(e),
                'video_path': video_path
            }

    def _analyze_single_chunk(self, video_path: str, prompt: str, 
                             video_info: Dict[str, Any], progress: Any) -> Dict[str, Any]:
        """Analyze video that fits in single chunk"""
        try:
            if progress is not None:
                progress(0.3, desc="Processing video frames...")

            # Create message for model
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "fps": self.preprocessor.target_fps,
                        "max_frames": config.MAX_FRAMES_PER_CHUNK
                    },
                    {"type": "text", "text": prompt}
                ]
            }]

            if progress is not None:
                progress(0.6, desc="Generating analysis...")

            analysis = self.model.generate(messages)

            if progress is not None:
                progress(1.0, desc="Complete!")

            return {
                'status': 'success',
                'video_path': video_path,
                'video_info': video_info,
                'prompt': prompt,
                'analysis': analysis,
                'chunks': 1,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error in single chunk analysis: {e}")
            raise

    def _analyze_chunked_video(self, video_path: str, prompt: str,
                               video_info: Dict[str, Any], progress: Any) -> Dict[str, Any]:
        """Analyze long video by splitting into chunks"""
        try:
            if progress is not None:
                progress(0.1, desc="Splitting video into chunks...")

            # Get chunks
            chunks = self.preprocessor.chunk_video(video_path)
            total_chunks = len(chunks)

            logger.info(f"Processing {total_chunks} chunks")

            chunk_analyses = []

            # Analyze each chunk
            for idx, (frames, start_time, end_time) in enumerate(chunks):
                if progress is not None:
                    progress(
                        0.1 + (0.8 * (idx / total_chunks)),
                        desc=f"Analyzing chunk {idx + 1}/{total_chunks} ({start_time:.1f}s-{end_time:.1f}s)..."
                    )

                # Create temporary video for chunk
                chunk_video_path = self._create_temp_video(frames, video_info['fps'])
                temp_files_to_cleanup.append(chunk_video_path)

                try:
                    chunk_prompt = f"{prompt}\n\nAnalyze this segment from {start_time:.1f}s to {end_time:.1f}s."

                    messages = [{
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": chunk_video_path,
                                "fps": self.preprocessor.target_fps,
                                "max_frames": len(frames)
                            },
                            {"type": "text", "text": chunk_prompt}
                        ]
                    }]

                    chunk_analysis = self.model.generate(messages)

                    chunk_analyses.append({
                        'chunk_id': idx + 1,
                        'start_time': start_time,
                        'end_time': end_time,
                        'num_frames': len(frames),
                        'analysis': chunk_analysis
                    })

                except Exception as chunk_error:
                    logger.error(f"Error processing chunk {idx + 1}: {chunk_error}")
                    chunk_analyses.append({
                        'chunk_id': idx + 1,
                        'start_time': start_time,
                        'end_time': end_time,
                        'num_frames': len(frames),
                        'analysis': f"Error processing this chunk: {str(chunk_error)}",
                        'error': True
                    })

            # Synthesize overall analysis
            if progress is not None:
                progress(0.9, desc="Synthesizing final analysis...")

            final_analysis = self._synthesize_analyses(chunk_analyses, prompt, video_info)

            if progress is not None:
                progress(1.0, desc="Complete!")

            return {
                'status': 'success',
                'video_path': video_path,
                'video_info': video_info,
                'prompt': prompt,
                'analysis': final_analysis,
                'chunk_analyses': chunk_analyses,
                'chunks': total_chunks,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error in chunked analysis: {e}")
            raise

    def _create_temp_video(self, frames: List[np.ndarray], fps: float) -> str:
        """Create temporary video file from frames"""
        try:
            temp_file = tempfile.NamedTemporaryFile(
                suffix='.mp4', 
                dir=config.TEMP_DIR, 
                delete=False
            )
            temp_path = temp_file.name
            temp_file.close()

            if not frames:
                raise ValueError("No frames provided for video creation")

            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

            if not out.isOpened():
                raise RuntimeError("Failed to create video writer")

            for frame in frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)

            out.release()

            # Verify file was created
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                raise RuntimeError("Failed to create temporary video file")

            return temp_path

        except Exception as e:
            logger.error(f"Error creating temporary video: {e}")
            raise

    def _synthesize_analyses(self, chunk_analyses: List[Dict[str, Any]], 
                            original_prompt: str, video_info: Dict[str, Any]) -> str:
        """Synthesize chunk analyses into comprehensive final analysis"""
        try:
            # Filter out error chunks
            valid_chunks = [c for c in chunk_analyses if not c.get('error', False)]

            if not valid_chunks:
                return "Error: All chunks failed to process. Please check the video file and try again."

            # Create synthesis prompt
            chunks_text = "\n\n".join([
                f"**Segment {chunk['chunk_id']} ({chunk['start_time']:.1f}s - {chunk['end_time']:.1f}s):**\n{chunk['analysis']}"
                for chunk in valid_chunks
            ])

            synthesis_prompt = f"""Based on the following analyses of different segments of a video (total duration: {video_info['duration']:.1f}s), provide a comprehensive, cohesive analysis of the entire video.

Original question: {original_prompt}

Segment analyses:
{chunks_text}

Provide a unified, detailed analysis that:
1. Summarizes the overall content and narrative
2. Identifies key themes and patterns across segments
3. Notes any important transitions or developments
4. Provides a cohesive understanding of the complete video

Do not simply concatenate the segments - synthesize them into a comprehensive whole."""

            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": synthesis_prompt}
                ]
            }]

            final_analysis = self.model.generate(messages, max_new_tokens=config.MAX_NEW_TOKENS)

            return final_analysis

        except Exception as e:
            logger.warning(f"Error synthesizing analyses: {e}")
            # Fallback: return concatenated analyses
            valid_chunks = [c for c in chunk_analyses if not c.get('error', False)]
            return "\n\n".join([
                f"**Segment {chunk['chunk_id']} ({chunk['start_time']:.1f}s - {chunk['end_time']:.1f}s):**\n{chunk['analysis']}"
                for chunk in valid_chunks
            ])

    def _save_result(self, result: Dict[str, Any], video_path: str):
        """Save analysis result to JSON file"""
        try:
            video_name = Path(video_path).stem
            output_path = os.path.join(
                config.OUTPUT_DIR,
                f"{video_name}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            logger.info(f"Result saved to {output_path}")

        except Exception as e:
            logger.error(f"Error saving result: {e}")

    def batch_search(self, folder_path: str, search_prompt: str,
                    progress: Any = None) -> Tuple[List[str], str]:
        """
        Search through multiple videos in a folder for specific content
        Returns: (list of matching video paths, detailed results text)
        """
        try:
            if not os.path.exists(folder_path):
                raise FileNotFoundError(f"Folder not found: {folder_path}")

            logger.info(f"Batch searching in folder: {folder_path}")
            logger.info(f"Search prompt: {search_prompt}")

            # Get all video files
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
            video_files = []

            for ext in video_extensions:
                video_files.extend(Path(folder_path).glob(f'*{ext}'))
                video_files.extend(Path(folder_path).glob(f'*{ext.upper()}'))

            video_files = sorted(list(set(video_files)))  # Remove duplicates and sort
            total_videos = len(video_files)

            logger.info(f"Found {total_videos} videos to search")

            if total_videos == 0:
                return [], "No video files found in the specified folder."

            matching_videos = []
            results_details = []

            # Process each video
            for idx, video_file in enumerate(video_files):
                video_path = str(video_file)

                if progress is not None:
                    progress(
                        idx / total_videos,
                        desc=f"Searching video {idx + 1}/{total_videos}: {video_file.name}"
                    )

                try:
                    # Analyze video with search prompt
                    search_query = f"Does this video contain the following: {search_prompt}? Provide a clear yes/no answer first, then explain what you see in detail."

                    result = self.analyze_single_video(
                        video_path,
                        search_query,
                        progress=None  # Disable nested progress
                    )

                    if result['status'] == 'success':
                        analysis = result['analysis']

                        # Check if it matches
                        is_match = self._check_match(analysis, search_prompt)

                        if is_match:
                            matching_videos.append(video_path)
                            results_details.append({
                                'video': video_file.name,
                                'path': video_path,
                                'analysis': analysis,
                                'match': True
                            })
                            logger.info(f"Match found: {video_file.name}")
                        else:
                            results_details.append({
                                'video': video_file.name,
                                'path': video_path,
                                'analysis': analysis,
                                'match': False
                            })
                    else:
                        results_details.append({
                            'video': video_file.name,
                            'path': video_path,
                            'error': result.get('error', 'Unknown error'),
                            'match': False
                        })

                except Exception as e:
                    logger.error(f"Error processing {video_file.name}: {e}")
                    results_details.append({
                        'video': video_file.name,
                        'path': video_path,
                        'error': str(e),
                        'match': False
                    })

            if progress is not None:
                progress(1.0, desc="Search complete!")

            # Format results text
            results_text = self._format_search_results(
                results_details, search_prompt, total_videos
            )

            return matching_videos, results_text

        except Exception as e:
            logger.error(f"Error in batch search: {e}")
            return [], f"Error during search: {str(e)}"

    def _check_match(self, analysis: str, search_prompt: str) -> bool:
        """Check if analysis matches search prompt"""
        analysis_lower = analysis.lower()

        # Check for explicit yes in first portion of response
        first_200 = analysis_lower[:200]

        # Look for affirmative responses
        affirmative = ['yes', 'contains', 'present', 'visible', 'shows', 'depicts', 'features']
        negative = ['no', 'does not', 'doesn\'t', 'absent', 'not visible', 'not present']

        affirmative_count = sum(1 for word in affirmative if word in first_200)
        negative_count = sum(1 for word in negative if word in first_200)

        if affirmative_count > negative_count:
            return True
        elif negative_count > affirmative_count:
            return False

        # Fallback to keyword matching
        search_keywords = search_prompt.lower().split()
        match_count = sum(1 for keyword in search_keywords if keyword in analysis_lower)

        return match_count / len(search_keywords) >= 0.5 if search_keywords else False

    def _format_search_results(self, results_details: List[Dict[str, Any]],
                               search_prompt: str, total_videos: int) -> str:
        """Format search results into readable text"""
        matches = [r for r in results_details if r.get('match', False)]

        output = f"""# Batch Search Results

**Search Query:** {search_prompt}
**Total Videos Searched:** {total_videos}
**Matches Found:** {len(matches)}

---

"""

        if matches:
            output += "## Matching Videos\n\n"
            for idx, match in enumerate(matches, 1):
                output += f"### {idx}. {match['video']}\n"
                output += f"**Path:** `{match['path']}`\n\n"
                output += f"**Analysis:**\n{match['analysis']}\n\n"
                output += "---\n\n"
        else:
            output += "No matching videos found.\n\n"

        # Add errors if any
        errors = [r for r in results_details if 'error' in r]
        if errors:
            output += "## Errors\n\n"
            for error in errors:
                output += f"- **{error['video']}**: {error['error']}\n"

        return output


# ==================== GRADIO INTERFACE ====================
class GradioInterface:
    """Professional Gradio interface for video analysis"""

    def __init__(self, analyzer: VideoAnalyzer):
        self.analyzer = analyzer
        logger.info("GradioInterface initialized")

    def create_interface(self) -> gr.Blocks:
        """Create comprehensive Gradio interface"""

        with gr.Blocks(
            title="Keye-VL Video Analysis System",
            theme=gr.themes.Soft(),
            css="""
                .header {text-align: center; margin-bottom: 20px;}
                .info-box {background-color: #f0f0f0; padding: 15px; border-radius: 5px; margin: 10px 0;}
                .result-box {border: 2px solid #4CAF50; padding: 15px; border-radius: 5px; margin: 10px 0;}
            """
        ) as interface:

            # Header
            gr.Markdown(
                """
                <div class="header">
                    <h1>ðŸŽ¬ Keye-VL Video Analysis System</h1>
                    <p>Professional-grade video understanding powered by Keye-VL-1.5-8B</p>
                    <p><em>Optimized for RTX A4000 â€¢ Supports videos of any length â€¢ Advanced preprocessing</em></p>
                </div>
                """
            )

            # Model info
            with gr.Accordion("â„¹ï¸ System Information", open=False):
                gr.Markdown(f"""
                <div class="info-box">
                <strong>Configuration:</strong><br>
                â€¢ Model Path: {config.MODEL_PATH}<br>
                â€¢ Device: {config.DEVICE}<br>
                â€¢ Max Frames per Chunk: {config.MAX_FRAMES_PER_CHUNK}<br>
                â€¢ Target FPS: {config.DEFAULT_FPS}<br>
                â€¢ Torch Dtype: {config.TORCH_DTYPE}<br>
                â€¢ Flash Attention: Disabled (not required)<br>
                </div>
                """)

            # Tabs
            with gr.Tabs():
                # ========== SINGLE VIDEO ANALYSIS TAB ==========
                with gr.Tab("ðŸ“¹ Single Video Analysis"):
                    gr.Markdown("""
                    ### Analyze Individual Videos
                    Upload a video and optionally provide a custom prompt for specific analysis.
                    The system automatically handles videos of any length through intelligent chunking.
                    """)

                    with gr.Row():
                        with gr.Column(scale=1):
                            video_input = gr.Video(
                                label="Upload Video"
                            )

                            prompt_input = gr.Textbox(
                                label="Custom Prompt (Optional)",
                                placeholder="Leave empty for default comprehensive analysis, or enter specific questions...",
                                lines=3
                            )

                            analyze_btn = gr.Button(
                                "ðŸ” Analyze Video",
                                variant="primary",
                                size="lg"
                            )

                            gr.Markdown("""
                            **Tips:**
                            - Supported formats: MP4, AVI, MOV, MKV, etc.
                            - Videos are automatically preprocessed for optimal results
                            - Long videos are intelligently chunked and synthesized
                            """)

                        with gr.Column(scale=1):
                            output_text = gr.Textbox(
                                label="Analysis Result",
                                lines=20,
                                show_copy_button=True
                            )

                            output_json = gr.JSON(
                                label="Detailed Results (JSON)",
                                visible=False
                            )

                            show_json_btn = gr.Button("Show Detailed JSON")

                    # Event handlers
                    analyze_btn.click(
                        fn=self.single_video_analysis,
                        inputs=[video_input, prompt_input],
                        outputs=[output_text, output_json]
                    )

                    show_json_btn.click(
                        fn=lambda: gr.update(visible=True),
                        outputs=output_json
                    )

                # ========== BATCH VIDEO SEARCH TAB ==========
                with gr.Tab("ðŸ”Ž Batch Video Search"):
                    gr.Markdown("""
                    ### Search Through Multiple Videos
                    Provide a folder path and search prompt to find videos containing specific content.
                    The system will analyze all videos and return matches.
                    """)

                    with gr.Row():
                        with gr.Column(scale=1):
                            folder_input = gr.Textbox(
                                label="Folder Path",
                                placeholder="/path/to/video/folder",
                                lines=1
                            )

                            search_prompt_input = gr.Textbox(
                                label="Search Prompt",
                                placeholder="Example: 'Search for videos containing a golden retriever'",
                                lines=3
                            )

                            search_btn = gr.Button(
                                "ðŸ” Search Videos",
                                variant="primary",
                                size="lg"
                            )

                            gr.Markdown("""
                            **Search Tips:**
                            - Be specific in your search prompt
                            - Searches all videos in the specified folder
                            - Results show matching videos with detailed analysis
                            - This may take a while for large folders
                            """)

                        with gr.Column(scale=1):
                            search_results_text = gr.Textbox(
                                label="Search Results",
                                lines=20,
                                show_copy_button=True
                            )

                    with gr.Row():
                        search_results_gallery = gr.Gallery(
                            label="Matching Videos (Thumbnails)",
                            columns=4,
                            height="auto"
                        )

                    # Event handlers
                    search_btn.click(
                        fn=self.batch_video_search,
                        inputs=[folder_input, search_prompt_input],
                        outputs=[search_results_gallery, search_results_text]
                    )

            # Footer
            gr.Markdown(f"""
            ---
            <div style="text-align: center; color: #666; font-size: 0.9em;">
                <p>Powered by Keye-VL-1.5-8B | Developed with advanced video preprocessing and optimization</p>
                <p>Â© 2025 Video Analysis System | Results saved to: {config.OUTPUT_DIR}</p>
            </div>
            """)

        return interface

    def single_video_analysis(self, video_path: Optional[str], prompt: Optional[str] = None,
                             progress=gr.Progress()) -> Tuple[str, Dict[str, Any]]:
        """Handler for single video analysis"""
        try:
            if not video_path:
                return "âš ï¸ Please upload a video first.", {}

            result = self.analyzer.analyze_single_video(video_path, prompt, progress)

            if result['status'] == 'success':
                # Format output text
                output = f"""# Video Analysis Results

**Video:** {Path(result['video_path']).name}
**Duration:** {result['video_info']['duration']:.2f} seconds
**Resolution:** {result['video_info']['width']}x{result['video_info']['height']}
**Chunks Processed:** {result['chunks']}
**Timestamp:** {result['timestamp']}

---

## Analysis

{result['analysis']}
"""

                if 'chunk_analyses' in result and len(result['chunk_analyses']) > 1:
                    output += "\n\n---\n\n## Segment Details\n\n"
                    for chunk in result['chunk_analyses']:
                        if not chunk.get('error', False):
                            output += f"**Segment {chunk['chunk_id']} ({chunk['start_time']:.1f}s - {chunk['end_time']:.1f}s)**\n"
                            output += f"{chunk['analysis']}\n\n"

                return output, result
            else:
                error_msg = f"âŒ Error: {result.get('error', 'Unknown error')}"
                return error_msg, result

        except Exception as e:
            logger.error(f"Error in single video analysis handler: {e}", exc_info=True)
            return f"âŒ Error: {str(e)}", {'status': 'error', 'error': str(e)}

    def batch_video_search(self, folder_path: str, search_prompt: str,
                          progress=gr.Progress()) -> Tuple[List[Tuple[np.ndarray, str]], str]:
        """Handler for batch video search"""
        try:
            if not folder_path:
                return [], "âš ï¸ Please provide a folder path."

            if not os.path.exists(folder_path):
                return [], f"âŒ Invalid folder path: {folder_path}"

            if not search_prompt:
                return [], "âš ï¸ Please provide a search prompt."

            matching_videos, results_text = self.analyzer.batch_search(
                folder_path, search_prompt, progress
            )

            # Create thumbnails for gallery
            gallery_items = []
            for video_path in matching_videos[:20]:  # Limit to 20 for display
                try:
                    cap = cv2.VideoCapture(video_path)
                    ret, frame = cap.read()
                    cap.release()

                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # Resize for display
                        height, width = frame_rgb.shape[:2]
                        max_size = 400
                        if max(height, width) > max_size:
                            scale = max_size / max(height, width)
                            new_width = int(width * scale)
                            new_height = int(height * scale)
                            frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))

                        gallery_items.append((frame_rgb, Path(video_path).name))
                except Exception as e:
                    logger.warning(f"Could not create thumbnail for {video_path}: {e}")

            return gallery_items, results_text

        except Exception as e:
            logger.error(f"Error in batch search handler: {e}", exc_info=True)
            return [], f"âŒ Error: {str(e)}"


# ==================== MAIN APPLICATION ====================
def main():
    """Main application entry point"""
    try:
        logger.info("=" * 60)
        logger.info("Starting Keye-VL Video Analysis System v2.2")
        logger.info("=" * 60)

        # Verify CUDA
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA version: {torch.version.cuda}")
        else:
            logger.warning("CUDA not available - running on CPU (will be slow)")

        # Verify model path
        if not os.path.exists(config.MODEL_PATH):
            raise ValueError(
                f"\n{'='*60}\n"
                f"Model path does not exist: {config.MODEL_PATH}\n\n"
                f"Please download the model first:\n"
                f"  huggingface-cli download Kwai-Keye/Keye-VL-1_5-8B --local-dir ./Keye-VL-1_5-8B\n\n"
                f"Then update MODEL_PATH in the Config class (line 62)\n"
                f"{'='*60}"
            )

        # Initialize components
        logger.info("Initializing components...")

        preprocessor = VideoPreprocessor(
            target_fps=config.DEFAULT_FPS,
            max_frames=config.MAX_FRAMES_PER_CHUNK
        )

        model = KeyeVLModel(config.MODEL_PATH)

        analyzer = VideoAnalyzer(model, preprocessor)

        interface = GradioInterface(analyzer)

        # Create and launch Gradio app
        logger.info("Creating Gradio interface...")
        app = interface.create_interface()

        logger.info("Launching application...")
        logger.info("=" * 60)

        # Launch with queue for progress tracking
        app.queue(max_size=10).launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
            inbrowser=True
        )

    except KeyboardInterrupt:
        logger.info("\nApplication stopped by user")
    except Exception as e:
        logger.error(f"\nFatal error: {e}", exc_info=True)
        print(f"\nâŒ ERROR: {e}\n")
        raise
    finally:
        # Cleanup
        logger.info("Cleaning up...")
        cleanup_temp_files()
        if config.DEVICE == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()