#!/usr/bin/env python3
"""
Professional Gemma 3n-E4B-it Video + Audio Analysis System
===========================================================
A comprehensive offline multimodal system using Gemma 3n-E4B-it model.
Designed for RTX A4000 (16GB VRAM) running on Windows.

Features:
- Video frame analysis with custom prompts
- Audio transcription and understanding
- Combined video + audio multimodal analysis
- Batch processing with intelligent filtering
- Professional Gradio interface
- Local model loading (offline capable)
- GPU optimization for RTX A4000

Based on: google/gemma-3n-e4b-it
Requirements: transformers >= 4.53.0

Author: AI Assistant
Version: 3.2 (Video + Audio Analysis - Fixed device mismatch)
"""

import os
import sys
import warnings
import logging
from typing import List, Tuple, Dict, Any, Optional, Union
import json
from datetime import datetime
import glob
from pathlib import Path
import re
import shutil
import tempfile

# Core libraries with robust error handling
import numpy as np

# Audio processing libraries
try:
    import librosa
    import soundfile as sf
except ImportError:
    print("❌ Error: librosa and soundfile are required for audio processing.")
    print("Install with: pip install librosa soundfile")
    sys.exit(1)

# MoviePy for audio extraction from video (v2.0+ syntax)
try:
    from moviepy import VideoFileClip  # MoviePy 2.0+ syntax
except ImportError:
    print("❌ Error: moviepy is required for audio extraction from video.")
    print("Install with: pip install moviepy")
    sys.exit(1)

# OpenCV for video processing
try:
    import cv2
except ImportError:
    print("❌ Error: OpenCV is required for video processing. Install with: pip install opencv-python")
    sys.exit(1)

# PIL for image processing
try:
    from PIL import Image, ImageEnhance
except ImportError:
    print("❌ Error: Pillow (PIL) is required. Install with: pip install Pillow")
    sys.exit(1)

# PyTorch with version check
try:
    import torch
    if torch.__version__ < "2.0.0":
        print(f"Warning: PyTorch {torch.__version__} detected. Recommended: >= 2.0.0")
except ImportError:
    print("❌ Error: PyTorch is required. Install with: pip install torch")
    sys.exit(1)

# Transformers with correct Gemma 3n imports
try:
    from transformers import (
        AutoProcessor,
        Gemma3nForConditionalGeneration,
        __version__ as transformers_version
    )
    
    # Critical version check
    if transformers_version < "4.53.0":
        print(f"❌ Error: transformers version {transformers_version} is too old!")
        print("Gemma 3n requires transformers >= 4.53.0")
        print("Install with: pip install 'transformers>=4.53.0'")
        sys.exit(1)
except ImportError as e:
    print(f"❌ Error importing transformers: {e}")
    print("Please install: pip install 'transformers>=4.53.0'")
    sys.exit(1)

# Gradio with version check
try:
    import gradio as gr
    gradio_version = getattr(gr, '__version__', '0.0.0')
    print(f"✅ Gradio version: {gradio_version}")
except ImportError:
    print("❌ Error: Gradio not installed. Please install: pip install gradio")
    sys.exit(1)

# Optional tqdm for progress bars
try:
    from tqdm import tqdm
except ImportError:
    print("Warning: tqdm not available. Progress bars will be disabled.")
    class tqdm:
        def __init__(self, iterable, desc="", **kwargs):
            self.iterable = iterable
        def __iter__(self):
            return iter(self.iterable)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Config:
    """Configuration class for the Gemma 3n-E4B-it video+audio system"""
    
    # Model settings
    MODEL_NAME = "google/gemma-3n-e4b-it"
    MODEL_PATH = os.path.join("models", "gemma-3n-e4b-it")
    
    # System settings
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    MAX_BATCH_SIZE = 1
    
    # Video settings
    SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', 
                               '.MP4', '.AVI', '.MOV', '.MKV', '.WMV', '.FLV', '.WEBM']
    
    # Frame extraction settings
    DEFAULT_FPS = 1  # Extract 1 frame per second by default
    MAX_FRAMES = 30  # Maximum frames to extract per video
    FRAME_SAMPLE_METHODS = ['uniform', 'fps', 'keyframe']
    
    # Audio settings (Gemma 3n specifications)
    AUDIO_SAMPLE_RATE = 16000  # 16 kHz as required by Gemma 3n
    AUDIO_MAX_DURATION = 30.0  # 30 seconds max per chunk
    AUDIO_CHANNELS = 1  # Mono audio required
    AUDIO_DTYPE = np.float32  # Float32 format
    AUDIO_NORMALIZATION_RANGE = (-1.0, 1.0)  # Audio range [-1, 1]
    
    # Image settings (validated for Gemma 3n)
    SUPPORTED_IMAGE_SIZES = [256, 512, 768]
    DEFAULT_IMAGE_SIZE = 512
    
    # Generation settings
    MAX_NEW_TOKENS = 512  # Increased for audio+video analysis
    TEMPERATURE = 0.7
    DO_SAMPLE = True
    TOP_P = 0.9
    TOP_K = 40
    REPETITION_PENALTY = 1.1
    
    # UI settings
    INTERFACE_TITLE = "🎬🎤 Gemma 3n-E4B-it Video + Audio Analysis System"
    INTERFACE_DESCRIPTION = """
## Advanced Multimodal Analysis with Gemma 3n-E4B-it

Upload videos and analyze both **visual content** and **audio/speech** using Google's Gemma 3n-E4B-it model.

**Multimodal Capabilities:**
- 🎥 Video frame analysis (temporal understanding)
- 🎤 Audio transcription (100+ languages)
- 🔊 Speech understanding and context extraction
- 🎯 Combined video-audio intelligence
- 📍 Named entity recognition in audio (places, names, events)
- 🌍 Multilingual support

**Model Features:**
- Gemma 3n-E4B-it (4B effective parameters, 8B total)
- MatFormer architecture with selective parameter activation
- Universal Speech Model (USM) audio encoder
- 32K context window
"""


class AudioProcessor:
    """Professional audio processing for Gemma 3n with comprehensive error handling"""
    
    def __init__(self):
        self.sample_rate = Config.AUDIO_SAMPLE_RATE
        self.max_duration = Config.AUDIO_MAX_DURATION
        self.channels = Config.AUDIO_CHANNELS
        self.temp_dir = tempfile.mkdtemp(prefix="gemma_audio_")
    
    def __del__(self):
        """Cleanup temporary directory"""
        try:
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)
        except:
            pass
    
    def extract_audio_from_video(self, video_path: str) -> Optional[str]:
        """
        Extract audio from video file
        Returns path to extracted audio file (WAV format)
        """
        try:
            logging.info(f"Extracting audio from: {video_path}")
            
            # Load video
            video = VideoFileClip(video_path)
            
            # Check if video has audio
            if video.audio is None:
                logging.warning(f"No audio track found in video: {video_path}")
                video.close()
                return None
            
            # Create temporary audio file
            audio_filename = f"audio_{os.path.basename(video_path)}.wav"
            audio_path = os.path.join(self.temp_dir, audio_filename)
            
            # Extract and save audio
            video.audio.write_audiofile(
                audio_path,
                codec='pcm_s16le',
                fps=44100,  # Initial extraction at 44.1kHz
                nbytes=2,
                logger=None  # Suppress moviepy logging
            )
            
            video.close()
            logging.info(f"Audio extracted successfully to: {audio_path}")
            return audio_path
        
        except Exception as e:
            logging.error(f"Error extracting audio from video: {e}")
            return None
    
    def preprocess_audio_for_gemma(self, audio_path: str) -> Optional[np.ndarray]:
        """
        Preprocess audio according to Gemma 3n specifications:
        - Mono channel
        - 16 kHz sample rate
        - Float32 format
        - Range: [-1, 1]
        - Max duration: 30 seconds
        
        Returns: numpy array of shape (samples,) with float32 dtype
        """
        try:
            logging.info(f"Preprocessing audio: {audio_path}")
            
            # Load audio with librosa (automatically converts to mono and resamples)
            audio_array, sr = librosa.load(
                audio_path,
                sr=self.sample_rate,  # Resample to 16kHz
                mono=True,  # Convert to mono
                dtype=np.float32  # Float32 format
            )
            
            # Ensure audio is in range [-1, 1]
            if audio_array.max() > 1.0 or audio_array.min() < -1.0:
                # Normalize to [-1, 1]
                max_val = max(abs(audio_array.max()), abs(audio_array.min()))
                if max_val > 0:
                    audio_array = audio_array / max_val
            
            logging.info(f"Audio preprocessed: shape={audio_array.shape}, "
                        f"sr={sr}, dtype={audio_array.dtype}, "
                        f"duration={len(audio_array)/sr:.2f}s")
            
            return audio_array
        
        except Exception as e:
            logging.error(f"Error preprocessing audio: {e}")
            return None
    
    def split_audio_chunks(self, audio_array: np.ndarray, chunk_duration: float = None) -> List[np.ndarray]:
        """
        Split long audio into chunks (max 30 seconds per chunk for Gemma 3n)
        
        Args:
            audio_array: Input audio array
            chunk_duration: Duration of each chunk in seconds (default: 30s)
        
        Returns: List of audio chunks
        """
        try:
            if chunk_duration is None:
                chunk_duration = self.max_duration
            
            chunk_samples = int(chunk_duration * self.sample_rate)
            total_samples = len(audio_array)
            
            # If audio is shorter than chunk duration, return as single chunk
            if total_samples <= chunk_samples:
                return [audio_array]
            
            # Split into chunks
            chunks = []
            for start_idx in range(0, total_samples, chunk_samples):
                end_idx = min(start_idx + chunk_samples, total_samples)
                chunk = audio_array[start_idx:end_idx]
                chunks.append(chunk)
            
            logging.info(f"Split audio into {len(chunks)} chunks of ~{chunk_duration}s each")
            return chunks
        
        except Exception as e:
            logging.error(f"Error splitting audio: {e}")
            return [audio_array]
    
    def get_audio_duration(self, audio_array: np.ndarray) -> float:
        """Calculate audio duration in seconds"""
        return len(audio_array) / self.sample_rate


class VideoProcessor:
    """Robust video processing with comprehensive error handling"""
    
    def __init__(self, target_size: int = 512):
        self.target_size = target_size
        self.supported_sizes = Config.SUPPORTED_IMAGE_SIZES
        self.temp_frame_dir = "temp_frames"
    
    def is_valid_video_file(self, file_path: str) -> bool:
        """Check if file is a valid video file"""
        try:
            _, ext = os.path.splitext(file_path.lower())
            if ext not in [fmt.lower() for fmt in Config.SUPPORTED_VIDEO_FORMATS]:
                return False
            
            if not os.path.isfile(file_path):
                return False
            
            # Try to open the video to verify it's valid
            cap = cv2.VideoCapture(file_path)
            is_valid = cap.isOpened()
            cap.release()
            return is_valid
        except Exception:
            return False
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Extract video metadata"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {}
            
            info = {
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'duration': 0
            }
            
            if info['fps'] > 0:
                info['duration'] = info['frame_count'] / info['fps']
            
            cap.release()
            return info
        except Exception as e:
            logging.error(f"Error getting video info: {e}")
            return {}
    
    def optimize_frame_size(self, frame: np.ndarray) -> Image.Image:
        """Optimize frame size for Gemma 3n"""
        try:
            # Convert BGR to RGB
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
            
            # Convert to PIL Image
            image = Image.fromarray(frame_rgb)
            
            # Ensure RGB mode
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize maintaining aspect ratio
            return self.resize_and_pad(image, self.target_size)
        except Exception as e:
            logging.error(f"Error optimizing frame size: {e}")
            return Image.new('RGB', (512, 512), (255, 255, 255))
    
    def resize_and_pad(self, image: Image.Image, target_size: int) -> Image.Image:
        """Resize image maintaining aspect ratio and pad to square"""
        try:
            width, height = image.size
            
            if width > height:
                new_width = target_size
                new_height = int((height * target_size) / width)
            else:
                new_height = target_size
                new_width = int((width * target_size) / height)
            
            new_width = max(1, new_width)
            new_height = max(1, new_height)
            
            image = image.resize((new_width, new_height), Image.LANCZOS)
            
            canvas = Image.new('RGB', (target_size, target_size), (255, 255, 255))
            x_offset = (target_size - new_width) // 2
            y_offset = (target_size - new_height) // 2
            
            x_offset = max(0, x_offset)
            y_offset = max(0, y_offset)
            
            canvas.paste(image, (x_offset, y_offset))
            return canvas
        except Exception as e:
            logging.error(f"Error resizing and padding: {e}")
            return image
    
    def extract_frames_uniform(self, video_path: str, num_frames: int = 10) -> List[Tuple[Image.Image, float]]:
        """Extract frames uniformly distributed across the video"""
        frames = []
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logging.error(f"Could not open video: {video_path}")
                return frames
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if total_frames == 0:
                cap.release()
                return frames
            
            # Calculate frame indices to extract
            frame_indices = np.linspace(0, total_frames - 1, min(num_frames, total_frames), dtype=int)
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if ret:
                    timestamp = idx / fps if fps > 0 else 0
                    optimized_frame = self.optimize_frame_size(frame)
                    frames.append((optimized_frame, timestamp))
            
            cap.release()
            logging.info(f"Extracted {len(frames)} frames from {video_path}")
        except Exception as e:
            logging.error(f"Error extracting frames: {e}")
        
        return frames
    
    def extract_frames_by_fps(self, video_path: str, target_fps: float = 1.0, max_frames: int = 30) -> List[Tuple[Image.Image, float]]:
        """Extract frames at specified FPS rate"""
        frames = []
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logging.error(f"Could not open video: {video_path}")
                return frames
            
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            if video_fps == 0:
                cap.release()
                return frames
            
            # Calculate frame skip interval
            frame_skip = int(video_fps / target_fps)
            frame_skip = max(1, frame_skip)
            
            frame_count = 0
            extracted_count = 0
            
            while extracted_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_skip == 0:
                    timestamp = frame_count / video_fps
                    optimized_frame = self.optimize_frame_size(frame)
                    frames.append((optimized_frame, timestamp))
                    extracted_count += 1
                
                frame_count += 1
            
            cap.release()
            logging.info(f"Extracted {len(frames)} frames at {target_fps} FPS from {video_path}")
        except Exception as e:
            logging.error(f"Error extracting frames by FPS: {e}")
        
        return frames
    
    def extract_key_frames(self, video_path: str, threshold: float = 30.0, max_frames: int = 30) -> List[Tuple[Image.Image, float]]:
        """Extract key frames based on scene changes"""
        frames = []
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isopened():
                logging.error(f"Could not open video: {video_path}")
                return frames
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            prev_frame = None
            frame_count = 0
            extracted_count = 0
            
            while extracted_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale for comparison
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # First frame is always a key frame
                if prev_frame is None:
                    timestamp = frame_count / fps if fps > 0 else 0
                    optimized_frame = self.optimize_frame_size(frame)
                    frames.append((optimized_frame, timestamp))
                    extracted_count += 1
                    prev_frame = gray
                else:
                    # Calculate difference between frames
                    frame_diff = cv2.absdiff(prev_frame, gray)
                    mean_diff = np.mean(frame_diff)
                    
                    # If difference exceeds threshold, it's a key frame
                    if mean_diff > threshold:
                        timestamp = frame_count / fps if fps > 0 else 0
                        optimized_frame = self.optimize_frame_size(frame)
                        frames.append((optimized_frame, timestamp))
                        extracted_count += 1
                        prev_frame = gray
                
                frame_count += 1
            
            cap.release()
            logging.info(f"Extracted {len(frames)} key frames from {video_path}")
        except Exception as e:
            logging.error(f"Error extracting key frames: {e}")
        
        return frames
    
    def load_videos_from_directory(self, directory_path: str) -> List[str]:
        """Load all supported videos from directory"""
        videos = []
        try:
            directory = Path(directory_path)
            if not directory.exists() or not directory.is_dir():
                logging.error(f"Invalid directory: {directory_path}")
                return videos
            
            for ext in Config.SUPPORTED_VIDEO_FORMATS:
                pattern = f"*{ext}"
                matching_files = list(directory.glob(pattern))
                videos.extend(matching_files)
            
            videos = sorted(list(set(videos)))
            videos = [str(v) for v in videos if self.is_valid_video_file(str(v))]
            
            logging.info(f"Found {len(videos)} valid videos in {directory_path}")
        except Exception as e:
            logging.error(f"Error loading videos from directory: {e}")
        
        return videos
    
    def create_frame_grid(self, frames: List[Image.Image], max_cols: int = 5) -> Optional[Image.Image]:
        """Create a grid of frames for display"""
        try:
            if not frames:
                return None
            
            num_frames = len(frames)
            cols = min(max_cols, num_frames)
            rows = (num_frames + cols - 1) // cols
            
            cell_size = 160
            grid_width = cols * cell_size
            grid_height = rows * cell_size
            grid_image = Image.new('RGB', (grid_width, grid_height), (240, 240, 240))
            
            for idx, frame in enumerate(frames):
                try:
                    row = idx // cols
                    col = idx % cols
                    x = col * cell_size
                    y = row * cell_size
                    
                    resized_frame = frame.resize((cell_size - 4, cell_size - 4), Image.LANCZOS)
                    grid_image.paste(resized_frame, (x + 2, y + 2))
                except Exception as e:
                    logging.warning(f"Error placing frame {idx} in grid: {e}")
                    continue
            
            return grid_image
        except Exception as e:
            logging.error(f"Error creating frame grid: {e}")
            return None


class Gemma3nMultimodalModel:
    """Gemma 3n-E4B-it model wrapper for video + audio analysis"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.processor = None
        self.model_loaded = False
    
    def load_model(self):
        """Load Gemma 3n-E4B-it model"""
        logging.info("Loading Gemma 3n-E4B-it model...")
        try:
            model_path = self.config.MODEL_PATH if os.path.exists(self.config.MODEL_PATH) else self.config.MODEL_NAME
            local_files_only = os.path.exists(self.config.MODEL_PATH)
            
            logging.info("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                local_files_only=local_files_only,
                trust_remote_code=True
            )
            logging.info("✅ Processor loaded successfully")
            
            logging.info("Loading model...")
            self.model = Gemma3nForConditionalGeneration.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=self.config.TORCH_DTYPE,
                local_files_only=local_files_only,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            ).eval()
            logging.info("✅ Model loaded successfully")
            
            self.model_loaded = True
            
            if self.config.DEVICE == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logging.info(f"GPU Memory: {allocated:.2f}GB / {total:.1f}GB")
            
            logging.info("🎉 Gemma 3n-E4B-it model loaded successfully!")
        except Exception as e:
            logging.error(f"❌ Critical error loading model: {e}")
            self.model_loaded = False
            raise
    
    def create_multimodal_messages(self, text_prompt: str, 
                                   frames: Optional[List[Image.Image]] = None,
                                   audio: Optional[np.ndarray] = None) -> List[Dict]:
        """
        Create properly formatted chat messages for Gemma 3n with video frames and/or audio
        
        Args:
            text_prompt: Text query/prompt
            frames: List of PIL Image frames (optional)
            audio: Audio array in Gemma 3n format (optional)
        
        Returns: Formatted messages list
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": [{
                        "type": "text",
                        "text": "You are a helpful AI assistant specialized in analyzing and understanding videos, audio, and multimodal content."
                    }]
                }
            ]
            
            user_content = []
            
            # Add video frames if provided
            if frames:
                for frame in frames:
                    if isinstance(frame, Image.Image):
                        user_content.append({"type": "image", "image": frame})
            
            # Add audio if provided
            if audio is not None and len(audio) > 0:
                user_content.append({"type": "audio", "audio": audio})
            
            # Add text prompt
            user_content.append({"type": "text", "text": text_prompt})
            
            messages.append({
                "role": "user",
                "content": user_content
            })
            
            return messages
        except Exception as e:
            logging.error(f"Error creating multimodal messages: {e}")
            return []
    
    def _move_inputs_to_device(self, inputs: Union[Dict, Any], device: torch.device, dtype: torch.dtype) -> Union[Dict, Any]:
        """
        Recursively move all tensors in inputs to the specified device and dtype.
        This fixes the "Expected all tensors to be on the same device" error.
        
        Args:
            inputs: Input dict or tensor
            device: Target device (e.g., cuda:0)
            dtype: Target dtype (e.g., torch.bfloat16)
        
        Returns: Inputs with all tensors moved to target device/dtype
        """
        if isinstance(inputs, dict):
            return {
                key: self._move_inputs_to_device(value, device, dtype)
                for key, value in inputs.items()
            }
        elif isinstance(inputs, (list, tuple)):
            return type(inputs)(self._move_inputs_to_device(item, device, dtype) for item in inputs)
        elif isinstance(inputs, torch.Tensor):
            # Move tensor to device and convert dtype if it's a floating point tensor
            if inputs.dtype.is_floating_point:
                return inputs.to(device=device, dtype=dtype)
            else:
                # For non-floating point tensors (e.g., int64), only move device
                return inputs.to(device=device)
        else:
            return inputs
    
    def analyze_multimodal(self, 
                          frames: Optional[List[Image.Image]] = None,
                          audio: Optional[np.ndarray] = None,
                          prompt: str = "Analyze this content.") -> str:
        """
        Analyze multimodal content (video frames + audio)
        
        Args:
            frames: List of video frames (PIL Images)
            audio: Audio array (preprocessed for Gemma 3n)
            prompt: Analysis prompt
        
        Returns: Analysis text
        """
        if not self.model_loaded:
            return "❌ Error: Model not loaded properly"
        
        try:
            if not frames and audio is None:
                return "❌ Error: No frames or audio provided"
            
            # Limit frames to prevent memory issues
            max_frames_per_analysis = 8 if audio is not None else 10
            if frames and len(frames) > max_frames_per_analysis:
                # Sample frames uniformly
                indices = np.linspace(0, len(frames) - 1, max_frames_per_analysis, dtype=int)
                sampled_frames = [frames[i] for i in indices]
            else:
                sampled_frames = frames
            
            messages = self.create_multimodal_messages(prompt, sampled_frames, audio)
            if not messages:
                return "❌ Error: Failed to create multimodal messages"
            
            try:
                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                
                # CRITICAL FIX: Move ALL inputs to model's device and dtype
                # This fixes "Expected all tensors to be on the same device" error
                device = self.model.device
                dtype = self.config.TORCH_DTYPE
                
                logging.debug(f"Model device: {device}, dtype: {dtype}")
                logging.debug(f"Input keys before move: {inputs.keys() if isinstance(inputs, dict) else 'not a dict'}")
                
                # Move all inputs to the correct device and dtype
                inputs = self._move_inputs_to_device(inputs, device, dtype)
                
                # Log device placement for debugging
                if isinstance(inputs, dict):
                    for key, value in inputs.items():
                        if isinstance(value, torch.Tensor):
                            logging.debug(f"{key}: device={value.device}, dtype={value.dtype}, shape={value.shape}")
                
                input_len = inputs["input_ids"].shape[-1]
                
                with torch.inference_mode():
                    generation = self.model.generate(
                        **inputs,
                        max_new_tokens=self.config.MAX_NEW_TOKENS,
                        temperature=self.config.TEMPERATURE,
                        do_sample=self.config.DO_SAMPLE,
                        top_p=self.config.TOP_P,
                        top_k=self.config.TOP_K,
                        repetition_penalty=self.config.REPETITION_PENALTY,
                        pad_token_id=self.processor.tokenizer.eos_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id
                    )
                
                if generation.dim() > 1:
                    generation = generation[0][input_len:]
                else:
                    generation = generation[input_len:]
                
                decoded = self.processor.decode(generation, skip_special_tokens=True)
                response = decoded.strip()
                
                if not response:
                    return "No response generated. Please try a different prompt."
                
                return response
            
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                return "❌ Error: GPU out of memory. Try reducing the number of frames or using shorter audio."
            except Exception as e:
                logging.error(f"Error during model inference: {e}")
                import traceback
                traceback.print_exc()
                return f"❌ Error during analysis: {str(e)}"
        
        except Exception as e:
            logging.error(f"Error analyzing multimodal content: {e}")
            import traceback
            traceback.print_exc()
            return f"❌ Error analyzing content: {str(e)}"
    
    def transcribe_audio(self, audio: np.ndarray) -> str:
        """
        Transcribe audio to text
        
        Args:
            audio: Audio array (preprocessed for Gemma 3n)
        
        Returns: Transcribed text
        """
        if not self.model_loaded:
            return "❌ Error: Model not loaded properly"
        
        try:
            prompt = "Transcribe the speech in this audio."
            return self.analyze_multimodal(frames=None, audio=audio, prompt=prompt)
        except Exception as e:
            logging.error(f"Error transcribing audio: {e}")
            return f"❌ Error transcribing audio: {str(e)}"
    
    def analyze_audio_content(self, audio: np.ndarray, focus_query: str = None) -> str:
        """
        Analyze audio content for specific information
        
        Args:
            audio: Audio array (preprocessed for Gemma 3n)
            focus_query: Specific query about audio content
        
        Returns: Analysis text
        """
        if not self.model_loaded:
            return "❌ Error: Model not loaded properly"
        
        try:
            if focus_query:
                prompt = f"Listen to this audio and answer: {focus_query}"
            else:
                prompt = """Analyze this audio comprehensively. Provide:
1. Transcription of speech
2. Identification of any names, places, organizations, or events mentioned
3. Overall tone and context
4. Key topics or themes discussed
5. Any notable sounds or background audio"""
            
            return self.analyze_multimodal(frames=None, audio=audio, prompt=prompt)
        except Exception as e:
            logging.error(f"Error analyzing audio content: {e}")
            return f"❌ Error analyzing audio: {str(e)}"


class MultimodalAnalyzer:
    """Main video + audio analyzer"""
    
    def __init__(self):
        self.config = Config()
        self.video_processor = VideoProcessor(self.config.DEFAULT_IMAGE_SIZE)
        self.audio_processor = AudioProcessor()
        self.model = Gemma3nMultimodalModel(self.config)
        self.system_ready = False
        self.error_message = ""
        self.initialize_system()
    
    def initialize_system(self):
        """Initialize system"""
        logging.info("🚀 Initializing Gemma 3n-E4B-it Multimodal Analysis System...")
        try:
            logging.info(f"Device: {self.config.DEVICE}")
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name()
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logging.info(f"GPU: {gpu_name}")
                logging.info(f"VRAM: {total_memory:.1f} GB")
            else:
                logging.warning("CUDA not available - using CPU")
            
            self.model.load_model()
            self.system_ready = True
            logging.info("✅ System initialized successfully!")
        except Exception as e:
            self.error_message = str(e)
            logging.error(f"❌ Failed to initialize system: {e}")
            self.system_ready = False
    
    def get_system_status(self) -> str:
        """Get system status"""
        if self.system_ready:
            return "✅ System Ready - Gemma 3n-E4B-it Loaded (Video + Audio)"
        else:
            return f"❌ System Error: {self.error_message}"
    
    def analyze_single_video(self, video_path: str, prompt: str = None, 
                            extraction_method: str = 'uniform', 
                            num_frames: int = 10,
                            analyze_audio: bool = True) -> Tuple[Optional[Image.Image], str]:
        """
        Analyze single video with optional audio analysis
        
        Args:
            video_path: Path to video file
            prompt: Custom analysis prompt
            extraction_method: Frame extraction method
            num_frames: Number of frames to extract
            analyze_audio: Whether to analyze audio track
        
        Returns: (frame_grid, analysis_text)
        """
        if not self.system_ready:
            return None, f"❌ System not ready: {self.error_message}"
        
        try:
            if not video_path or not os.path.exists(video_path):
                return None, "❌ Invalid video path provided"
            
            if not self.video_processor.is_valid_video_file(video_path):
                return None, f"❌ Invalid video file: {video_path}"
            
            # Get video info
            video_info = self.video_processor.get_video_info(video_path)
            
            # Extract frames
            if extraction_method == 'uniform':
                frames_with_timestamps = self.video_processor.extract_frames_uniform(video_path, num_frames)
            elif extraction_method == 'fps':
                frames_with_timestamps = self.video_processor.extract_frames_by_fps(
                    video_path, Config.DEFAULT_FPS, num_frames
                )
            elif extraction_method == 'keyframe':
                frames_with_timestamps = self.video_processor.extract_key_frames(video_path, max_frames=num_frames)
            else:
                frames_with_timestamps = self.video_processor.extract_frames_uniform(video_path, num_frames)
            
            if not frames_with_timestamps:
                return None, "❌ Failed to extract frames from video"
            
            frames = [f[0] for f in frames_with_timestamps]
            timestamps = [f[1] for f in frames_with_timestamps]
            
            # Audio analysis
            audio_analysis = None
            audio_transcription = None
            audio_duration = 0
            has_audio = False
            audio_chunks = []
            
            if analyze_audio:
                logging.info("🎤 Processing audio track...")
                audio_path = self.audio_processor.extract_audio_from_video(video_path)
                
                if audio_path:
                    has_audio = True
                    audio_array = self.audio_processor.preprocess_audio_for_gemma(audio_path)
                    
                    if audio_array is not None:
                        audio_duration = self.audio_processor.get_audio_duration(audio_array)
                        
                        # Split audio if longer than 30 seconds
                        audio_chunks = self.audio_processor.split_audio_chunks(audio_array)
                        
                        # Analyze first chunk (or combine if needed)
                        if len(audio_chunks) > 0:
                            try:
                                # For comprehensive analysis, use first chunk
                                audio_analysis = self.model.analyze_audio_content(audio_chunks[0])
                                
                                # If multiple chunks, also get transcription
                                if len(audio_chunks) > 1:
                                    logging.info(f"Processing {len(audio_chunks)} audio chunks...")
                                    transcriptions = []
                                    for i, chunk in enumerate(audio_chunks[:3]):  # Limit to first 3 chunks
                                        trans = self.model.transcribe_audio(chunk)
                                        if not trans.startswith("❌"):
                                            transcriptions.append(f"[Segment {i+1}] {trans}")
                                    
                                    if transcriptions:
                                        audio_transcription = "

".join(transcriptions)
                            except Exception as e:
                                logging.error(f"Error processing audio chunks: {e}")
                                audio_analysis = f"❌ Error processing audio: {str(e)}"
            
            # Video analysis with or without audio context
            if prompt is None or not prompt.strip():
                if analyze_audio and has_audio:
                    prompt = """Analyze this video comprehensively considering both visual and audio content.

**Visual Analysis:**
- Describe the main scenes, activities, objects, and people
- Identify the setting and environment
- Note any significant visual changes or events

**Audio Analysis (if available):**
- Summarize what is being said
- Identify any names of people, places, or organizations mentioned
- Note key topics, themes, or events discussed in the audio

**Combined Insights:**
- Draw connections between visual and audio content
- Provide an overall summary and key findings"""
                else:
                    prompt = "Analyze this video. Describe the main content, activities, objects, people, setting, and any notable events or changes throughout the video."
            
            # Perform analysis
            if analyze_audio and has_audio and len(audio_chunks) > 0:
                # Use first audio chunk for combined analysis
                audio_chunk = audio_chunks[0]
                visual_audio_analysis = self.model.analyze_multimodal(frames, audio_chunk, prompt)
            else:
                visual_audio_analysis = self.model.analyze_multimodal(frames, None, prompt)
            
            # Create frame grid
            frame_grid = self.video_processor.create_frame_grid(frames)
            
            # Format result
            result_text = f"# 🎬 Multimodal Video Analysis Results

"
            result_text += f"**Video:** `{os.path.basename(video_path)}`
"
            result_text += f"**Analysis Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"
            result_text += f"**Model:** Gemma 3n-E4B-it (Multimodal)

"
            
            if video_info:
                result_text += f"### 📊 Video Information

"
                result_text += f"- **Duration:** {video_info.get('duration', 0):.2f} seconds
"
                result_text += f"- **FPS:** {video_info.get('fps', 0):.2f}
"
                result_text += f"- **Resolution:** {video_info.get('width', 0)}x{video_info.get('height', 0)}
"
                result_text += f"- **Total Frames:** {video_info.get('frame_count', 0)}
"
                if has_audio:
                    result_text += f"- **Audio Track:** ✅ Present ({audio_duration:.2f}s)
"
                else:
                    result_text += f"- **Audio Track:** ❌ Not available
"
                result_text += "
"
            
            result_text += f"### 🎯 Extraction Details

"
            result_text += f"- **Frame Method:** {extraction_method}
"
            result_text += f"- **Frames Extracted:** {len(frames)}
"
            result_text += f"- **Audio Analysis:** {'✅ Enabled' if analyze_audio else '❌ Disabled'}

"
            
            result_text += f"---

"
            result_text += f"### 🤖 AI Analysis

"
            result_text += visual_audio_analysis
            
            # Add separate audio findings if available
            if analyze_audio and audio_analysis and not audio_analysis.startswith("❌"):
                result_text += f"

---

"
                result_text += f"### 🎤 Detailed Audio Analysis

"
                result_text += audio_analysis
            
            # Add full transcription if multiple chunks
            if audio_transcription:
                result_text += f"

---

"
                result_text += f"### 📝 Full Audio Transcription

"
                result_text += audio_transcription
            
            return frame_grid, result_text
        
        except Exception as e:
            error_msg = f"❌ Error analyzing video: {str(e)}"
            logging.error(error_msg)
            import traceback
            traceback.print_exc()
            return None, error_msg
    
    def process_video_batch(self, directory_path: str, query: str, 
                           num_frames: int = 10,
                           analyze_audio: bool = True) -> Tuple[Optional[Image.Image], str]:
        """
        Process video batch with optional audio filtering
        
        Args:
            directory_path: Directory containing videos
            query: Search query
            num_frames: Frames per video
            analyze_audio: Whether to analyze audio in filtering
        
        Returns: (grid_image, results_text)
        """
        if not self.system_ready:
            return None, f"❌ System not ready: {self.error_message}"
        
        try:
            if not directory_path or not directory_path.strip():
                return None, "❌ Please provide a valid directory path"
            
            if not query or not query.strip():
                return None, "❌ Please provide a search query"
            
            directory_path = os.path.normpath(directory_path.strip())
            video_paths = self.video_processor.load_videos_from_directory(directory_path)
            
            if not video_paths:
                return None, f"❌ No valid videos found in: {directory_path}"
            
            # Filter videos
            matched_videos = []
            
            if analyze_audio:
                filter_prompt = f"""Look at the video frames and listen to the audio carefully. I am searching for: "{query}"

Consider both visual content and spoken words/audio content.

Respond in exactly this format:
YES or NO
Brief explanation of your decision"""
            else:
                filter_prompt = f"""Look at these video frames carefully. I am searching for: "{query}"

Respond in exactly this format:
YES or NO
Brief explanation of your decision"""
            
            logging.info(f"Filtering {len(video_paths)} videos with query: '{query}'")
            logging.info(f"Audio analysis: {'Enabled' if analyze_audio else 'Disabled'}")
            
            for video_path in tqdm(video_paths, desc="Analyzing videos"):
                try:
                    # Extract fewer frames for filtering
                    frames_with_timestamps = self.video_processor.extract_frames_uniform(video_path, min(5, num_frames))
                    if not frames_with_timestamps:
                        continue
                    
                    frames = [f[0] for f in frames_with_timestamps]
                    
                    # Process audio if enabled
                    audio_chunk = None
                    if analyze_audio:
                        audio_path = self.audio_processor.extract_audio_from_video(video_path)
                        if audio_path:
                            audio_array = self.audio_processor.preprocess_audio_for_gemma(audio_path)
                            if audio_array is not None:
                                # Use first 30s chunk
                                audio_chunks = self.audio_processor.split_audio_chunks(audio_array)
                                if audio_chunks:
                                    audio_chunk = audio_chunks[0]
                    
                    # Analyze with both video and audio
                    response = self.model.analyze_multimodal(frames, audio_chunk, filter_prompt)
                    
                    if response.startswith("❌"):
                        continue
                    
                    lines = response.strip().split('
')
                    if len(lines) >= 1:
                        decision_line = lines[0].strip().upper()
                        explanation = '
'.join(lines[1:]).strip() if len(lines) > 1 else "No explanation"
                        
                        is_match = (
                            decision_line.startswith('YES') or 'yes' in decision_line.lower() or
                            'match' in decision_line.lower() or 'found' in decision_line.lower()
                        )
                        
                        if is_match:
                            matched_videos.append((video_path, explanation, frames[0]))
                            logging.info(f"✅ Match found: {os.path.basename(video_path)}")
                
                except Exception as e:
                    logging.error(f"Error analyzing {video_path}: {e}")
                    continue
                
                # Memory cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            if not matched_videos:
                return None, f"❌ No videos found matching: '{query}'

Processed: {len(video_paths)} videos"
            
            # Generate results
            success_rate = (len(matched_videos) / len(video_paths)) * 100
            result_text = f"# 🎯 Multimodal Video Search Results

"
            result_text += f"**Search Query:** "{query}"
"
            result_text += f"**Search Mode:** {'🎥🎤 Video + Audio' if analyze_audio else '🎥 Video Only'}
"
            result_text += f"**Found:** {len(matched_videos)} matches out of {len(video_paths)} total videos
"
            result_text += f"**Success Rate:** {success_rate:.1f}%

"
            result_text += "---

"
            result_text += "## 📋 Matched Videos

"
            
            preview_frames = []
            for i, (video_path, explanation, frame) in enumerate(matched_videos, 1):
                filename = os.path.basename(video_path)
                result_text += f"### {i}. {filename}
"
                result_text += f"**📁 Location:** `{video_path}`
"
                result_text += f"**🤖 Analysis:** {explanation}

"
                preview_frames.append(frame)
            
            grid_image = self.video_processor.create_frame_grid(preview_frames)
            return grid_image, result_text
        
        except Exception as e:
            error_msg = f"❌ Error processing batch: {str(e)}"
            logging.error(error_msg)
            return None, error_msg


class GradioInterface:
    """Professional Gradio interface for multimodal video+audio analysis"""
    
    def __init__(self, analyzer: MultimodalAnalyzer):
        self.analyzer = analyzer
        self.config = analyzer.config
    
    def analyze_single_video_interface(self, video_path, custom_prompt, extraction_method, 
                                      num_frames, analyze_audio):
        """Single video analysis interface"""
        if not video_path:
            return None, "❌ No video uploaded. Please select a video to analyze."
        
        try:
            return self.analyzer.analyze_single_video(
                video_path, custom_prompt, extraction_method, num_frames, analyze_audio
            )
        except Exception as e:
            return None, f"❌ Error: {str(e)}"
    
    def analyze_batch_interface(self, directory_path, query, num_frames, analyze_audio):
        """Batch analysis interface"""
        if not directory_path or not directory_path.strip():
            return None, "❌ Please provide a valid directory path."
        
        if not query or not query.strip():
            return None, "❌ Please provide a search query."
        
        try:
            return self.analyzer.process_video_batch(directory_path, query, num_frames, analyze_audio)
        except Exception as e:
            return None, f"❌ Error: {str(e)}"
    
    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface"""
        
        custom_css = """
        .gradio-container {
            max-width: 1400px !important;
            margin: auto !important;
        }
        .main-header {
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 25px;
        }
        .audio-toggle {
            background-color: #f0f7ff;
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }
        """
        
        with gr.Blocks(
            theme=gr.themes.Soft(),
            title="Gemma 3n-E4B-it Multimodal Analysis",
            css=custom_css
        ) as interface:
            
            # Header
            gr.HTML(f"""
            <div class="main-header">
                <h1>{self.config.INTERFACE_TITLE}</h1>
                <p><strong>Powered by Google's Gemma 3n-E4B-it Model</strong></p>
                <p>4B Effective Parameters • MatFormer Architecture • Video + Audio Understanding</p>
                <p>🎥 Visual Analysis | 🎤 Speech Recognition | 🌍 100+ Languages</p>
            </div>
            """)
            
            gr.Markdown(self.config.INTERFACE_DESCRIPTION)
            
            # System status
            status = self.analyzer.get_system_status()
            gr.Markdown(f"**System Status:** {status}")
            
            with gr.Tabs():
                # Single Video Analysis Tab
                with gr.Tab("📹 Single Video Analysis"):
                    gr.Markdown("### Analyze a single video file with optional audio analysis")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            video_input = gr.Video(label="Upload Video")
                            
                            # Audio toggle
                            gr.HTML('<div class="audio-toggle">')
                            audio_toggle = gr.Checkbox(
                                label="🎤 Enable Audio Analysis",
                                value=True,
                                info="Analyze speech, transcribe audio, and extract names/places mentioned"
                            )
                            gr.HTML('</div>')
                            
                            prompt_input = gr.Textbox(
                                label="Custom Prompt (Optional)",
                                placeholder="Ask about specific aspects of the video...",
                                lines=3
                            )
                            
                            extraction_method = gr.Radio(
                                choices=['uniform', 'fps', 'keyframe'],
                                value='uniform',
                                label="Frame Extraction Method",
                                info="uniform: evenly distributed | fps: time-based | keyframe: scene changes"
                            )
                            
                            num_frames_slider = gr.Slider(
                                minimum=5,
                                maximum=30,
                                value=10,
                                step=1,
                                label="Number of Frames to Extract"
                            )
                            
                            analyze_btn = gr.Button("🎬 Analyze Video", variant="primary", size="lg")
                        
                        with gr.Column(scale=2):
                            frame_gallery = gr.Image(label="Extracted Frames")
                            analysis_output = gr.Markdown(label="Analysis Results")
                    
                    analyze_btn.click(
                        fn=self.analyze_single_video_interface,
                        inputs=[video_input, prompt_input, extraction_method, num_frames_slider, audio_toggle],
                        outputs=[frame_gallery, analysis_output]
                    )
                
                # Batch Video Analysis Tab
                with gr.Tab("📂 Batch Video Search"):
                    gr.Markdown("### Search through multiple videos with multimodal understanding")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            directory_input = gr.Textbox(
                                label="Video Directory Path",
                                placeholder="C:\\Videos\\MyVideos",
                                lines=1
                            )
                            
                            query_input = gr.Textbox(
                                label="Search Query",
                                placeholder="E.g., 'videos mentioning Paris', 'people dancing', 'news about technology'...",
                                lines=2
                            )
                            
                            # Audio toggle for batch
                            gr.HTML('<div class="audio-toggle">')
                            batch_audio_toggle = gr.Checkbox(
                                label="🎤 Include Audio in Search",
                                value=True,
                                info="Search both visual content and spoken words/audio"
                            )
                            gr.HTML('</div>')
                            
                            batch_num_frames = gr.Slider(
                                minimum=3,
                                maximum=15,
                                value=5,
                                step=1,
                                label="Frames per Video (for filtering)"
                            )
                            
                            batch_btn = gr.Button("🔍 Search Videos", variant="primary", size="lg")
                        
                        with gr.Column(scale=2):
                            batch_gallery = gr.Image(label="Matched Video Previews")
                            batch_output = gr.Markdown(label="Search Results")
                    
                    batch_btn.click(
                        fn=self.analyze_batch_interface,
                        inputs=[directory_input, query_input, batch_num_frames, batch_audio_toggle],
                        outputs=[batch_gallery, batch_output]
                    )
                
                # Help Tab
                with gr.Tab("ℹ️ Help & Info"):
                    gr.Markdown("""
                    ## How to Use
                    
                    ### Single Video Analysis
                    1. Upload a video file (MP4, AVI, MOV, MKV, etc.)
                    2. **Toggle Audio Analysis ON** to analyze speech and extract key information
                    3. (Optional) Enter a custom prompt for specific analysis
                    4. Choose frame extraction method
                    5. Set number of frames to extract (5-30)
                    6. Click "Analyze Video"
                    
                    ### Batch Video Search
                    1. Enter directory path containing videos
                    2. Enter search query (can reference visual OR audio content)
                    3. **Toggle "Include Audio in Search"** to search both visuals and speech
                    4. Set frames per video for filtering
                    5. Click "Search Videos"
                    
                    ## Audio Analysis Features
                    
                    When audio analysis is enabled, the system will:
                    - 🎤 **Transcribe speech** in 100+ languages
                    - 📍 **Extract named entities**: people, places, organizations
                    - 🗣️ **Analyze tone and context** of spoken content
                    - 🔍 **Identify key topics** and themes mentioned
                    - 🎯 **Correlate audio with visual** content for deeper insights
                    
                    ### Example Search Queries (with Audio)
                    - "Videos mentioning New York or Paris"
                    - "People discussing technology or AI"
                    - "News reports about elections"
                    - "Interviews with person named John"
                    - "Videos with music or singing"
                    
                    ## Supported Formats
                    - **Video**: MP4, AVI, MOV, MKV, WMV, FLV, WEBM
                    - **Audio**: Any audio track in supported video formats
                    - **Languages**: 100+ spoken languages for transcription
                    
                    ## Technical Details
                    - **Model**: google/gemma-3n-e4b-it
                    - **Parameters**: 4B effective (8B total)
                    - **Audio Encoder**: Universal Speech Model (USM)
                    - **Audio Format**: 16kHz, mono, float32, [-1, 1] range
                    - **Max Audio Length**: 30 seconds per chunk (auto-split for longer audio)
                    - **Context**: 32K tokens
                    - **Framework**: Transformers >= 4.53.0
                    - **Device Fix**: All inputs recursively moved to model.device
                    
                    ## Tips for Best Results
                    - ✅ Enable audio analysis for richer insights
                    - ✅ Use specific queries mentioning topics, names, or places
                    - ✅ For long videos, keyframe extraction works well
                    - ✅ Audio quality impacts transcription accuracy
                    - ⚠️ Videos without audio will skip audio analysis
                    - ⚠️ GPU with 16GB+ VRAM recommended for best performance
                    
                    ## Troubleshooting
                    - **"device mismatch" error**: Fixed by recursively moving all tensors to model.device
                    - **"dtype mismatch" error**: Fixed by converting float tensors to bfloat16
                    - **"MoviePy import" error**: Use `from moviepy import VideoFileClip` (v2.0+)
                    - **Out of memory**: Reduce number of frames or disable audio analysis
                    """)
            
            gr.Markdown("---")
            gr.Markdown("*⚠️ AI-generated analysis. Verify results for critical applications. Audio transcription accuracy depends on audio quality and clarity.*")
        
        return interface


def main():
    """Main entry point"""
    try:
        logging.info("Starting Gemma 3n-E4B-it Multimodal Analysis System...")
        
        # Check dependencies
        logging.info("Checking dependencies...")
        logging.info(f"✅ PyTorch: {torch.__version__}")
        logging.info(f"✅ Transformers: {transformers_version}")
        logging.info(f"✅ OpenCV: {cv2.__version__}")
        logging.info(f"✅ Librosa available")
        logging.info(f"✅ MoviePy available")
        
        # Initialize analyzer
        analyzer = MultimodalAnalyzer()
        
        # Create and launch interface
        gradio_interface = GradioInterface(analyzer)
        interface = gradio_interface.create_interface()
        
        # Launch
        interface.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            show_error=True
        )
    
    except Exception as e:
        logging.error(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()