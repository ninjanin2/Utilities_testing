"""
Advanced LLaVA-NeXT-Video Analysis System - Memory Optimized Edition
Professional-grade video analysis software optimized for RTX A4000 (16GB VRAM) + 32GB RAM
Features: 8-bit quantization, enhanced UI, batch processing
"""

import os
import gc
import av
import cv2
import torch
import numpy as np
import gradio as gr
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor, BitsAndBytesConfig
import logging
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# ================================
# GLOBAL CONFIGURATION - OPTIMIZED FOR RTX A4000
# ================================

# SET YOUR LOCAL MODEL PATH HERE
MODEL_LOCAL_PATH = "/path/to/your/local/LLaVA-NeXT-Video-7B-hf"

# Memory-Optimized Configuration
MAX_FRAMES_PER_CLIP = 16  # Reduced from 32 for better memory management
MAX_VIDEO_LENGTH_SECONDS = 60  # Reduced for memory efficiency
OVERLAP_SECONDS = 1  # Reduced overlap
TARGET_FPS = 0.5  # Lower sampling rate
MAX_NEW_TOKENS = 384  # Reduced token count
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}

# Preprocessing configuration - Optimized
RESIZE_WIDTH = 224  # Reduced from 336 for memory
RESIZE_HEIGHT = 224
CONTRAST_ENHANCEMENT = False  # Disabled for speed
BRIGHTNESS_NORMALIZATION = False  # Disabled for speed
ENABLE_DENOISING = False  # Disabled for memory

# Memory management
CLEAR_CACHE_EVERY_N_CLIPS = 1  # Clear cache after each clip
USE_8BIT_QUANTIZATION = True  # Enable 8-bit quantization

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('video_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ================================
# DATA CLASSES
# ================================

@dataclass
class VideoClip:
    """Represents a video clip segment"""
    frames: np.ndarray
    start_time: float
    end_time: float
    clip_index: int
    total_clips: int

@dataclass
class AnalysisResult:
    """Stores analysis results"""
    video_path: str
    prompt: str
    analysis: str
    timestamp: str
    duration: float
    frame_count: int
    clips_analyzed: int
    memory_used_mb: float

# ================================
# MODEL MANAGER WITH 8-BIT QUANTIZATION
# ================================

class ModelManager:
    """Manages model loading and inference with 8-bit quantization for memory optimization"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing ModelManager with device: {self.device}")
        
    def load_model(self):
        """Load model with 8-bit quantization to reduce VRAM usage"""
        try:
            logger.info(f"Loading model from {self.model_path} with 8-bit quantization")
            
            # Load processor
            self.processor = LlavaNextVideoProcessor.from_pretrained(
                self.model_path,
                local_files_only=True
            )
            
            # Configure 8-bit quantization for memory efficiency
            if USE_8BIT_QUANTIZATION and torch.cuda.is_available():
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                )
                logger.info("Using 8-bit quantization for memory optimization")
            else:
                quantization_config = None
                logger.warning("8-bit quantization disabled or CUDA not available")
            
            # Load model with optimizations
            self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
                self.model_path,
                quantization_config=quantization_config,
                device_map="auto",
                local_files_only=True,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16 if not USE_8BIT_QUANTIZATION else None,
            )
            
            self.model.eval()
            
            # Log memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
            
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def generate_response(self, frames: np.ndarray, prompt: str) -> str:
        """Generate response with aggressive memory management"""
        try:
            # Clear cache before processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Prepare conversation
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video"},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            
            # Apply chat template
            prompt_formatted = self.processor.apply_chat_template(
                conversation, 
                add_generation_prompt=True
            )
            
            # Process inputs with memory optimization
            inputs = self.processor(
                text=prompt_formatted,
                videos=frames,
                padding=True,
                return_tensors="pt"
            )
            
            # Move to device
            if USE_8BIT_QUANTIZATION:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            else:
                inputs = {k: v.to(self.device, torch.float16) for k, v in inputs.items()}
            
            # Generate with memory-efficient settings
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=True):
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        num_beams=1,
                        use_cache=True,
                    )
            
            # Decode output
            response = self.processor.decode(output[0], skip_special_tokens=True)
            
            # Extract assistant response
            if "ASSISTANT:" in response:
                response = response.split("ASSISTANT:")[-1].strip()
            
            # Aggressive cleanup
            del inputs, output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            # Emergency memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            return f"Error during analysis: {str(e)}"
    
    def cleanup(self):
        """Clean up GPU memory"""
        if self.model is not None:
            del self.model
            del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

# ================================
# VIDEO PREPROCESSING - MEMORY OPTIMIZED
# ================================

class VideoPreprocessor:
    """Optimized video preprocessing with minimal memory footprint"""
    
    @staticmethod
    def read_video_pyav(container, indices: List[int]) -> np.ndarray:
        """Decode video frames efficiently"""
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                # Convert and resize immediately to save memory
                frame_array = frame.to_ndarray(format="rgb24")
                frame_resized = cv2.resize(
                    frame_array, 
                    (RESIZE_WIDTH, RESIZE_HEIGHT), 
                    interpolation=cv2.INTER_AREA
                )
                frames.append(frame_resized)
        
        return np.stack(frames) if frames else np.array([])
    
    @staticmethod
    def enhance_frame(frame: np.ndarray) -> np.ndarray:
        """Lightweight frame enhancement"""
        # Only apply if enabled
        if BRIGHTNESS_NORMALIZATION:
            frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
        
        if CONTRAST_ENHANCEMENT:
            lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        if ENABLE_DENOISING:
            frame = cv2.fastNlMeansDenoisingColored(frame, None, 5, 5, 7, 21)
        
        return frame
    
    @staticmethod
    def extract_frames_optimized(video_path: str, max_frames: int = MAX_FRAMES_PER_CLIP) -> Tuple[np.ndarray, dict]:
        """Extract frames with memory-efficient approach"""
        try:
            container = av.open(video_path)
            stream = container.streams.video[0]
            
            total_frames = stream.frames
            fps = float(stream.average_rate)
            duration = float(stream.duration * stream.time_base) if stream.duration else 0
            
            metadata = {
                'total_frames': total_frames,
                'fps': fps,
                'duration': duration,
                'width': stream.width,
                'height': stream.height
            }
            
            # Calculate indices
            if total_frames <= max_frames:
                indices = np.arange(0, total_frames).astype(int)
            else:
                indices = np.linspace(0, total_frames - 1, max_frames).astype(int)
            
            # Extract frames (already resized in read_video_pyav)
            frames = VideoPreprocessor.read_video_pyav(container, indices.tolist())
            container.close()
            
            # Optional enhancement
            if CONTRAST_ENHANCEMENT or BRIGHTNESS_NORMALIZATION or ENABLE_DENOISING:
                enhanced_frames = []
                for frame in frames:
                    enhanced_frame = VideoPreprocessor.enhance_frame(frame)
                    enhanced_frames.append(enhanced_frame)
                frames = np.stack(enhanced_frames)
            
            logger.info(f"Extracted {len(frames)} frames from {video_path}")
            return frames, metadata
            
        except Exception as e:
            logger.error(f"Error extracting frames from {video_path}: {str(e)}")
            raise
    
    @staticmethod
    def split_long_video(video_path: str) -> List[VideoClip]:
        """Split long videos into memory-efficient clips"""
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            cap.release()
            
            clips = []
            
            if duration <= MAX_VIDEO_LENGTH_SECONDS:
                frames, metadata = VideoPreprocessor.extract_frames_optimized(video_path)
                clips.append(VideoClip(
                    frames=frames,
                    start_time=0,
                    end_time=duration,
                    clip_index=0,
                    total_clips=1
                ))
            else:
                clip_duration = MAX_VIDEO_LENGTH_SECONDS
                num_clips = int(np.ceil(duration / clip_duration))
                
                container = av.open(video_path)
                stream = container.streams.video[0]
                stream_fps = float(stream.average_rate)
                
                for i in range(num_clips):
                    start_time = max(0, i * clip_duration - (OVERLAP_SECONDS if i > 0 else 0))
                    end_time = min(duration, (i + 1) * clip_duration + OVERLAP_SECONDS)
                    
                    start_frame = int(start_time * stream_fps)
                    end_frame = int(end_time * stream_fps)
                    
                    clip_frames_count = min(MAX_FRAMES_PER_CLIP, end_frame - start_frame)
                    indices = np.linspace(start_frame, end_frame - 1, clip_frames_count).astype(int)
                    
                    frames = VideoPreprocessor.read_video_pyav(container, indices.tolist())
                    
                    clips.append(VideoClip(
                        frames=frames,
                        start_time=start_time,
                        end_time=end_time,
                        clip_index=i,
                        total_clips=num_clips
                    ))
                    
                    # Clear memory after each clip
                    gc.collect()
                
                container.close()
                logger.info(f"Split video into {num_clips} clips")
            
            return clips
            
        except Exception as e:
            logger.error(f"Error splitting video: {str(e)}")
            raise

# ================================
# VIDEO ANALYZER
# ================================

class VideoAnalyzer:
    """Main video analysis engine with memory optimization"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.preprocessor = VideoPreprocessor()
    
    def analyze_single_video(self, video_path: str, prompt: Optional[str] = None) -> AnalysisResult:
        """Analyze single video with memory tracking"""
        try:
            start_time = datetime.now()
            
            if not prompt or prompt.strip() == "":
                prompt = "Provide a detailed description of this video, including all activities, objects, people, setting, and any notable events or actions occurring."
            
            logger.info(f"Analyzing video: {video_path}")
            
            # Split video into clips
            clips = self.preprocessor.split_long_video(video_path)
            
            # Analyze each clip
            clip_analyses = []
            for idx, clip in enumerate(clips):
                logger.info(f"Processing clip {idx + 1}/{len(clips)}")
                
                if clip.total_clips > 1:
                    clip_prompt = f"{prompt}

[Clip {clip.clip_index + 1}/{clip.total_clips}, time {clip.start_time:.1f}s-{clip.end_time:.1f}s]"
                else:
                    clip_prompt = prompt
                
                analysis = self.model_manager.generate_response(clip.frames, clip_prompt)
                clip_analyses.append({
                    'clip_index': clip.clip_index,
                    'start_time': clip.start_time,
                    'end_time': clip.end_time,
                    'analysis': analysis,
                    'frame_count': len(clip.frames)
                })
                
                # Clear memory after each clip
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            
            # Combine analyses
            if len(clips) == 1:
                final_analysis = clip_analyses[0]['analysis']
            else:
                final_analysis = self._synthesize_multi_clip_analysis(clip_analyses, prompt)
            
            # Calculate metrics
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            total_frames = sum(clip['frame_count'] for clip in clip_analyses)
            
            # Get memory usage
            memory_used = 0
            if torch.cuda.is_available():
                memory_used = torch.cuda.max_memory_allocated() / 1024**2
                torch.cuda.reset_peak_memory_stats()
            
            result = AnalysisResult(
                video_path=video_path,
                prompt=prompt,
                analysis=final_analysis,
                timestamp=start_time.strftime("%Y-%m-%d %H:%M:%S"),
                duration=duration,
                frame_count=total_frames,
                clips_analyzed=len(clips),
                memory_used_mb=memory_used
            )
            
            logger.info(f"Analysis completed in {duration:.2f}s, Memory: {memory_used:.2f}MB")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing video: {str(e)}")
            raise
    
    def _synthesize_multi_clip_analysis(self, clip_analyses: List[Dict], original_prompt: str) -> str:
        """Create comprehensive report from multiple clips"""
        final_report = ""
        
        for clip_analysis in clip_analyses:
            time_range = f"{clip_analysis['start_time']:.1f}s - {clip_analysis['end_time']:.1f}s"
            final_report += f"**[Segment {clip_analysis['clip_index'] + 1}]** ({time_range}):
{clip_analysis['analysis']}

"
        
        return final_report
    
    def batch_search_videos(self, folder_path: str, search_prompt: str) -> Tuple[List[str], str]:
        """Search through videos with memory management"""
        try:
            logger.info(f"Batch search in: {folder_path}")
            
            folder = Path(folder_path)
            video_files = [
                str(f) for f in folder.iterdir() 
                if f.suffix.lower() in VIDEO_EXTENSIONS
            ]
            
            if not video_files:
                return [], "❌ No video files found in the specified folder."
            
            logger.info(f"Found {len(video_files)} video files")
            
            matching_videos = []
            results = []
            
            for video_path in video_files:
                try:
                    video_name = Path(video_path).name
                    logger.info(f"Analyzing {video_name}...")
                    
                    result = self.analyze_single_video(video_path, search_prompt)
                    
                    # Matching logic
                    response_lower = result.analysis.lower()
                    positive_indicators = ['yes', 'found', 'detected', 'present', 'visible', 'appears', 'shows', 'contains']
                    negative_indicators = ['no', 'not found', 'absent', 'missing', 'does not', 'cannot see']
                    
                    positive_score = sum(1 for indicator in positive_indicators if indicator in response_lower)
                    negative_score = sum(1 for indicator in negative_indicators if indicator in response_lower)
                    
                    is_match = positive_score > negative_score
                    
                    results.append({
                        'video_path': video_path,
                        'video_name': video_name,
                        'is_match': is_match,
                        'analysis': result.analysis,
                        'confidence': positive_score - negative_score
                    })
                    
                    if is_match:
                        matching_videos.append(video_path)
                    
                    # Memory cleanup
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    
                except Exception as e:
                    logger.error(f"Error processing {video_path}: {str(e)}")
                    results.append({
                        'video_path': video_path,
                        'video_name': Path(video_path).name,
                        'is_match': False,
                        'analysis': f"Error: {str(e)}",
                        'confidence': -999
                    })
            
            markdown_report = self._generate_batch_report(results, search_prompt)
            
            logger.info(f"Batch search completed. Found {len(matching_videos)} matches")
            return matching_videos, markdown_report
            
        except Exception as e:
            logger.error(f"Batch search error: {str(e)}")
            return [], f"Error: {str(e)}"
    
    def _generate_batch_report(self, results: List[Dict], search_prompt: str) -> str:
        """Generate formatted markdown report"""
        report = f"# 🔍 Batch Video Search Results

"
        report += f"**Search Query:** {search_prompt}

"
        report += f"**Total Videos:** {len(results)} | **Matches:** {sum(1 for r in results if r['is_match'])}

"
        report += "---

"
        
        results_sorted = sorted(results, key=lambda x: x['confidence'], reverse=True)
        
        matching = [r for r in results_sorted if r['is_match']]
        if matching:
            report += "## ✅ Matching Videos

"
            for i, result in enumerate(matching, 1):
                report += f"### {i}. {result['video_name']}
"
                report += f"**Confidence:** {result['confidence']} | **Analysis:**
{result['analysis']}

---

"
        
        non_matching = [r for r in results_sorted if not r['is_match']]
        if non_matching:
            report += "## ❌ Non-Matching Videos

"
            for i, result in enumerate(non_matching, 1):
                report += f"### {i}. {result['video_name']}
"
                report += f"{result['analysis']}

---

"
        
        return report

# ================================
# GRADIO INTERFACE WITH ENHANCED UI
# ================================

class GradioInterface:
    """Professional Gradio interface with beautiful report visualization"""
    
    def __init__(self, analyzer: VideoAnalyzer):
        self.analyzer = analyzer
    
    def format_single_video_output(self, result: AnalysisResult) -> str:
        """Format output as beautiful HTML"""
        html = f"""
        <div style="font-family: 'Segoe UI', Arial, sans-serif; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;">
            <h1 style="margin: 0 0 20px 0; font-size: 28px;">📹 Video Analysis Report</h1>
        </div>
        
        <div style="padding: 25px; background: #f8f9fa; border-radius: 10px; margin-top: 20px;">
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 25px;">
                <div style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <div style="color: #6c757d; font-size: 12px; margin-bottom: 5px;">📁 FILE</div>
                    <div style="font-weight: bold; color: #212529;">{Path(result.video_path).name}</div>
                </div>
                <div style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <div style="color: #6c757d; font-size: 12px; margin-bottom: 5px;">⏱️ DURATION</div>
                    <div style="font-weight: bold; color: #212529;">{result.duration:.2f}s</div>
                </div>
                <div style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <div style="color: #6c757d; font-size: 12px; margin-bottom: 5px;">🎞️ FRAMES</div>
                    <div style="font-weight: bold; color: #212529;">{result.frame_count}</div>
                </div>
                <div style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <div style="color: #6c757d; font-size: 12px; margin-bottom: 5px;">🎬 CLIPS</div>
                    <div style="font-weight: bold; color: #212529;">{result.clips_analyzed}</div>
                </div>
                <div style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <div style="color: #6c757d; font-size: 12px; margin-bottom: 5px;">💾 MEMORY</div>
                    <div style="font-weight: bold; color: #212529;">{result.memory_used_mb:.1f} MB</div>
                </div>
                <div style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <div style="color: #6c757d; font-size: 12px; margin-bottom: 5px;">🕐 TIMESTAMP</div>
                    <div style="font-weight: bold; color: #212529; font-size: 11px;">{result.timestamp}</div>
                </div>
            </div>
            
            <div style="background: white; padding: 25px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h2 style="color: #667eea; margin-top: 0; font-size: 20px; border-bottom: 2px solid #667eea; padding-bottom: 10px;">
                    🔍 Analysis Results
                </h2>
                <div style="color: #212529; line-height: 1.8; font-size: 15px;">
                    {result.analysis.replace(chr(10), '<br>')}
                </div>
            </div>
            
            <div style="margin-top: 20px; padding: 15px; background: #e3f2fd; border-left: 4px solid #2196f3; border-radius: 5px;">
                <strong style="color: #1976d2;">💡 Tip:</strong> <span style="color: #424242;">Use custom prompts to get specific insights about your video!</span>
            </div>
        </div>
        """
        return html
    
    def single_video_interface(self, video_path, prompt, progress=gr.Progress()):
        """Handle single video analysis with progress"""
        try:
            if not video_path:
                return "<div style='padding: 20px; background: #fff3cd; border-radius: 10px; color: #856404;'>⚠️ Please upload a video file.</div>"
            
            progress(0, desc="Initializing analysis...")
            result = self.analyzer.analyze_single_video(video_path, prompt)
            progress(1, desc="Complete!")
            
            return self.format_single_video_output(result)
            
        except Exception as e:
            error_html = f"""
            <div style="padding: 20px; background: #f8d7da; border-radius: 10px; color: #721c24; border-left: 4px solid #dc3545;">
                <h3 style="margin-top: 0;">❌ Error During Analysis</h3>
                <p>{str(e)}</p>
                <small>Check logs for more details.</small>
            </div>
            """
            return error_html
    
    def batch_video_interface(self, folder_path, search_prompt, progress=gr.Progress()):
        """Handle batch video search"""
        try:
            if not folder_path or not os.path.isdir(folder_path):
                return None, "⚠️ Please provide a valid folder path."
            
            if not search_prompt or search_prompt.strip() == "":
                return None, "⚠️ Please provide a search prompt."
            
            progress(0, desc="Starting batch search...")
            matching_videos, markdown_report = self.analyzer.batch_search_videos(folder_path, search_prompt)
            progress(1, desc="Search complete!")
            
            gallery_data = matching_videos[:12] if matching_videos else None
            
            return gallery_data, markdown_report
            
        except Exception as e:
            return None, f"❌ **Error:** {str(e)}"
    
    def create_interface(self):
        """Create enhanced Gradio interface"""
        
        custom_css = """
        .main-header {
            text-align: center;
            padding: 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 15px;
            margin-bottom: 30px;
        }
        .output-html {
            border: none !important;
        }
        .video-input {
            border: 2px dashed #667eea !important;
            border-radius: 10px !important;
        }
        """
        
        with gr.Blocks(
            title="LLaVA Video Analysis System",
            theme=gr.themes.Soft(primary_hue="purple", secondary_hue="blue"),
            css=custom_css
        ) as interface:
            
            gr.HTML("""
            <div class="main-header">
                <h1 style="margin: 0; font-size: 36px; font-weight: bold;">🎥 Advanced Video Analysis System</h1>
                <p style="margin: 10px 0 0 0; font-size: 16px; opacity: 0.9;">
                    Powered by LLaVA-NeXT-Video-7B with 8-bit Quantization
                </p>
                <p style="margin: 5px 0 0 0; font-size: 14px; opacity: 0.8;">
                    Optimized for RTX A4000 (16GB VRAM) | Professional AI Video Understanding
                </p>
            </div>
            """)
            
            with gr.Tabs():
                # Tab 1: Single Video Analysis
                with gr.Tab("📹 Single Video Analysis", id=0):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### 📤 Upload Video")
                            video_input = gr.Video(
                                label="Video Input",
                                interactive=True,  # Enable trim functionality
                                show_label=False,
                                elem_classes="video-input"
                            )
                            
                            with gr.Accordion("⚙️ Advanced Options", open=False):
                                prompt_input = gr.Textbox(
                                    label="Custom Analysis Prompt (Optional)",
                                    placeholder="e.g., 'Describe all activities and objects in detail'",
                                    lines=3,
                                    info="Leave empty for default comprehensive analysis"
                                )
                            
                            analyze_btn = gr.Button(
                                "🔍 Analyze Video",
                                variant="primary",
                                size="lg",
                                scale=1
                            )
                            
                            gr.Markdown("""
                            #### 💡 Quick Tips:
                            - Use the **trim tool** on the video player to analyze specific segments
                            - Custom prompts help focus on specific aspects
                            - Supports videos up to several minutes
                            """)
                        
                        with gr.Column(scale=1):
                            gr.Markdown("### 📊 Analysis Results")
                            output_html = gr.HTML(
                                label="Results",
                                elem_classes="output-html"
                            )
                    
                    analyze_btn.click(
                        fn=self.single_video_interface,
                        inputs=[video_input, prompt_input],
                        outputs=output_html
                    )
                    
                    with gr.Accordion("📝 Example Prompts", open=False):
                        gr.Examples(
                            examples=[
                                [None, "Describe all visible objects and their interactions in detail."],
                                [None, "What activities are people performing in this video?"],
                                [None, "Identify the setting and describe the environment."],
                                [None, "Are there any animals in this video? Describe them."],
                                [None, "What is the main action or event happening?"]
                            ],
                            inputs=[video_input, prompt_input],
                        )
                
                # Tab 2: Batch Video Search
                with gr.Tab("🔍 Batch Video Search", id=1):
                    gr.Markdown("""
                    ### 📂 Search Through Multiple Videos
                    Analyze multiple videos in a folder and find content matching your criteria.
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            folder_input = gr.Textbox(
                                label="📁 Video Folder Path",
                                placeholder="/path/to/video/folder",
                                info="Folder containing video files to search through"
                            )
                        with gr.Column(scale=2):
                            search_prompt_input = gr.Textbox(
                                label="🔎 Search Query",
                                placeholder="e.g., 'Find videos with dogs playing'",
                                info="Describe what you're looking for"
                            )
                    
                    search_btn = gr.Button(
                        "🔎 Start Batch Search",
                        variant="primary",
                        size="lg"
                    )
                    
                    gr.Markdown("### 📊 Search Results")
                    
                    with gr.Row():
                        gallery_output = gr.Gallery(
                            label="Matching Videos (Preview)",
                            columns=4,
                            height=300,
                            object_fit="contain",
                            show_label=True
                        )
                    
                    with gr.Accordion("📄 Detailed Report", open=True):
                        report_output = gr.Markdown()
                    
                    search_btn.click(
                        fn=self.batch_video_interface,
                        inputs=[folder_input, search_prompt_input],
                        outputs=[gallery_output, report_output]
                    )
                    
                    with gr.Accordion("💡 Example Searches", open=False):
                        gr.Examples(
                            examples=[
                                ["", "Find videos showing people cooking"],
                                ["", "Search for videos with cats or dogs"],
                                ["", "Locate videos with sports activities"],
                                ["", "Identify videos showing outdoor scenes"]
                            ],
                            inputs=[folder_input, search_prompt_input],
                        )
            
            # Footer
            gr.HTML("""
            <div style="margin-top: 40px; padding: 25px; background: #f8f9fa; border-radius: 10px; border-top: 3px solid #667eea;">
                <h3 style="color: #667eea; margin-top: 0;">⚙️ System Specifications</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;">
                    <div>
                        <strong>🤖 Model:</strong> LLaVA-NeXT-Video-7B (8-bit)<br>
                        <strong>🎞️ Max Frames:</strong> 16 frames per clip<br>
                        <strong>⏱️ Max Clip Length:</strong> 60 seconds
                    </div>
                    <div>
                        <strong>🖼️ Resolution:</strong> 224x224 pixels<br>
                        <strong>💾 Memory Mode:</strong> 8-bit Quantization<br>
                        <strong>🎯 Optimized For:</strong> RTX A4000 (16GB)
                    </div>
                    <div>
                        <strong>📁 Formats:</strong> MP4, AVI, MOV, MKV, FLV, WMV, WEBM<br>
                        <strong>✨ Features:</strong> Auto-split, Batch processing<br>
                        <strong>🔧 Video Trim:</strong> Enabled (use player controls)
                    </div>
                </div>
                <div style="margin-top: 20px; padding: 15px; background: #e7f3ff; border-left: 4px solid #2196f3; border-radius: 5px;">
                    <strong>🚀 Performance Optimizations:</strong> 8-bit quantization reduces VRAM by ~50% | Aggressive memory management | 
                    Frame-level optimization | Support for long videos via auto-splitting
                </div>
            </div>
            """)
        
        return interface

# ================================
# MAIN APPLICATION
# ================================

def main():
    """Main application entry point"""
    
    print("=" * 70)
    print("🎥 Advanced Video Analysis System - Memory Optimized Edition")
    print("=" * 70)
    print("
Optimizations:")
    print("  ✓ 8-bit quantization enabled")
    print("  ✓ Reduced frame count (16 frames)")
    print("  ✓ Aggressive memory management")
    print("  ✓ Enhanced UI with HTML components")
    print("  ✓ Video trim functionality enabled")
    print("=" * 70)
    
    # Verify model path
    if not os.path.exists(MODEL_LOCAL_PATH):
        print(f"
❌ ERROR: Model not found at {MODEL_LOCAL_PATH}")
        print("
📥 Please download the model:")
        print("   https://huggingface.co/llava-hf/LLaVA-NeXT-Video-7B-hf")
        print("
🔧 Then update MODEL_LOCAL_PATH in the script.")
        return
    
    try:
        # Initialize
        print("
[1/3] Initializing Model Manager...")
        model_manager = ModelManager(MODEL_LOCAL_PATH)
        
        print("
[2/3] Loading Model with 8-bit Quantization...")
        print("      (This may take 1-2 minutes on first load)")
        model_manager.load_model()
        
        print("
[3/3] Setting up Enhanced Gradio Interface...")
        analyzer = VideoAnalyzer(model_manager)
        gradio_interface = GradioInterface(analyzer)
        interface = gradio_interface.create_interface()
        
        print("
" + "=" * 70)
        print("✅ System Ready!")
        print("=" * 70)
        print("
🌐 Launching web interface...")
        print("   Local URL: http://localhost:7860")
        print("
⌨️  Press Ctrl+C to stop the server")
        print("=" * 70 + "
")
        
        # Launch
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            inbrowser=True,
            show_error=True,
            max_threads=10
        )
        
    except KeyboardInterrupt:
        print("

🛑 Shutting down gracefully...")
        model_manager.cleanup()
        print("✅ Cleanup complete. Goodbye!")
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        print(f"
❌ Fatal Error: {str(e)}")
        print("
💡 Try reducing MAX_FRAMES_PER_CLIP if memory issues persist")
        raise

if __name__ == "__main__":
    main()