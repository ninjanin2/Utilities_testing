"""
Advanced LLaVA-NeXT-Video Analysis System
Professional-grade video analysis software with single and batch processing capabilities
Optimized for RTX A4000 (16GB) + 32GB RAM
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
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
import logging
from datetime import datetime
import json

# ================================
# GLOBAL CONFIGURATION
# ================================

# SET YOUR LOCAL MODEL PATH HERE
MODEL_LOCAL_PATH = "/path/to/your/local/LLaVA-NeXT-Video-7B-hf"

# Advanced Configuration
MAX_FRAMES_PER_CLIP = 32  # Optimal for RTX A4000
MAX_VIDEO_LENGTH_SECONDS = 120  # Split videos longer than this
OVERLAP_SECONDS = 2  # Overlap between clips for context continuity
TARGET_FPS = 1  # Frame sampling rate
MAX_NEW_TOKENS = 512  # Output length
BATCH_SIZE = 1  # Keep at 1 for 16GB VRAM
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}

# Preprocessing configuration
RESIZE_WIDTH = 336  # LLaVA-NeXT optimal resolution
RESIZE_HEIGHT = 336
CONTRAST_ENHANCEMENT = True
BRIGHTNESS_NORMALIZATION = True

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

# ================================
# MODEL MANAGER
# ================================

class ModelManager:
    """Manages model loading and inference with memory optimization"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing ModelManager with device: {self.device}")
        
    def load_model(self):
        """Load model and processor from local path"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            # Load processor
            self.processor = LlavaNextVideoProcessor.from_pretrained(
                self.model_path,
                local_files_only=True
            )
            
            # Load model with optimizations for RTX A4000
            self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,  # FP16 for memory efficiency
                device_map="auto",
                local_files_only=True,
                low_cpu_mem_usage=True
            )
            
            self.model.eval()  # Set to evaluation mode
            logger.info("Model loaded successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def generate_response(self, frames: np.ndarray, prompt: str) -> str:
        """Generate response for video frames with given prompt"""
        try:
            # Prepare conversation format
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
            
            # Process inputs
            inputs = self.processor(
                text=prompt_formatted,
                videos=frames,
                padding=True,
                return_tensors="pt"
            ).to(self.device, torch.float16)
            
            # Generate with optimized parameters
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    temperature=0.2,
                    top_p=0.9,
                    num_beams=1  # Greedy decoding for speed
                )
            
            # Decode output
            response = self.processor.decode(
                output[0], 
                skip_special_tokens=True
            )
            
            # Extract assistant response
            if "ASSISTANT:" in response:
                response = response.split("ASSISTANT:")[-1].strip()
            
            # Clear cache
            del inputs, output
            torch.cuda.empty_cache()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error during analysis: {str(e)}"
    
    def cleanup(self):
        """Clean up GPU memory"""
        if self.model is not None:
            del self.model
            del self.processor
        torch.cuda.empty_cache()
        gc.collect()

# ================================
# VIDEO PREPROCESSING
# ================================

class VideoPreprocessor:
    """Advanced video preprocessing for optimal model performance"""
    
    @staticmethod
    def read_video_pyav(container, indices: List[int]) -> np.ndarray:
        """
        Decode video frames using PyAV with specified indices
        Returns: np.ndarray of shape (num_frames, height, width, 3)
        """
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame.to_ndarray(format="rgb24"))
        
        return np.stack([x for x in frames])
    
    @staticmethod
    def enhance_frame(frame: np.ndarray) -> np.ndarray:
        """Apply preprocessing enhancements to frame"""
        
        # Brightness normalization
        if BRIGHTNESS_NORMALIZATION:
            lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.normalize(l, None, 0, 255, cv2.NORM_MINMAX)
            lab = cv2.merge([l, a, b])
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Contrast enhancement using CLAHE
        if CONTRAST_ENHANCEMENT:
            lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Denoise
        frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
        
        # Resize to optimal resolution
        frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
        
        return frame
    
    @staticmethod
    def extract_frames_optimized(video_path: str, max_frames: int = MAX_FRAMES_PER_CLIP) -> Tuple[np.ndarray, dict]:
        """
        Extract and preprocess frames from video with optimal sampling
        Returns: (frames_array, metadata)
        """
        try:
            container = av.open(video_path)
            stream = container.streams.video[0]
            
            # Get video metadata
            total_frames = stream.frames
            fps = float(stream.average_rate)
            duration = float(stream.duration * stream.time_base)
            
            metadata = {
                'total_frames': total_frames,
                'fps': fps,
                'duration': duration,
                'width': stream.width,
                'height': stream.height
            }
            
            # Calculate optimal frame indices
            if total_frames <= max_frames:
                indices = np.arange(0, total_frames).astype(int)
            else:
                indices = np.linspace(0, total_frames - 1, max_frames).astype(int)
            
            # Extract frames
            frames = VideoPreprocessor.read_video_pyav(container, indices.tolist())
            
            # Apply preprocessing to each frame
            enhanced_frames = []
            for frame in frames:
                enhanced_frame = VideoPreprocessor.enhance_frame(frame)
                enhanced_frames.append(enhanced_frame)
            
            frames_array = np.stack(enhanced_frames)
            container.close()
            
            logger.info(f"Extracted {len(frames_array)} frames from {video_path}")
            return frames_array, metadata
            
        except Exception as e:
            logger.error(f"Error extracting frames from {video_path}: {str(e)}")
            raise
    
    @staticmethod
    def split_long_video(video_path: str) -> List[VideoClip]:
        """
        Split long videos into manageable clips with overlap
        Returns: List of VideoClip objects
        """
        try:
            # Get video info
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            cap.release()
            
            clips = []
            
            if duration <= MAX_VIDEO_LENGTH_SECONDS:
                # Video is short enough, process as single clip
                frames, metadata = VideoPreprocessor.extract_frames_optimized(video_path)
                clips.append(VideoClip(
                    frames=frames,
                    start_time=0,
                    end_time=duration,
                    clip_index=0,
                    total_clips=1
                ))
            else:
                # Split into clips
                clip_duration = MAX_VIDEO_LENGTH_SECONDS
                num_clips = int(np.ceil(duration / clip_duration))
                
                container = av.open(video_path)
                stream = container.streams.video[0]
                stream_fps = float(stream.average_rate)
                total_stream_frames = stream.frames
                
                for i in range(num_clips):
                    start_time = max(0, i * clip_duration - (OVERLAP_SECONDS if i > 0 else 0))
                    end_time = min(duration, (i + 1) * clip_duration + OVERLAP_SECONDS)
                    
                    start_frame = int(start_time * stream_fps)
                    end_frame = int(end_time * stream_fps)
                    
                    # Calculate indices for this clip
                    clip_frames_count = min(MAX_FRAMES_PER_CLIP, end_frame - start_frame)
                    indices = np.linspace(start_frame, end_frame - 1, clip_frames_count).astype(int)
                    
                    # Extract frames
                    frames = VideoPreprocessor.read_video_pyav(container, indices.tolist())
                    
                    # Enhance frames
                    enhanced_frames = []
                    for frame in frames:
                        enhanced_frame = VideoPreprocessor.enhance_frame(frame)
                        enhanced_frames.append(enhanced_frame)
                    
                    frames_array = np.stack(enhanced_frames)
                    
                    clips.append(VideoClip(
                        frames=frames_array,
                        start_time=start_time,
                        end_time=end_time,
                        clip_index=i,
                        total_clips=num_clips
                    ))
                
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
    """Main video analysis engine"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.preprocessor = VideoPreprocessor()
    
    def analyze_single_video(self, video_path: str, prompt: Optional[str] = None) -> AnalysisResult:
        """
        Analyze a single video with optional prompt
        """
        try:
            start_time = datetime.now()
            
            # Default prompt if none provided
            if not prompt or prompt.strip() == "":
                prompt = "Provide a detailed description of this video, including all activities, objects, people, setting, and any notable events or actions occurring."
            
            logger.info(f"Analyzing video: {video_path}")
            logger.info(f"Prompt: {prompt}")
            
            # Split video into clips if necessary
            clips = self.preprocessor.split_long_video(video_path)
            
            # Analyze each clip
            clip_analyses = []
            for clip in clips:
                if clip.total_clips > 1:
                    clip_prompt = f"{prompt}

[Analyzing clip {clip.clip_index + 1}/{clip.total_clips}, timestamp {clip.start_time:.1f}s - {clip.end_time:.1f}s]"
                else:
                    clip_prompt = prompt
                
                analysis = self.model_manager.generate_response(clip.frames, clip_prompt)
                clip_analyses.append({
                    'clip_index': clip.clip_index,
                    'start_time': clip.start_time,
                    'end_time': clip.end_time,
                    'analysis': analysis
                })
            
            # Combine analyses
            if len(clips) == 1:
                final_analysis = clip_analyses[0]['analysis']
            else:
                # For multiple clips, synthesize a comprehensive report
                final_analysis = self._synthesize_multi_clip_analysis(clip_analyses, prompt)
            
            # Calculate metadata
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Get frame count
            total_frames = sum(len(clip.frames) for clip in clips)
            
            result = AnalysisResult(
                video_path=video_path,
                prompt=prompt,
                analysis=final_analysis,
                timestamp=start_time.strftime("%Y-%m-%d %H:%M:%S"),
                duration=duration,
                frame_count=total_frames,
                clips_analyzed=len(clips)
            )
            
            logger.info(f"Analysis completed in {duration:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing video: {str(e)}")
            raise
    
    def _synthesize_multi_clip_analysis(self, clip_analyses: List[Dict], original_prompt: str) -> str:
        """Synthesize multiple clip analyses into comprehensive report"""
        
        synthesis_prompt = f"""Based on the following analyses of different segments of a video, provide a comprehensive summary:

Original Question: {original_prompt}

Segment Analyses:
"""
        for clip_analysis in clip_analyses:
            synthesis_prompt += f"
[Segment {clip_analysis['clip_index'] + 1}] ({clip_analysis['start_time']:.1f}s - {clip_analysis['end_time']:.1f}s):
{clip_analysis['analysis']}
"
        
        # Use first clip frames as reference for synthesis
        # In production, you might want to extract key frames from all clips
        synthesis_prompt += "
Provide a coherent, comprehensive description of the entire video."
        
        # For now, concatenate analyses with clear segmentation
        final_report = f"# Comprehensive Video Analysis

"
        final_report += f"**Total Segments Analyzed:** {len(clip_analyses)}

"
        
        for clip_analysis in clip_analyses:
            final_report += f"## Segment {clip_analysis['clip_index'] + 1} ({clip_analysis['start_time']:.1f}s - {clip_analysis['end_time']:.1f}s)

"
            final_report += f"{clip_analysis['analysis']}

"
        
        return final_report
    
    def batch_search_videos(self, folder_path: str, search_prompt: str) -> Tuple[List[str], str]:
        """
        Search through multiple videos in a folder for specific content
        Returns: (matching_video_paths, markdown_report)
        """
        try:
            logger.info(f"Starting batch search in folder: {folder_path}")
            logger.info(f"Search prompt: {search_prompt}")
            
            # Find all video files
            folder = Path(folder_path)
            video_files = [
                str(f) for f in folder.iterdir() 
                if f.suffix.lower() in VIDEO_EXTENSIONS
            ]
            
            if not video_files:
                return [], "No video files found in the specified folder."
            
            logger.info(f"Found {len(video_files)} video files")
            
            # Analyze each video
            matching_videos = []
            results = []
            
            for video_path in video_files:
                try:
                    video_name = Path(video_path).name
                    logger.info(f"Analyzing {video_name}...")
                    
                    # Analyze with search prompt
                    result = self.analyze_single_video(video_path, search_prompt)
                    
                    # Simple matching: check if response indicates presence
                    # You can implement more sophisticated matching logic
                    response_lower = result.analysis.lower()
                    
                    # Check for positive indicators
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
                    
                except Exception as e:
                    logger.error(f"Error processing {video_path}: {str(e)}")
                    results.append({
                        'video_path': video_path,
                        'video_name': Path(video_path).name,
                        'is_match': False,
                        'analysis': f"Error: {str(e)}",
                        'confidence': -999
                    })
            
            # Generate markdown report
            markdown_report = self._generate_batch_report(results, search_prompt)
            
            logger.info(f"Batch search completed. Found {len(matching_videos)} matching videos.")
            return matching_videos, markdown_report
            
        except Exception as e:
            logger.error(f"Error in batch search: {str(e)}")
            return [], f"Error during batch search: {str(e)}"
    
    def _generate_batch_report(self, results: List[Dict], search_prompt: str) -> str:
        """Generate markdown report for batch search results"""
        
        report = f"# Batch Video Search Results

"
        report += f"**Search Query:** {search_prompt}

"
        report += f"**Total Videos Analyzed:** {len(results)}

"
        
        # Sort by confidence
        results_sorted = sorted(results, key=lambda x: x['confidence'], reverse=True)
        
        # Matching videos
        matching = [r for r in results_sorted if r['is_match']]
        report += f"**Matching Videos:** {len(matching)}

"
        
        if matching:
            report += "## ✅ Matching Videos

"
            for i, result in enumerate(matching, 1):
                report += f"### {i}. {result['video_name']}

"
                report += f"**Confidence Score:** {result['confidence']}

"
                report += f"**Analysis:**
{result['analysis']}

"
                report += "---

"
        
        # Non-matching videos
        non_matching = [r for r in results_sorted if not r['is_match']]
        if non_matching:
            report += "## ❌ Non-Matching Videos

"
            for i, result in enumerate(non_matching, 1):
                report += f"### {i}. {result['video_name']}

"
                report += f"**Analysis:**
{result['analysis']}

"
                report += "---

"
        
        return report

# ================================
# GRADIO INTERFACE
# ================================

class GradioInterface:
    """Professional Gradio interface for video analysis"""
    
    def __init__(self, analyzer: VideoAnalyzer):
        self.analyzer = analyzer
    
    def single_video_interface(self, video_path, prompt):
        """Handle single video analysis"""
        try:
            if not video_path:
                return "Please upload a video file."
            
            result = self.analyzer.analyze_single_video(video_path, prompt)
            
            # Format output
            output = f"""# Video Analysis Report

**Video:** {Path(result.video_path).name}
**Analysis Time:** {result.timestamp}
**Processing Duration:** {result.duration:.2f} seconds
**Frames Analyzed:** {result.frame_count}
**Clips Processed:** {result.clips_analyzed}

---

## Analysis Results

{result.analysis}

---

*Generated by LLaVA-NeXT-Video Analysis System*
"""
            return output
            
        except Exception as e:
            return f"Error during analysis: {str(e)}"
    
    def batch_video_interface(self, folder_path, search_prompt):
        """Handle batch video search"""
        try:
            if not folder_path or not os.path.isdir(folder_path):
                return None, "Please provide a valid folder path."
            
            if not search_prompt or search_prompt.strip() == "":
                return None, "Please provide a search prompt."
            
            matching_videos, markdown_report = self.analyzer.batch_search_videos(
                folder_path, 
                search_prompt
            )
            
            # Create gallery data
            gallery_data = []
            for video_path in matching_videos[:12]:  # Limit to 12 for display
                gallery_data.append(video_path)
            
            return gallery_data, markdown_report
            
        except Exception as e:
            return None, f"Error during batch search: {str(e)}"
    
    def create_interface(self):
        """Create and return Gradio interface"""
        
        with gr.Blocks(
            title="Advanced Video Analysis System",
            theme=gr.themes.Soft(),
            css="""
            .main-header {text-align: center; padding: 20px;}
            .output-markdown {max-height: 600px; overflow-y: auto;}
            """
        ) as interface:
            
            gr.Markdown("""
            # 🎥 Advanced Video Analysis System
            ### Powered by LLaVA-NeXT-Video-7B
            
            Professional-grade video understanding and search capabilities with AI-powered analysis.
            """, elem_classes="main-header")
            
            with gr.Tabs():
                # Tab 1: Single Video Analysis
                with gr.Tab("📹 Single Video Analysis"):
                    gr.Markdown("""
                    ### Analyze Individual Videos
                    Upload a video and optionally provide a custom prompt for detailed analysis.
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            video_input = gr.Video(
                                label="Upload Video",
                                format="mp4"
                            )
                            prompt_input = gr.Textbox(
                                label="Custom Prompt (Optional)",
                                placeholder="e.g., 'Describe the main activities and objects in this video'",
                                lines=3
                            )
                            analyze_btn = gr.Button(
                                "🔍 Analyze Video",
                                variant="primary",
                                size="lg"
                            )
                        
                        with gr.Column(scale=1):
                            output_markdown = gr.Markdown(
                                label="Analysis Results",
                                elem_classes="output-markdown"
                            )
                    
                    analyze_btn.click(
                        fn=self.single_video_interface,
                        inputs=[video_input, prompt_input],
                        outputs=output_markdown
                    )
                    
                    gr.Examples(
                        examples=[
                            [None, "Describe all the activities happening in this video in detail."],
                            [None, "What objects are present in this video?"],
                            [None, "Identify any people and describe what they are doing."],
                            [None, "Describe the setting and environment of this video."]
                        ],
                        inputs=[video_input, prompt_input],
                        label="Example Prompts"
                    )
                
                # Tab 2: Batch Video Search
                with gr.Tab("🔍 Batch Video Search"):
                    gr.Markdown("""
                    ### Search Through Multiple Videos
                    Provide a folder path containing videos and a search query to find matching content.
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
                                placeholder="e.g., 'Find videos with a golden retriever'",
                                lines=2
                            )
                            search_btn = gr.Button(
                                "🔎 Search Videos",
                                variant="primary",
                                size="lg"
                            )
                    
                    gr.Markdown("### 📊 Search Results")
                    
                    with gr.Row():
                        gallery_output = gr.Gallery(
                            label="Matching Videos",
                            columns=4,
                            height="auto",
                            object_fit="contain"
                        )
                    
                    with gr.Row():
                        report_output = gr.Markdown(
                            label="Detailed Report",
                            elem_classes="output-markdown"
                        )
                    
                    search_btn.click(
                        fn=self.batch_video_interface,
                        inputs=[folder_input, search_prompt_input],
                        outputs=[gallery_output, report_output]
                    )
                    
                    gr.Examples(
                        examples=[
                            ["", "Search for videos with cats"],
                            ["", "Find videos with people playing sports"],
                            ["", "Locate videos showing cooking activities"],
                            ["", "Identify videos with vehicles"]
                        ],
                        inputs=[folder_input, search_prompt_input],
                        label="Example Search Queries"
                    )
            
            gr.Markdown("""
            ---
            ### 📝 System Information
            - **Model:** LLaVA-NeXT-Video-7B-hf
            - **Max Frames per Clip:** 32 frames
            - **Video Preprocessing:** Enhanced (CLAHE, Denoising, Normalization)
            - **Supported Formats:** MP4, AVI, MOV, MKV, FLV, WMV, WEBM
            
            ### ⚙️ Features
            - ✅ Automatic video splitting for long videos
            - ✅ Advanced frame preprocessing and enhancement
            - ✅ Context-aware clip overlap
            - ✅ GPU-optimized inference (FP16)
            - ✅ Batch processing with intelligent search
            - ✅ Detailed markdown reports
            
            *Optimized for RTX A4000 (16GB) + 32GB RAM*
            """)
        
        return interface

# ================================
# MAIN APPLICATION
# ================================

def main():
    """Main application entry point"""
    
    print("="*60)
    print("Advanced Video Analysis System")
    print("Powered by LLaVA-NeXT-Video-7B")
    print("="*60)
    
    # Verify model path
    if not os.path.exists(MODEL_LOCAL_PATH):
        print(f"
❌ ERROR: Model not found at {MODEL_LOCAL_PATH}")
        print("
Please update MODEL_LOCAL_PATH variable with your local model directory.")
        print("Download the model from: https://huggingface.co/llava-hf/LLaVA-NeXT-Video-7B-hf")
        return
    
    try:
        # Initialize components
        print("
[1/3] Initializing Model Manager...")
        model_manager = ModelManager(MODEL_LOCAL_PATH)
        
        print("
[2/3] Loading Model (this may take a minute)...")
        model_manager.load_model()
        
        print("
[3/3] Setting up Gradio Interface...")
        analyzer = VideoAnalyzer(model_manager)
        gradio_interface = GradioInterface(analyzer)
        interface = gradio_interface.create_interface()
        
        print("
" + "="*60)
        print("✅ System Ready!")
        print("="*60)
        print("
Launching web interface...")
        print("Access the application at: http://localhost:7860")
        print("
Press Ctrl+C to stop the server.")
        print("="*60 + "
")
        
        # Launch interface
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            inbrowser=True,
            show_error=True
        )
        
    except KeyboardInterrupt:
        print("

Shutting down gracefully...")
        model_manager.cleanup()
        print("Goodbye!")
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        print(f"
❌ Fatal Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()