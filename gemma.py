#!/usr/bin/env python3
"""
Professional Gemma 3n-E4B-it Vision Analysis System
===================================================

A comprehensive offline vision-language system using only the Gemma 3n-E4B-it model.
Designed for RTX A4000 (16GB VRAM) running on Windows.

Features:
- Single image analysis with custom prompts
- Batch processing with intelligent filtering
- Professional Gradio interface
- Local model loading (offline capable)
- GPU optimization for RTX A4000

Based on: google/gemma-3n-E4B-it
Requirements: transformers >= 4.53.0

Author: AI Assistant
Version: 1.1 (Fixed)
"""

import os
import sys
import warnings
import logging
from typing import List, Tuple, Dict, Any, Optional
import json
from datetime import datetime
import glob
from pathlib import Path
import re

# Core libraries
import numpy as np
try:
    import cv2
except ImportError:
    print("Warning: OpenCV not available. Some image processing features may be limited.")
    cv2 = None

from PIL import Image, ImageEnhance
import torch

# Transformers for Gemma 3n
try:
    from transformers import (
        AutoProcessor, 
        Gemma3nForConditionalGeneration,  # Correct class name
        pipeline,
        __version__ as transformers_version
    )
    
    # Check transformers version
    if transformers_version < "4.53.0":
        print(f"Warning: transformers version {transformers_version} detected.")
        print("Gemma 3n requires transformers >= 4.53.0")
        print("Install with: pip install transformers>=4.53.0")
        
except ImportError as e:
    print(f"Error importing transformers: {e}")
    print("Please install: pip install transformers>=4.53.0")
    sys.exit(1)

# UI and utilities
try:
    import gradio as gr
except ImportError:
    print("Error: Gradio not installed. Please install: pip install gradio")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("Warning: tqdm not available. Progress bars will be disabled.")
    tqdm = lambda x, desc="": x

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Config:
    """Configuration class for the Gemma 3n-E4B-it system"""
    
    # Model settings - Gemma 3n-E4B-it specific
    MODEL_NAME = "google/gemma-3n-e4b-it"
    MODEL_PATH = "models/gemma-3n-e4b-it"  # Local model path
    
    # System settings
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TORCH_DTYPE = torch.bfloat16  # Recommended for Gemma 3n
    MAX_BATCH_SIZE = 1  # Conservative for stability
    SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    # Image settings (based on Gemma 3n specifications)
    SUPPORTED_IMAGE_SIZES = [256, 512, 768]  # Gemma 3n supports these sizes
    DEFAULT_IMAGE_SIZE = 512  # Good balance of quality and performance
    
    # Generation settings optimized for Gemma 3n
    MAX_NEW_TOKENS = 256  # Reduced for stability
    TEMPERATURE = 0.7
    DO_SAMPLE = True
    TOP_P = 0.9
    TOP_K = 40
    
    # UI settings
    INTERFACE_TITLE = "🔍 Gemma 3n-E4B-it Vision Analysis System"
    INTERFACE_DESCRIPTION = """
    ## Advanced Vision-Language Analysis with Gemma 3n-E4B-it
    
    Upload single images or batch process directories using Google's latest Gemma 3n-E4B-it model.
    This system supports text, image, and multimodal inputs with state-of-the-art performance.
    
    **Model Features:**
    - Gemma 3n-E4B-it (4B effective parameters, 8B total)
    - Multimodal: Text + Image analysis
    - 32K context window
    - 140+ language support
    - Selective parameter activation for efficiency
    """

class ImageProcessor:
    """Gemma 3n-specific image processing with error handling"""
    
    def __init__(self, target_size: int = 512):
        self.target_size = target_size
        self.supported_sizes = Config.SUPPORTED_IMAGE_SIZES
        
    def optimize_image_size(self, image: Image.Image) -> Image.Image:
        """Optimize image size for Gemma 3n (256x256, 512x512, or 768x768)"""
        try:
            width, height = image.size
            max_dim = max(width, height)
            
            # Choose the best supported size
            best_size = min(self.supported_sizes, key=lambda x: abs(x - max_dim))
            
            # If image is already smaller than the target, don't upscale
            if max_dim <= best_size:
                target_size = best_size
            else:
                # Use the largest size if image is very large
                target_size = max(self.supported_sizes)
            
            # Resize maintaining aspect ratio, then pad to square
            return self.resize_and_pad(image, target_size)
        except Exception as e:
            logging.error(f"Error optimizing image size: {e}")
            return image
    
    def resize_and_pad(self, image: Image.Image, target_size: int) -> Image.Image:
        """Resize image maintaining aspect ratio and pad to square"""
        try:
            # Calculate resize dimensions
            width, height = image.size
            if width > height:
                new_width = target_size
                new_height = int((height * target_size) / width)
            else:
                new_height = target_size
                new_width = int((width * target_size) / height)
            
            # Resize image
            image = image.resize((new_width, new_height), Image.LANCZOS)
            
            # Create square canvas and center the image
            canvas = Image.new('RGB', (target_size, target_size), (255, 255, 255))
            x_offset = (target_size - new_width) // 2
            y_offset = (target_size - new_height) // 2
            canvas.paste(image, (x_offset, y_offset))
            
            return canvas
        except Exception as e:
            logging.error(f"Error resizing and padding image: {e}")
            return image
    
    def load_and_preprocess_image(self, image_path: str) -> Optional[Image.Image]:
        """Load and preprocess image from file path with error handling"""
        try:
            if not os.path.exists(image_path):
                logging.error(f"Image file does not exist: {image_path}")
                return None
                
            image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            return self.optimize_image_size(image)
            
        except Exception as e:
            logging.error(f"Error loading image {image_path}: {e}")
            return None
    
    def load_images_from_directory(self, directory_path: str) -> List[Tuple[str, Image.Image]]:
        """Load all supported images from a directory with error handling"""
        images = []
        
        try:
            directory = Path(directory_path)
            
            if not directory.exists():
                logging.error(f"Directory does not exist: {directory_path}")
                return images
            
            if not directory.is_dir():
                logging.error(f"Path is not a directory: {directory_path}")
                return images
            
            # Find all supported image files
            for ext in Config.SUPPORTED_FORMATS:
                # Check both lowercase and uppercase
                for pattern in [f"*{ext}", f"*{ext.upper()}"]:
                    for image_path in directory.glob(pattern):
                        try:
                            image = self.load_and_preprocess_image(str(image_path))
                            if image is not None:
                                images.append((str(image_path), image))
                        except Exception as e:
                            logging.warning(f"Skipping corrupted image {image_path}: {e}")
                            continue
            
            logging.info(f"Successfully loaded {len(images)} images from {directory_path}")
            return images
            
        except Exception as e:
            logging.error(f"Error loading images from directory {directory_path}: {e}")
            return images
    
    def create_image_grid(self, images: List[Image.Image], max_cols: int = 4) -> Optional[Image.Image]:
        """Create a grid of images for display with error handling"""
        try:
            if not images:
                return None
            
            # Calculate grid dimensions
            num_images = len(images)
            cols = min(max_cols, num_images)
            rows = (num_images + cols - 1) // cols
            
            # Use consistent cell size
            cell_size = 200  # Fixed size for display
            
            # Create grid canvas
            grid_width = cols * cell_size
            grid_height = rows * cell_size
            grid_image = Image.new('RGB', (grid_width, grid_height), (240, 240, 240))
            
            # Place images in grid
            for idx, image in enumerate(images):
                try:
                    row = idx // cols
                    col = idx % cols
                    x = col * cell_size
                    y = row * cell_size
                    
                    # Resize image to fit cell
                    resized_image = image.resize((cell_size - 4, cell_size - 4), Image.LANCZOS)
                    
                    # Center the image in its cell
                    grid_image.paste(resized_image, (x + 2, y + 2))
                except Exception as e:
                    logging.warning(f"Error placing image {idx} in grid: {e}")
                    continue
            
            return grid_image
            
        except Exception as e:
            logging.error(f"Error creating image grid: {e}")
            return None

class Gemma3nVisionModel:
    """Gemma 3n-E4B-it model wrapper with robust error handling"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.processor = None
        self.model_loaded = False
        
    def load_model(self):
        """Load the Gemma 3n-E4B-it model with comprehensive error handling"""
        logging.info("Loading Gemma 3n-E4B-it model...")
        
        try:
            # Check transformers version first
            if transformers_version < "4.53.0":
                raise ValueError(f"transformers version {transformers_version} is too old. Need >= 4.53.0")
            
            # Determine model path (local first, then online)
            model_path = self.config.MODEL_PATH
            if os.path.exists(model_path) and os.path.isdir(model_path):
                logging.info(f"Loading model from local path: {model_path}")
                local_files_only = True
            else:
                logging.info(f"Local model not found. Using online model: {self.config.MODEL_NAME}")
                model_path = self.config.MODEL_NAME
                local_files_only = False
            
            # Load processor first
            try:
                self.processor = AutoProcessor.from_pretrained(
                    model_path,
                    local_files_only=local_files_only,
                    trust_remote_code=True
                )
                logging.info("✅ Processor loaded successfully")
            except Exception as e:
                logging.error(f"Failed to load processor: {e}")
                raise
            
            # Load model with optimizations for RTX A4000
            try:
                self.model = Gemma3nForConditionalGeneration.from_pretrained(
                    model_path,
                    device_map="auto",
                    torch_dtype=self.config.TORCH_DTYPE,
                    local_files_only=local_files_only,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                ).eval()
                logging.info("✅ Model loaded successfully")
            except Exception as e:
                logging.error(f"Failed to load model: {e}")
                raise
            
            self.model_loaded = True
            
            # Memory optimization
            if self.config.DEVICE == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                allocated_memory = torch.cuda.memory_allocated(0) / 1024**3
                logging.info(f"GPU Memory allocated: {allocated_memory:.2f} GB")
            
            logging.info("🎉 Gemma 3n-E4B-it model loaded successfully!")
            
        except Exception as e:
            logging.error(f"❌ Error loading model: {e}")
            logging.error("Please ensure you have:")
            logging.error("1. transformers >= 4.53.0 installed")
            logging.error("2. Accepted the Gemma license on HuggingFace")
            logging.error("3. Valid model files or internet connection")
            self.model_loaded = False
            raise
    
    def create_chat_messages(self, text_prompt: str, image: Optional[Image.Image] = None) -> List[Dict]:
        """Create properly formatted chat messages for Gemma 3n"""
        try:
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful AI assistant specialized in image analysis and understanding."}]
                }
            ]
            
            user_content = []
            
            # Add image if provided
            if image is not None:
                user_content.append({"type": "image", "image": image})
            
            # Add text prompt
            user_content.append({"type": "text", "text": text_prompt})
            
            messages.append({
                "role": "user",
                "content": user_content
            })
            
            return messages
        except Exception as e:
            logging.error(f"Error creating chat messages: {e}")
            return []
    
    def analyze_image(self, image: Image.Image, prompt: str = "Describe this image in detail.") -> str:
        """Analyze a single image with Gemma 3n-E4B-it"""
        if not self.model_loaded:
            return "❌ Error: Model not loaded properly"
        
        try:
            # Create chat messages
            messages = self.create_chat_messages(prompt, image)
            if not messages:
                return "❌ Error: Failed to create chat messages"
            
            # Apply chat template and process
            try:
                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(self.model.device)
                
                input_len = inputs["input_ids"].shape[-1]
                
                # Generate response with error handling
                with torch.inference_mode():
                    generation = self.model.generate(
                        **inputs,
                        max_new_tokens=self.config.MAX_NEW_TOKENS,
                        temperature=self.config.TEMPERATURE,
                        do_sample=self.config.DO_SAMPLE,
                        top_p=self.config.TOP_P,
                        top_k=self.config.TOP_K,
                        pad_token_id=self.processor.tokenizer.eos_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id
                    )
                
                # Decode response
                generation = generation[0][input_len:]
                decoded = self.processor.decode(generation, skip_special_tokens=True)
                
                return decoded.strip() if decoded.strip() else "No response generated"
                
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                return "❌ Error: GPU out of memory. Try reducing image size or restarting."
            except Exception as e:
                logging.error(f"Error during model inference: {e}")
                return f"❌ Error during analysis: {str(e)}"
                
        except Exception as e:
            logging.error(f"Error analyzing image: {e}")
            return f"❌ Error analyzing image: {str(e)}"
    
    def filter_images_by_query(self, images_with_paths: List[Tuple[str, Image.Image]], 
                              query: str) -> List[Tuple[str, Image.Image, str]]:
        """Filter images based on natural language query using Gemma 3n"""
        filtered_results = []
        
        if not self.model_loaded:
            logging.error("Model not loaded for filtering")
            return filtered_results
        
        # Create optimized filtering prompt
        filter_prompt = f"""Look at this image carefully. I am searching for: "{query}"

Respond with exactly this format:
Line 1: YES (if this image matches) or NO (if it doesn't match)
Line 2: Brief explanation why it matches or doesn't match

Be precise and focus on the specific criteria: "{query}" """
        
        logging.info(f"Filtering {len(images_with_paths)} images with query: '{query}'")
        
        # Process images one by one for stability
        for i, (image_path, image) in enumerate(tqdm(images_with_paths, desc="Filtering images")):
            try:
                response = self.analyze_image(image, filter_prompt)
                
                # Parse the response more robustly
                lines = response.strip().split('\n')
                if len(lines) >= 1:
                    decision_line = lines[0].strip().upper()
                    explanation = '\n'.join(lines[1:]).strip() if len(lines) > 1 else "No explanation provided"
                    
                    # Check for positive match with multiple variations
                    is_match = (
                        decision_line.startswith('YES') or 
                        'yes' in decision_line.lower() or
                        decision_line.startswith('MATCH') or
                        'match' in decision_line.lower()
                    )
                    
                    if is_match:
                        filtered_results.append((image_path, image, explanation))
                        logging.info(f"✅ Match found: {os.path.basename(image_path)}")
                else:
                    logging.warning(f"Empty response for {image_path}")
                    
            except Exception as e:
                logging.error(f"Error filtering image {image_path}: {e}")
                continue
            
            # Clear GPU cache periodically
            if i % 5 == 0 and self.config.DEVICE == "cuda":
                torch.cuda.empty_cache()
        
        logging.info(f"🎯 Found {len(filtered_results)} matching images out of {len(images_with_paths)}")
        return filtered_results

class VisionLanguageAnalyzer:
    """Main analyzer class using Gemma 3n-E4B-it with robust error handling"""
    
    def __init__(self):
        self.config = Config()
        self.image_processor = ImageProcessor(self.config.DEFAULT_IMAGE_SIZE)
        self.model = Gemma3nVisionModel(self.config)
        self.system_ready = False
        self.initialize_system()
    
    def initialize_system(self):
        """Initialize the analysis system with comprehensive error handling"""
        logging.info("🚀 Initializing Gemma 3n-E4B-it Vision Analysis System...")
        logging.info(f"Device: {self.config.DEVICE}")
        
        try:
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name()
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logging.info(f"GPU: {gpu_name}")
                logging.info(f"VRAM: {total_memory:.1f} GB")
            else:
                logging.warning("CUDA not available - using CPU (will be slow)")
            
            # Load model
            self.model.load_model()
            self.system_ready = True
            logging.info("✅ System initialized successfully!")
            
        except Exception as e:
            logging.error(f"❌ Failed to initialize system: {e}")
            self.system_ready = False
    
    def analyze_single_image(self, image: Image.Image, prompt: str = None) -> str:
        """Analyze a single image with Gemma 3n-E4B-it"""
        if not self.system_ready:
            return "❌ System not ready. Please check model loading."
        
        try:
            if prompt is None or not prompt.strip():
                prompt = "Describe this image in detail, including objects, people, colors, setting, and any notable features you observe."
            
            # Optimize image for Gemma 3n
            optimized_image = self.image_processor.optimize_image_size(image)
            
            return self.model.analyze_image(optimized_image, prompt)
            
        except Exception as e:
            logging.error(f"Error in single image analysis: {e}")
            return f"❌ Error analyzing image: {str(e)}"
    
    def process_image_batch(self, directory_path: str, query: str) -> Tuple[Optional[Image.Image], str]:
        """Process a batch of images from directory with filtering using Gemma 3n"""
        if not self.system_ready:
            return None, "❌ System not ready. Please check model loading."
        
        try:
            # Validate inputs
            if not directory_path or not directory_path.strip():
                return None, "❌ Please provide a valid directory path"
            
            if not query or not query.strip():
                return None, "❌ Please provide a search query"
            
            # Load all images from directory
            images_with_paths = self.image_processor.load_images_from_directory(directory_path)
            
            if not images_with_paths:
                return None, f"❌ No valid images found in directory: {directory_path}\n\nSupported formats: {', '.join(Config.SUPPORTED_FORMATS)}"
            
            # Filter images based on query using Gemma 3n
            filtered_results = self.model.filter_images_by_query(images_with_paths, query)
            
            if not filtered_results:
                return None, f"❌ No images found matching: '{query}'\n\n📊 **Statistics:**\n- Processed: {len(images_with_paths)} images\n- Matches: 0\n\n💡 **Tips:**\n- Try more general terms\n- Check if images actually contain what you're looking for\n- Verify image quality and clarity"
            
            # Create result summary
            result_text = f"# 🎯 Query Results\n\n"
            result_text += f"**Search Query:** \"{query}\"\n"
            result_text += f"**Found:** {len(filtered_results)} matches out of {len(images_with_paths)} total images\n"
            result_text += f"**Success Rate:** {(len(filtered_results)/len(images_with_paths)*100):.1f}%\n\n"
            result_text += f"**Model:** Gemma 3n-E4B-it\n"
            result_text += f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            result_text += "---\n\n"
            result_text += "## 📋 Detailed Analysis Results\n\n"
            
            # Extract images and create detailed results
            filtered_images = []
            for i, (image_path, image, explanation) in enumerate(filtered_results, 1):
                filtered_images.append(image)
                filename = os.path.basename(image_path)
                result_text += f"### {i}. {filename}\n"
                result_text += f"**📁 Path:** `{image_path}`\n"
                result_text += f"**🤖 AI Analysis:** {explanation}\n\n"
                result_text += "---\n\n"
            
            # Create image grid for display
            grid_image = self.image_processor.create_image_grid(filtered_images)
            
            return grid_image, result_text
            
        except Exception as e:
            error_msg = f"❌ Error processing image batch: {str(e)}"
            logging.error(error_msg)
            return None, error_msg

class GradioInterface:
    """Professional Gradio interface for Gemma 3n-E4B-it system"""
    
    def __init__(self, analyzer: VisionLanguageAnalyzer):
        self.analyzer = analyzer
        self.config = analyzer.config
    
    def analyze_single_image_interface(self, image, custom_prompt):
        """Interface function for single image analysis"""
        if image is None:
            return "❌ No image uploaded. Please select an image to analyze."
        
        try:
            # Use custom prompt if provided, otherwise use default
            prompt = custom_prompt if custom_prompt and custom_prompt.strip() else None
            result = self.analyzer.analyze_single_image(image, prompt)
            
            # Format the result nicely
            formatted_result = f"# 🤖 Gemma 3n-E4B-it Analysis\n\n"
            formatted_result += f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            formatted_result += f"**Model:** google/gemma-3n-e4b-it\n"
            formatted_result += f"**Image Size:** {image.size[0]}×{image.size[1]} pixels\n"
            formatted_result += f"**Status:** {'✅ Ready' if self.analyzer.system_ready else '❌ Not Ready'}\n\n"
            formatted_result += "---\n\n"
            formatted_result += "### 📝 Analysis Result:\n\n"
            formatted_result += result
            
            return formatted_result
            
        except Exception as e:
            return f"❌ Error analyzing image: {str(e)}\n\nPlease check that the model is properly loaded and try again."
    
    def analyze_batch_interface(self, directory_path, query):
        """Interface function for batch analysis"""
        if not directory_path or not directory_path.strip():
            return None, "❌ Please provide a valid directory path containing images."
        
        if not query or not query.strip():
            return None, "❌ Please provide a search query to filter images."
        
        try:
            grid_image, result_text = self.analyzer.process_image_batch(directory_path, query)
            return grid_image, result_text
        except Exception as e:
            error_msg = f"❌ Error processing batch: {str(e)}"
            return None, error_msg
    
    def get_system_status(self):
        """Get current system status"""
        if self.analyzer.system_ready:
            return "✅ System Ready - Gemma 3n-E4B-it Loaded"
        else:
            return "❌ System Not Ready - Check Model Loading"
    
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface"""
        with gr.Blocks(
            theme=gr.themes.Soft(),
            title="Gemma 3n-E4B-it Vision Analysis",
            css="""
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
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .status-box {
                padding: 15px;
                border-radius: 10px;
                margin: 10px 0;
                text-align: center;
                font-weight: bold;
            }
            .status-ready {
                background-color: #d4edda;
                border: 1px solid #c3e6cb;
                color: #155724;
            }
            .status-error {
                background-color: #f8d7da;
                border: 1px solid #f5c6cb;
                color: #721c24;
            }
            .result-box {
                border: 2px solid #e1e5e9;
                border-radius: 10px;
                padding: 20px;
                margin: 15px 0;
                max-height: 600px;
                overflow-y: auto;
                background-color: #f8f9fa;
            }
            .model-info {
                background-color: #e8f4fd;
                border: 1px solid #bee5eb;
                border-radius: 8px;
                padding: 15px;
                margin: 10px 0;
            }
            """
        ) as interface:
            
            # Header
            gr.HTML(f"""
            <div class="main-header">
                <h1>{self.config.INTERFACE_TITLE}</h1>
                <p><strong>Powered by Google's Gemma 3n-E4B-it Model</strong></p>
                <p>4B Effective Parameters • Multimodal AI • RTX A4000 Optimized</p>
            </div>
            """)
            
            # System Status
            status_html = gr.HTML(
                f'<div class="status-box {"status-ready" if self.analyzer.system_ready else "status-error"}">'
                f'{self.get_system_status()}</div>'
            )
            
            # Model information
            gr.HTML("""
            <div class="model-info">
                <h3>🔬 Model Information</h3>
                <ul>
                    <li><strong>Model:</strong> google/gemma-3n-e4b-it (Instruction-tuned)</li>
                    <li><strong>Capabilities:</strong> Text + Image → Text Generation</li>
                    <li><strong>Context Window:</strong> 32K tokens</li>
                    <li><strong>Languages:</strong> 140+ supported languages</li>
                    <li><strong>Image Support:</strong> 256×256, 512×512, 768×768 pixels</li>
                    <li><strong>Architecture:</strong> MatFormer with selective parameter activation</li>
                </ul>
            </div>
            """)
            
            # Description
            gr.Markdown(self.config.INTERFACE_DESCRIPTION)
            
            with gr.Tabs():
                # Single Image Analysis Tab
                with gr.Tab("🖼️ Single Image Analysis"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("## 📤 Upload Image")
                            single_image_input = gr.Image(
                                type="pil",
                                label="Select Image for Analysis",
                                height=400
                            )
                            
                            single_prompt_input = gr.Textbox(
                                label="Custom Analysis Prompt (Optional)",
                                placeholder="e.g., 'What animals are in this image?', 'Describe the colors and mood', 'Count the objects'",
                                lines=3,
                                info="Leave empty for default detailed description"
                            )
                            
                            single_analyze_btn = gr.Button(
                                "🤖 Analyze with Gemma 3n",
                                variant="primary",
                                size="lg"
                            )
                            
                            # Example prompts
                            gr.Markdown("""
                            ### 💡 Example Prompts:
                            - "What is happening in this image?"
                            - "Describe the emotions and mood"
                            - "List all objects you can see"
                            - "What is the setting or location?"
                            - "Are there any people? What are they doing?"
                            - "What colors dominate this image?"
                            - "Is this image taken indoors or outdoors?"
                            - "What story does this image tell?"
                            """)
                        
                        with gr.Column(scale=1):
                            gr.Markdown("## 📋 Analysis Result")
                            single_result = gr.Markdown(
                                "Upload an image and click 'Analyze with Gemma 3n' to see detailed AI analysis.",
                                elem_classes=["result-box"]
                            )
                
                # Batch Processing Tab
                with gr.Tab("📁 Batch Image Processing & Filtering"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("## ⚙️ Batch Processing Settings")
                            
                            directory_input = gr.Textbox(
                                label="📁 Directory Path",
                                placeholder="C:/Users/YourName/Pictures/MyImages",
                                info="Full path to directory containing images to analyze",
                                lines=1
                            )
                            
                            query_input = gr.Textbox(
                                label="🔍 Search Query",
                                placeholder="e.g., 'cats', 'people wearing glasses', 'red cars', 'outdoor scenes'",
                                info="Describe what you want to find using natural language",
                                lines=2
                            )
                            
                            batch_analyze_btn = gr.Button(
                                "🚀 Process Batch with Gemma 3n",
                                variant="primary",
                                size="lg"
                            )
                            
                            gr.Markdown("""
                            ### 🎯 Query Examples:
                            - **Animals:** "cats", "dogs", "white cats", "animals sleeping"
                            - **People:** "people smiling", "children playing", "people wearing glasses"
                            - **Objects:** "red cars", "flowers", "food on plates", "books"
                            - **Scenes:** "outdoor scenes", "indoor rooms", "night scenes", "beaches"
                            - **Activities:** "people cooking", "sports activities", "celebrations"
                            - **Colors:** "blue objects", "colorful images", "black and white photos"
                            - **Technical:** "blurry images", "close-up shots", "landscape photos"
                            """)
                            
                            gr.Markdown("""
                            ### ⚡ Performance Notes:
                            - Processing time: ~4-8 seconds per image
                            - Memory usage optimized for RTX A4000
                            - Supports JPG, PNG, BMP, TIFF, WebP formats
                            - Images automatically resized for optimal processing
                            - Progress shown in console logs
                            """)
                        
                        with gr.Column(scale=2):
                            gr.Markdown("## 📊 Filtered Results")
                            
                            batch_result_images = gr.Image(
                                label="🖼️ Matching Images (Grid View)",
                                height=400,
                                interactive=False
                            )
                            
                            batch_result_text = gr.Markdown(
                                "Configure settings and click 'Process Batch' to see filtered results with detailed explanations.",
                                elem_classes=["result-box"]
                            )
            
            # Event handlers
            single_analyze_btn.click(
                fn=self.analyze_single_image_interface,
                inputs=[single_image_input, single_prompt_input],
                outputs=[single_result]
            )
            
            batch_analyze_btn.click(
                fn=self.analyze_batch_interface,
                inputs=[directory_input, query_input],
                outputs=[batch_result_images, batch_result_text]
            )
            
            # Footer
            gr.HTML("""
            <div style="text-align: center; margin-top: 30px; padding: 20px; background-color: #f8f9fa; border-radius: 10px; border: 1px solid #dee2e6;">
                <h4>🚀 Gemma 3n-E4B-it Vision Analysis System v1.1</h4>
                <p><strong>Model:</strong> google/gemma-3n-e4b-it • <strong>Hardware:</strong> RTX A4000 Optimized • <strong>Mode:</strong> Offline Capable</p>
                <p style="color: #6c757d; font-size: 0.9em;">
                    <em>⚠️ This system provides AI-generated analysis using Google's Gemma 3n model. 
                    Results should be verified for critical applications.</em>
                </p>
                <p style="color: #6c757d; font-size: 0.85em;">
                    Requirements: transformers ≥ 4.53.0 • PyTorch ≥ 2.0 • CUDA 11.8+ • HuggingFace License Accepted
                </p>
            </div>
            """)
        
        return interface

def main():
    """Main application entry point with comprehensive error handling"""
    print("🚀 Initializing Gemma 3n-E4B-it Vision Analysis System...")
    print("=" * 80)
    
    try:
        # System requirements check
        print("🔧 System Requirements Check:")
        print(f"   Python: {sys.version}")
        
        # Check PyTorch
        try:
            print(f"   PyTorch: {torch.__version__}")
            print(f"   CUDA Available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"   GPU: {torch.cuda.get_device_name()}")
                print(f"   CUDA Version: {torch.version.cuda}")
                print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        except Exception as e:
            print(f"   ❌ PyTorch issue: {e}")
        
        # Check transformers version
        print(f"   Transformers: {transformers_version}")
        if transformers_version < "4.53.0":
            print("   ⚠️  WARNING: Gemma 3n requires transformers >= 4.53.0")
            print("   Install with: pip install transformers>=4.53.0")
        
        print()
        
        # Initialize analyzer
        analyzer = VisionLanguageAnalyzer()
        
        if not analyzer.system_ready:
            print("❌ System initialization failed. Please check the error messages above.")
            print("\nCommon solutions:")
            print("1. pip install transformers>=4.53.0")
            print("2. Accept Gemma license at: https://huggingface.co/google/gemma-3n-e4b-it")
            print("3. Check model files or internet connection")
            return
        
        # Create interface
        interface_manager = GradioInterface(analyzer)
        app = interface_manager.create_interface()
        
        # Launch application
        print("\n✅ System ready! Starting web interface...")
        print("🌐 Access the application at: http://localhost:7860")
        print("📱 The interface will open automatically in your browser")
        print("\n" + "=" * 80)
        print("🎯 Features available:")
        print("  📸 Single Image Analysis - Upload and analyze with custom prompts")
        print("  📁 Batch Processing - Filter multiple images with natural language")
        print("  🤖 Gemma 3n-E4B-it - Latest Google multimodal AI model")
        print("  ⚡ RTX A4000 Optimized - Efficient GPU memory usage")
        print("  🔧 Error Handling - Robust error recovery and reporting")
        print("=" * 80)
        print("Press Ctrl+C to stop the application")
        
        app.launch(
            server_name="localhost",
            server_port=7860,
            share=False,
            inbrowser=True,
            show_error=True,
            show_tips=True
        )
        
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Critical error starting application: {e}")
        print("\nTroubleshooting steps:")
        print("1. Check transformers version: pip install transformers>=4.53.0")
        print("2. Check PyTorch CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("3. Accept Gemma license on HuggingFace")
        print("4. Verify CUDA drivers and GPU availability")

if __name__ == "__main__":
    # Ensure required directories exist
    try:
        os.makedirs("models", exist_ok=True)
        os.makedirs("models/gemma-3n-e4b-it", exist_ok=True)
        os.makedirs("temp", exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create directories: {e}")
    
    main()