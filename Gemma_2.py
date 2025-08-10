#!/usr/bin/env python3
"""
Professional Multi-Model Vision Analysis System
===============================================

A comprehensive offline vision analysis system with:
1. Gemma 3n-E4B-it for detailed image analysis and description
2. YOLO12x for advanced object detection and visualization

Designed for RTX A4000 (16GB VRAM) running on Windows.

Features:
- Model selection (Gemma 3n-E4B-it / YOLO12x)
- Single image and batch processing for both models
- Professional object detection with YOLO12x
- Advanced vision-language analysis with Gemma 3n
- Modern, professional Gradio interface
- Local model loading (offline capable)
- GPU optimization for RTX A4000

Author: AI Assistant
Version: 2.0 (Multi-Model System)
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

# Core libraries with robust error handling
import numpy as np
import cv2

# PIL for image processing
try:
    from PIL import Image, ImageEnhance, ImageDraw, ImageFont
except ImportError:
    print("Error: Pillow (PIL) is required. Install with: pip install Pillow")
    sys.exit(1)

# PyTorch with version check
try:
    import torch
    if torch.__version__ < "2.0.0":
        print(f"Warning: PyTorch {torch.__version__} detected. Recommended: >= 2.0.0")
except ImportError:
    print("Error: PyTorch is required. Install with: pip install torch")
    sys.exit(1)

# Transformers for Gemma 3n
try:
    from transformers import (
        AutoProcessor, 
        Gemma3nForConditionalGeneration,
        __version__ as transformers_version
    )
    
    if transformers_version < "4.53.0":
        print(f"Warning: transformers version {transformers_version} < 4.53.0")
        print("Gemma 3n requires transformers >= 4.53.0 for optimal performance")
        
except ImportError as e:
    print(f"Warning: Transformers not available: {e}")
    print("Gemma 3n features will be disabled. Install with: pip install 'transformers>=4.53.0'")
    transformers_version = "0.0.0"

# YOLO for object detection
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("Warning: Ultralytics YOLO not available. Install with: pip install ultralytics")
    YOLO_AVAILABLE = False

# Gradio with version check
try:
    import gradio as gr
    gradio_version = getattr(gr, '__version__', '0.0.0')
    print(f"✅ Gradio version: {gradio_version}")
except ImportError:
    print("❌ Error: Gradio not installed. Please install: pip install gradio")
    sys.exit(1)

# Optional dependencies
try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable, desc="", **kwargs):
            self.iterable = iterable
        def __iter__(self):
            return iter(self.iterable)

# Suppress warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Config:
    """Configuration for both Gemma 3n and YOLO12x models"""
    
    # Model paths
    GEMMA_MODEL_NAME = "google/gemma-3n-e4b-it"
    GEMMA_MODEL_PATH = os.path.join("models", "gemma-3n-e4b-it")
    YOLO_MODEL_PATH = os.path.join("models", "yolo12x.pt")  # Local YOLO12x model
    
    # System settings
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF', '.WEBP']
    
    # Gemma 3n settings
    GEMMA_MAX_NEW_TOKENS = 512
    GEMMA_TEMPERATURE = 0.8
    GEMMA_IMAGE_SIZES = [256, 512, 768]
    GEMMA_DEFAULT_SIZE = 512
    
    # YOLO12x settings
    YOLO_CONFIDENCE = 0.25      # Detection confidence threshold
    YOLO_IOU = 0.45            # IoU threshold for NMS
    YOLO_MAX_DETECTIONS = 1000 # Maximum detections per image
    YOLO_LINE_WIDTH = 2        # Bounding box line width
    YOLO_FONT_SIZE = 12        # Label font size
    
    # UI settings
    INTERFACE_TITLE = "🔍 Multi-Model Vision Analysis System"
    INTERFACE_DESCRIPTION = """
    ## Advanced AI Vision Analysis with Model Selection
    
    Choose between two powerful AI models for different vision tasks:
    
    **🤖 Gemma 3n-E4B-it**: Detailed image analysis and natural language descriptions
    **🎯 YOLO12x**: Advanced object detection with bounding boxes and labels
    
    Both models support single image analysis and batch processing with professional results.
    """

class ImageProcessor:
    """Universal image processor for both models"""
    
    def __init__(self):
        self.gemma_sizes = Config.GEMMA_IMAGE_SIZES
        
    def is_valid_image_file(self, file_path: str) -> bool:
        """Check if file is a valid image"""
        try:
            _, ext = os.path.splitext(file_path.lower())
            if ext not in [fmt.lower() for fmt in Config.SUPPORTED_FORMATS]:
                return False
            
            if not os.path.isfile(file_path):
                return False
                
            with Image.open(file_path) as img:
                img.verify()
            return True
        except Exception:
            return False
    
    def prepare_image_for_gemma(self, image: Image.Image, target_size: int = 512) -> Image.Image:
        """Prepare image for Gemma 3n processing"""
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            width, height = image.size
            max_dim = max(width, height)
            
            # Choose appropriate size for Gemma 3n
            if max_dim <= self.gemma_sizes[0]:
                target_size = self.gemma_sizes[0]
            elif max_dim <= self.gemma_sizes[1]:
                target_size = self.gemma_sizes[1]
            else:
                target_size = self.gemma_sizes[2]
            
            # Resize maintaining aspect ratio, then pad to square
            if width > height:
                new_width = target_size
                new_height = int((height * target_size) / width)
            else:
                new_height = target_size
                new_width = int((width * target_size) / height)
            
            new_width = max(1, new_width)
            new_height = max(1, new_height)
            
            image = image.resize((new_width, new_height), Image.LANCZOS)
            
            # Create square canvas and center
            canvas = Image.new('RGB', (target_size, target_size), (255, 255, 255))
            x_offset = (target_size - new_width) // 2
            y_offset = (target_size - new_height) // 2
            canvas.paste(image, (max(0, x_offset), max(0, y_offset)))
            
            return canvas
        except Exception as e:
            logging.error(f"Error preparing image for Gemma: {e}")
            return image
    
    def prepare_image_for_yolo(self, image: Image.Image) -> Image.Image:
        """Prepare image for YOLO processing (keep original size)"""
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            logging.error(f"Error preparing image for YOLO: {e}")
            return image
    
    def load_image(self, image_path: str) -> Optional[Image.Image]:
        """Load and validate image"""
        try:
            if not self.is_valid_image_file(image_path):
                return None
            
            with Image.open(image_path) as img:
                return img.copy().convert('RGB')
        except Exception as e:
            logging.error(f"Error loading image {image_path}: {e}")
            return None
    
    def load_images_from_directory(self, directory_path: str) -> List[Tuple[str, Image.Image]]:
        """Load all images from directory"""
        images = []
        try:
            directory = Path(directory_path)
            if not directory.exists() or not directory.is_dir():
                logging.error(f"Invalid directory: {directory_path}")
                return images
            
            # Find all image files
            for ext in Config.SUPPORTED_FORMATS:
                for image_path in directory.glob(f"*{ext}"):
                    image = self.load_image(str(image_path))
                    if image:
                        images.append((str(image_path), image))
            
            logging.info(f"Loaded {len(images)} images from {directory_path}")
            return images
        except Exception as e:
            logging.error(f"Error loading images from directory: {e}")
            return images
    
    def create_image_grid(self, images: List[Image.Image], max_cols: int = 4) -> Optional[Image.Image]:
        """Create grid of images for display"""
        try:
            if not images:
                return None
            
            valid_images = [img for img in images if img is not None]
            if not valid_images:
                return None
            
            num_images = len(valid_images)
            cols = min(max_cols, num_images)
            rows = (num_images + cols - 1) // cols
            
            cell_size = 200
            grid_width = cols * cell_size
            grid_height = rows * cell_size
            grid_image = Image.new('RGB', (grid_width, grid_height), (240, 240, 240))
            
            for idx, image in enumerate(valid_images):
                try:
                    row = idx // cols
                    col = idx % cols
                    x = col * cell_size
                    y = row * cell_size
                    
                    if image.size[0] > 0 and image.size[1] > 0:
                        resized_image = image.resize((cell_size - 4, cell_size - 4), Image.LANCZOS)
                        grid_image.paste(resized_image, (x + 2, y + 2))
                except Exception as e:
                    logging.warning(f"Error placing image {idx}: {e}")
                    continue
            
            return grid_image
        except Exception as e:
            logging.error(f"Error creating image grid: {e}")
            return None

class GemmaModel:
    """Gemma 3n-E4B-it model wrapper"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.processor = None
        self.loaded = False
        
    def load_model(self):
        """Load Gemma 3n model"""
        if transformers_version < "4.53.0":
            raise ValueError("Gemma 3n requires transformers >= 4.53.0")
        
        logging.info("Loading Gemma 3n-E4B-it model...")
        
        try:
            # Determine model path
            model_path = self.config.GEMMA_MODEL_PATH
            if os.path.exists(model_path) and os.path.isdir(model_path):
                logging.info(f"Loading from local: {model_path}")
                local_files_only = True
            else:
                logging.info(f"Loading from online: {self.config.GEMMA_MODEL_NAME}")
                model_path = self.config.GEMMA_MODEL_NAME
                local_files_only = False
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                local_files_only=local_files_only,
                trust_remote_code=True
            )
            
            # Load model
            self.model = Gemma3nForConditionalGeneration.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=self.config.TORCH_DTYPE,
                local_files_only=local_files_only,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            ).eval()
            
            self.loaded = True
            logging.info("✅ Gemma 3n model loaded successfully")
            
        except Exception as e:
            logging.error(f"Failed to load Gemma 3n: {e}")
            raise
    
    def analyze_image(self, image: Image.Image, prompt: str = None) -> str:
        """Analyze image with Gemma 3n"""
        if not self.loaded:
            return "❌ Gemma 3n model not loaded"
        
        try:
            if prompt is None:
                prompt = "Provide a comprehensive analysis of this image covering: objects, people, setting, mood, colors, composition, activities, and overall narrative. Be thorough and descriptive."
            
            # Create chat messages
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a professional image analyst providing detailed descriptions."}]
                },
                {
                    "role": "user", 
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Process with model
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            
            if hasattr(inputs, 'to'):
                inputs = inputs.to(self.model.device)
            else:
                inputs = {k: v.to(self.model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            input_len = inputs["input_ids"].shape[-1]
            
            # Generate response
            with torch.inference_mode():
                generation = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.GEMMA_MAX_NEW_TOKENS,
                    temperature=self.config.GEMMA_TEMPERATURE,
                    do_sample=True,
                    top_p=0.9,
                    top_k=40,
                    repetition_penalty=1.1,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode response
            if generation.dim() > 1:
                generation = generation[0][input_len:]
            else:
                generation = generation[input_len:]
            
            response = self.processor.decode(generation, skip_special_tokens=True).strip()
            return response if response else "No response generated."
            
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return "❌ GPU out of memory. Try reducing image size."
        except Exception as e:
            logging.error(f"Gemma analysis error: {e}")
            return f"❌ Analysis error: {str(e)}"

class YOLOModel:
    """YOLO12x object detection model wrapper"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.loaded = False
        
        # COCO class names for better labeling
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        # Color palette for different classes
        self.colors = self._generate_colors(len(self.class_names))
    
    def _generate_colors(self, num_classes: int) -> List[Tuple[int, int, int]]:
        """Generate distinct colors for each class"""
        colors = []
        for i in range(num_classes):
            hue = int(360 * i / num_classes)
            # Convert HSV to RGB for better color distribution
            import colorsys
            rgb = colorsys.hsv_to_rgb(hue/360, 0.8, 0.9)
            colors.append(tuple(int(255 * c) for c in rgb))
        return colors
    
    def load_model(self):
        """Load YOLO12x model"""
        if not YOLO_AVAILABLE:
            raise ValueError("Ultralytics YOLO not available. Install with: pip install ultralytics")
        
        logging.info("Loading YOLO12x model...")
        
        try:
            model_path = self.config.YOLO_MODEL_PATH
            
            if os.path.exists(model_path):
                logging.info(f"Loading YOLO12x from local: {model_path}")
                self.model = YOLO(model_path)
            else:
                logging.info("Local YOLO12x not found, downloading YOLOv8x...")
                # Fallback to YOLOv8x if YOLO12x not available
                self.model = YOLO('yolov8x.pt')
                logging.info("Note: Using YOLOv8x as fallback. Place yolo12x.pt in models/ for YOLO12x")
            
            # Set model parameters
            self.model.conf = self.config.YOLO_CONFIDENCE
            self.model.iou = self.config.YOLO_IOU
            self.model.max_det = self.config.YOLO_MAX_DETECTIONS
            
            self.loaded = True
            logging.info("✅ YOLO model loaded successfully")
            
        except Exception as e:
            logging.error(f"Failed to load YOLO model: {e}")
            raise
    
    def detect_objects(self, image: Image.Image) -> Tuple[Image.Image, str]:
        """Detect objects and return annotated image with details"""
        if not self.loaded:
            return image, "❌ YOLO model not loaded"
        
        try:
            # Convert PIL to numpy array
            img_array = np.array(image)
            
            # Run detection
            results = self.model(img_array, verbose=False)
            
            # Process results
            if len(results) == 0 or len(results[0].boxes) == 0:
                return image, "No objects detected in the image."
            
            # Get detection data
            boxes = results[0].boxes
            annotated_image = image.copy()
            draw = ImageDraw.Draw(annotated_image)
            
            # Try to load a better font
            try:
                font = ImageFont.truetype("arial.ttf", self.config.YOLO_FONT_SIZE)
            except:
                try:
                    font = ImageFont.truetype("Arial.ttf", self.config.YOLO_FONT_SIZE)
                except:
                    font = ImageFont.load_default()
            
            detections = []
            detection_summary = {}
            
            # Draw bounding boxes and labels
            for i, box in enumerate(boxes):
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                
                # Get class name and color
                if class_id < len(self.class_names):
                    class_name = self.class_names[class_id]
                    color = self.colors[class_id % len(self.colors)]
                else:
                    class_name = f"Class_{class_id}"
                    color = (255, 0, 0)  # Red for unknown classes
                
                # Draw bounding box
                draw.rectangle([x1, y1, x2, y2], outline=color, width=self.config.YOLO_LINE_WIDTH)
                
                # Create label
                label = f"{class_name}: {confidence:.2f}"
                
                # Get text size for background
                bbox = draw.textbbox((0, 0), label, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # Draw label background
                draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], fill=color)
                
                # Draw label text
                draw.text((x1 + 2, y1 - text_height - 2), label, fill=(255, 255, 255), font=font)
                
                # Store detection info
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': [float(x1), float(y1), float(x2), float(y2)]
                })
                
                # Update summary
                if class_name not in detection_summary:
                    detection_summary[class_name] = []
                detection_summary[class_name].append(confidence)
            
            # Create detailed text summary
            summary_text = self._create_detection_summary(detection_summary, len(detections))
            
            return annotated_image, summary_text
            
        except Exception as e:
            logging.error(f"YOLO detection error: {e}")
            return image, f"❌ Detection error: {str(e)}"
    
    def _create_detection_summary(self, detection_summary: Dict, total_detections: int) -> str:
        """Create detailed detection summary"""
        summary = f"# 🎯 Object Detection Results\n\n"
        summary += f"**Total Objects Detected:** {total_detections}\n"
        summary += f"**Detection Confidence Threshold:** {self.config.YOLO_CONFIDENCE}\n"
        summary += f"**Model:** YOLO12x Object Detection\n"
        summary += f"**Analysis Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        summary += "---\n\n## 📊 Detected Objects by Category:\n\n"
        
        # Sort by number of detections
        sorted_classes = sorted(detection_summary.items(), key=lambda x: len(x[1]), reverse=True)
        
        for class_name, confidences in sorted_classes:
            count = len(confidences)
            avg_confidence = sum(confidences) / count
            max_confidence = max(confidences)
            min_confidence = min(confidences)
            
            summary += f"### {class_name.title()}\n"
            summary += f"- **Count:** {count} object{'s' if count > 1 else ''}\n"
            summary += f"- **Average Confidence:** {avg_confidence:.2f}\n"
            summary += f"- **Confidence Range:** {min_confidence:.2f} - {max_confidence:.2f}\n\n"
        
        summary += "---\n\n## 📋 Detection Details:\n\n"
        summary += "The image has been analyzed using advanced YOLO12x object detection. "
        summary += "Each detected object is marked with a colored bounding box and confidence score. "
        summary += "Higher confidence scores indicate more certain detections.\n\n"
        
        # Add recommendations based on detection count
        if total_detections == 0:
            summary += "**Note:** No objects detected. Try adjusting the confidence threshold or use a different image."
        elif total_detections > 50:
            summary += "**Note:** High number of detections found. Consider increasing confidence threshold for cleaner results."
        else:
            summary += "**Note:** Detection results look good. Bounding boxes show precise object localization."
        
        return summary

class MultiModelAnalyzer:
    """Main analyzer supporting both Gemma 3n and YOLO12x"""
    
    def __init__(self):
        self.config = Config()
        self.image_processor = ImageProcessor()
        self.gemma_model = GemmaModel(self.config) if transformers_version >= "4.53.0" else None
        self.yolo_model = YOLOModel(self.config) if YOLO_AVAILABLE else None
        self.current_model = None
        
    def load_model(self, model_type: str):
        """Load selected model"""
        try:
            if model_type == "Gemma 3n-E4B-it":
                if self.gemma_model is None:
                    raise ValueError("Gemma 3n not available. Install transformers >= 4.53.0")
                if not self.gemma_model.loaded:
                    self.gemma_model.load_model()
                self.current_model = "gemma"
                return "✅ Gemma 3n-E4B-it loaded successfully"
                
            elif model_type == "YOLO12x":
                if self.yolo_model is None:
                    raise ValueError("YOLO not available. Install ultralytics")
                if not self.yolo_model.loaded:
                    self.yolo_model.load_model()
                self.current_model = "yolo"
                return "✅ YOLO12x loaded successfully"
                
            else:
                return "❌ Unknown model type"
                
        except Exception as e:
            return f"❌ Error loading {model_type}: {str(e)}"
    
    def process_single_image(self, image: Image.Image, prompt_or_settings: str = None) -> Union[str, Tuple[Image.Image, str]]:
        """Process single image with current model"""
        if self.current_model is None:
            return "❌ No model loaded. Please select and load a model first."
        
        try:
            if self.current_model == "gemma":
                # Prepare image for Gemma 3n
                processed_image = self.image_processor.prepare_image_for_gemma(image)
                result = self.gemma_model.analyze_image(processed_image, prompt_or_settings)
                return result
                
            elif self.current_model == "yolo":
                # Prepare image for YOLO
                processed_image = self.image_processor.prepare_image_for_yolo(image)
                annotated_image, summary = self.yolo_model.detect_objects(processed_image)
                return annotated_image, summary
                
        except Exception as e:
            logging.error(f"Error processing image: {e}")
            if self.current_model == "yolo":
                return image, f"❌ Processing error: {str(e)}"
            else:
                return f"❌ Processing error: {str(e)}"
    
    def process_image_batch(self, directory_path: str, query_or_settings: str = None) -> Tuple[Optional[Image.Image], str]:
        """Process batch of images"""
        if self.current_model is None:
            return None, "❌ No model loaded. Please select and load a model first."
        
        try:
            # Load images
            images_with_paths = self.image_processor.load_images_from_directory(directory_path)
            
            if not images_with_paths:
                return None, f"❌ No valid images found in: {directory_path}"
            
            if self.current_model == "gemma":
                return self._process_gemma_batch(images_with_paths, query_or_settings)
            elif self.current_model == "yolo":
                return self._process_yolo_batch(images_with_paths)
                
        except Exception as e:
            return None, f"❌ Batch processing error: {str(e)}"
    
    def _process_gemma_batch(self, images_with_paths: List[Tuple[str, Image.Image]], query: str) -> Tuple[Optional[Image.Image], str]:
        """Process batch with Gemma 3n for filtering"""
        if not query:
            return None, "❌ Please provide a search query for batch filtering"
        
        filter_prompt = f"""Look at this image carefully. I am searching for: "{query}"

Respond in exactly this format:
YES or NO
Brief explanation of your decision"""
        
        filtered_results = []
        
        for image_path, image in tqdm(images_with_paths, desc="Analyzing images"):
            try:
                processed_image = self.image_processor.prepare_image_for_gemma(image)
                response = self.gemma_model.analyze_image(processed_image, filter_prompt)
                
                lines = response.strip().split('\n')
                if lines and lines[0].strip().upper().startswith('YES'):
                    explanation = '\n'.join(lines[1:]).strip() if len(lines) > 1 else "Match found"
                    filtered_results.append((image_path, image, explanation))
                    
            except Exception as e:
                logging.warning(f"Error processing {image_path}: {e}")
                continue
        
        if not filtered_results:
            return None, f"❌ No images found matching: '{query}'"
        
        # Create results
        filtered_images = [img for _, img, _ in filtered_results]
        grid_image = self.image_processor.create_image_grid(filtered_images)
        
        # Create summary
        result_text = f"# 🎯 Batch Analysis Results\n\n"
        result_text += f"**Query:** \"{query}\"\n"
        result_text += f"**Found:** {len(filtered_results)} matches out of {len(images_with_paths)} images\n\n"
        
        for i, (path, _, explanation) in enumerate(filtered_results, 1):
            result_text += f"### {i}. {os.path.basename(path)}\n"
            result_text += f"**Analysis:** {explanation}\n\n"
        
        return grid_image, result_text
    
    def _process_yolo_batch(self, images_with_paths: List[Tuple[str, Image.Image]]) -> Tuple[Optional[Image.Image], str]:
        """Process batch with YOLO for object detection"""
        processed_images = []
        all_detections = {}
        total_objects = 0
        
        for image_path, image in tqdm(images_with_paths, desc="Detecting objects"):
            try:
                processed_image = self.image_processor.prepare_image_for_yolo(image)
                annotated_image, summary = self.yolo_model.detect_objects(processed_image)
                processed_images.append(annotated_image)
                
                # Extract detection count for summary
                if "Total Objects Detected:" in summary:
                    count_line = [line for line in summary.split('\n') if 'Total Objects Detected:' in line][0]
                    count = int(count_line.split(':')[1].strip())
                    total_objects += count
                    all_detections[os.path.basename(image_path)] = count
                    
            except Exception as e:
                logging.warning(f"Error processing {image_path}: {e}")
                processed_images.append(image)  # Add original if processing fails
                continue
        
        # Create image grid
        grid_image = self.image_processor.create_image_grid(processed_images)
        
        # Create batch summary
        result_text = f"# 🎯 Batch Object Detection Results\n\n"
        result_text += f"**Total Images Processed:** {len(images_with_paths)}\n"
        result_text += f"**Total Objects Detected:** {total_objects}\n"
        result_text += f"**Average Objects per Image:** {total_objects/len(images_with_paths):.1f}\n\n"
        
        result_text += "## 📊 Detection Summary by Image:\n\n"
        for filename, count in all_detections.items():
            result_text += f"- **{filename}:** {count} objects\n"
        
        result_text += "\n**Note:** Each image in the grid shows detected objects with bounding boxes and labels."
        
        return grid_image, result_text

class ModernGradioInterface:
    """Modern, professional Gradio interface for multi-model system"""
    
    def __init__(self, analyzer: MultiModelAnalyzer):
        self.analyzer = analyzer
        self.config = analyzer.config
        
    def load_model_interface(self, model_choice):
        """Interface function to load selected model"""
        if not model_choice:
            return "❌ Please select a model", gr.update(), gr.update()
        
        result = self.analyzer.load_model(model_choice)
        
        # Update interface visibility based on model
        if "✅" in result:
            if model_choice == "Gemma 3n-E4B-it":
                return result, gr.update(visible=True), gr.update(visible=False)
            else:  # YOLO12x
                return result, gr.update(visible=False), gr.update(visible=True)
        else:
            return result, gr.update(visible=False), gr.update(visible=False)
    
    def process_single_gemma(self, image, prompt):
        """Process single image with Gemma 3n"""
        if image is None:
            return "❌ Please upload an image"
        
        result = self.analyzer.process_single_image(image, prompt)
        
        # Format result
        formatted = f"# 🤖 Gemma 3n-E4B-it Analysis\n\n"
        formatted += f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        formatted += f"**Image Size:** {image.size[0]}×{image.size[1]} pixels\n\n"
        formatted += "---\n\n### 📝 Detailed Analysis:\n\n"
        formatted += result
        
        return formatted
    
    def process_single_yolo(self, image):
        """Process single image with YOLO"""
        if image is None:
            return None, "❌ Please upload an image"
        
        result = self.analyzer.process_single_image(image)
        
        if isinstance(result, tuple):
            annotated_image, summary = result
            return annotated_image, summary
        else:
            return None, result
    
    def process_batch_gemma(self, directory, query):
        """Process batch with Gemma 3n"""
        if not directory or not query:
            return None, "❌ Please provide directory path and search query"
        
        return self.analyzer.process_image_batch(directory, query)
    
    def process_batch_yolo(self, directory):
        """Process batch with YOLO"""
        if not directory:
            return None, "❌ Please provide directory path"
        
        return self.analyzer.process_image_batch(directory)
    
    def create_interface(self) -> gr.Blocks:
        """Create modern, professional Gradio interface"""
        
        # Advanced CSS for modern look
        custom_css = """
        .gradio-container {
            max-width: 1600px !important;
            margin: auto !important;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .main-header {
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            color: white;
            padding: 30px;
            border-radius: 20px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        .model-selector {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 25px;
            border-radius: 15px;
            margin: 20px 0;
            border: 2px solid #e1e5e9;
        }
        .status-box {
            padding: 15px;
            border-radius: 12px;
            margin: 15px 0;
            text-align: center;
            font-weight: bold;
            font-size: 14px;
        }
        .status-success {
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            border: 2px solid #28a745;
            color: #155724;
        }
        .status-error {
            background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
            border: 2px solid #dc3545;
            color: #721c24;
        }
        .feature-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            border: 2px solid #e9ecef;
            border-radius: 15px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        }
        .result-container {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border: 2px solid #dee2e6;
            border-radius: 15px;
            padding: 20px;
            margin: 15px 0;
            max-height: 600px;
            overflow-y: auto;
        }
        .tab-content {
            padding: 25px;
        }
        .model-info {
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            border: 2px solid #2196f3;
            border-radius: 12px;
            padding: 20px;
            margin: 15px 0;
        }
        """
        
        with gr.Blocks(
            theme=gr.themes.Soft(),
            title="Multi-Model Vision Analysis System",
            head="<meta name='viewport' content='width=device-width, initial-scale=1'>",
            css=custom_css
        ) as interface:
            
            # Main Header
            gr.HTML(f"""
            <div class="main-header">
                <h1>🔍 Professional Multi-Model Vision Analysis System</h1>
                <p style="font-size: 18px; margin: 10px 0;"><strong>Choose Your AI Model for Advanced Vision Tasks</strong></p>
                <p style="font-size: 14px; opacity: 0.9;">Gemma 3n-E4B-it for Detailed Analysis • YOLO12x for Object Detection • RTX A4000 Optimized</p>
            </div>
            """)
            
            # Model Selection Section
            with gr.Row():
                gr.HTML('<div class="model-selector">')
                with gr.Column():
                    gr.Markdown("## 🎯 Select AI Model")
                    
                    model_choice = gr.Dropdown(
                        choices=["Gemma 3n-E4B-it", "YOLO12x"],
                        label="Choose AI Model",
                        info="Select the model for your vision analysis task",
                        scale=2
                    )
                    
                    load_model_btn = gr.Button(
                        "🚀 Load Selected Model",
                        variant="primary",
                        size="lg"
                    )
                    
                    model_status = gr.HTML('<div class="status-box">Select and load a model to begin</div>')
                gr.HTML('</div>')
            
            # Model Information Cards
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("""
                    <div class="model-info">
                        <h3>🤖 Gemma 3n-E4B-it</h3>
                        <p><strong>Best for:</strong> Detailed image analysis, natural language descriptions, content understanding</p>
                        <ul>
                            <li>4B effective parameters (8B total)</li>
                            <li>Multimodal vision-language model</li>
                            <li>32K context window</li>
                            <li>140+ language support</li>
                            <li>Advanced image understanding</li>
                        </ul>
                    </div>
                    """)
                
                with gr.Column(scale=1):
                    gr.HTML("""
                    <div class="model-info">
                        <h3>🎯 YOLO12x</h3>
                        <p><strong>Best for:</strong> Object detection, bounding boxes, real-time analysis, counting objects</p>
                        <ul>
                            <li>State-of-the-art object detection</li>
                            <li>80+ object classes (COCO dataset)</li>
                            <li>Precise bounding box localization</li>
                            <li>Confidence scoring</li>
                            <li>Real-time performance</li>
                        </ul>
                    </div>
                    """)
            
            # Description
            gr.Markdown(self.config.INTERFACE_DESCRIPTION)
            
            # Gemma 3n Interface (Initially Hidden)
            with gr.Group(visible=False) as gemma_interface:
                gr.HTML('<div class="feature-card">')
                gr.Markdown("# 🤖 Gemma 3n-E4B-it: Advanced Vision-Language Analysis")
                
                with gr.Tabs():
                    with gr.Tab("🖼️ Single Image Analysis"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("### 📤 Upload Image")
                                gemma_single_image = gr.Image(
                                    type="pil",
                                    label="Upload Image for Analysis",
                                    height=400,
                                    sources=["upload", "clipboard"]
                                )
                                
                                gemma_prompt = gr.Textbox(
                                    label="Custom Analysis Prompt (Optional)",
                                    placeholder="e.g., 'Analyze the emotions and artistic elements', 'Describe every detail you can see'",
                                    lines=3,
                                    info="Leave empty for comprehensive default analysis"
                                )
                                
                                gemma_single_btn = gr.Button(
                                    "🤖 Analyze with Gemma 3n",
                                    variant="primary",
                                    size="lg"
                                )
                            
                            with gr.Column(scale=1):
                                gr.Markdown("### 📋 Detailed Analysis")
                                gemma_single_result = gr.Markdown(
                                    "Upload an image to see comprehensive AI analysis",
                                    elem_classes=["result-container"]
                                )
                    
                    with gr.Tab("📁 Batch Image Filtering"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("### ⚙️ Batch Settings")
                                gemma_directory = gr.Textbox(
                                    label="📁 Directory Path",
                                    placeholder="C:\\Users\\YourName\\Pictures\\MyImages",
                                    info="Path to folder containing images"
                                )
                                
                                gemma_query = gr.Textbox(
                                    label="🔍 Search Query",
                                    placeholder="e.g., 'people smiling', 'outdoor nature scenes', 'red objects'",
                                    info="Describe what to find using natural language"
                                )
                                
                                gemma_batch_btn = gr.Button(
                                    "🔍 Filter Images with Gemma 3n",
                                    variant="primary",
                                    size="lg"
                                )
                            
                            with gr.Column(scale=2):
                                gr.Markdown("### 📊 Filtered Results")
                                gemma_batch_images = gr.Image(
                                    label="Matching Images",
                                    height=400
                                )
                                gemma_batch_text = gr.Markdown(
                                    "Configure settings to filter images by content",
                                    elem_classes=["result-container"]
                                )
                gr.HTML('</div>')
            
            # YOLO Interface (Initially Hidden)
            with gr.Group(visible=False) as yolo_interface:
                gr.HTML('<div class="feature-card">')
                gr.Markdown("# 🎯 YOLO12x: Advanced Object Detection")
                
                with gr.Tabs():
                    with gr.Tab("🖼️ Single Image Detection"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("### 📤 Upload Image")
                                yolo_single_image = gr.Image(
                                    type="pil",
                                    label="Upload Image for Object Detection",
                                    height=400,
                                    sources=["upload", "clipboard"]
                                )
                                
                                yolo_single_btn = gr.Button(
                                    "🎯 Detect Objects with YOLO12x",
                                    variant="primary",
                                    size="lg"
                                )
                                
                                gr.Markdown("""
                                ### 🎯 Detection Features:
                                - **80+ Object Classes**: People, vehicles, animals, household items
                                - **Precise Localization**: Accurate bounding boxes
                                - **Confidence Scores**: Reliability indicators
                                - **Color-coded Labels**: Easy visual identification
                                - **Professional Annotations**: Publication-ready results
                                """)
                            
                            with gr.Column(scale=1):
                                gr.Markdown("### 🖼️ Detected Objects")
                                yolo_single_result_image = gr.Image(
                                    label="Annotated Image with Detections",
                                    height=400
                                )
                                
                                gr.Markdown("### 📊 Detection Summary")
                                yolo_single_result_text = gr.Markdown(
                                    "Upload an image to see object detection results",
                                    elem_classes=["result-container"]
                                )
                    
                    with gr.Tab("📁 Batch Object Detection"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("### ⚙️ Batch Settings")
                                yolo_directory = gr.Textbox(
                                    label="📁 Directory Path", 
                                    placeholder="C:\\Users\\YourName\\Pictures\\MyImages",
                                    info="Path to folder containing images for batch detection"
                                )
                                
                                yolo_batch_btn = gr.Button(
                                    "🎯 Detect Objects in All Images",
                                    variant="primary",
                                    size="lg"
                                )
                                
                                gr.Markdown("""
                                ### 📊 Batch Detection Features:
                                - **Mass Processing**: Analyze entire folders
                                - **Statistical Summary**: Object counts and averages
                                - **Visual Grid**: All results in one view
                                - **Detailed Reports**: Per-image breakdowns
                                - **Export Ready**: Professional documentation
                                """)
                            
                            with gr.Column(scale=2):
                                gr.Markdown("### 📊 Batch Detection Results")
                                yolo_batch_images = gr.Image(
                                    label="All Images with Detections",
                                    height=400
                                )
                                yolo_batch_text = gr.Markdown(
                                    "Specify directory to detect objects in all images",
                                    elem_classes=["result-container"]
                                )
                gr.HTML('</div>')
            
            # Event Handlers
            load_model_btn.click(
                fn=self.load_model_interface,
                inputs=[model_choice],
                outputs=[model_status, gemma_interface, yolo_interface]
            )
            
            # Gemma 3n event handlers
            gemma_single_btn.click(
                fn=self.process_single_gemma,
                inputs=[gemma_single_image, gemma_prompt],
                outputs=[gemma_single_result]
            )
            
            gemma_batch_btn.click(
                fn=self.process_batch_gemma,
                inputs=[gemma_directory, gemma_query],
                outputs=[gemma_batch_images, gemma_batch_text]
            )
            
            # YOLO event handlers
            yolo_single_btn.click(
                fn=self.process_single_yolo,
                inputs=[yolo_single_image],
                outputs=[yolo_single_result_image, yolo_single_result_text]
            )
            
            yolo_batch_btn.click(
                fn=self.process_batch_yolo,
                inputs=[yolo_directory],
                outputs=[yolo_batch_images, yolo_batch_text]
            )
            
            # Footer
            gr.HTML("""
            <div style="text-align: center; margin-top: 40px; padding: 25px; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 15px; border: 2px solid #dee2e6;">
                <h4>🚀 Multi-Model Vision Analysis System v2.0</h4>
                <p><strong>Models:</strong> Gemma 3n-E4B-it + YOLO12x • <strong>Hardware:</strong> RTX A4000 Optimized • <strong>Mode:</strong> Professional Offline System</p>
                <p style="color: #6c757d; font-size: 0.9em;">
                    <em>⚠️ Advanced AI system for professional vision analysis. Results should be verified for critical applications.</em>
                </p>
                <p style="color: #6c757d; font-size: 0.85em;">
                    <strong>Requirements:</strong> transformers ≥ 4.53.0 • ultralytics • PyTorch ≥ 2.0 • CUDA 11.8+
                </p>
            </div>
            """)
        
        return interface

def main():
    """Main application entry point"""
    print("🚀 Initializing Multi-Model Vision Analysis System...")
    print("=" * 80)
    
    try:
        # System validation
        print("🔧 System Requirements Check:")
        print(f"   Python: {sys.version.split()[0]}")
        print(f"   PyTorch: {torch.__version__}")
        print(f"   CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name()}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        print(f"   Transformers: {transformers_version} {'✅' if transformers_version >= '4.53.0' else '⚠️'}")
        print(f"   Ultralytics YOLO: {'✅ Available' if YOLO_AVAILABLE else '❌ Not Available'}")
        print(f"   Gradio: {gradio_version}")
        
        # Initialize analyzer
        analyzer = MultiModelAnalyzer()
        
        # Create interface
        interface_manager = ModernGradioInterface(analyzer)
        app = interface_manager.create_interface()
        
        print("\n✅ System ready! Starting multi-model interface...")
        print("🌐 Access: http://localhost:7860")
        print("\n" + "=" * 80)
        print("🎯 Available Models:")
        print("  🤖 Gemma 3n-E4B-it - Advanced vision-language analysis")
        print("  🎯 YOLO12x - Professional object detection")
        print("  📊 Both models support single image and batch processing")
        print("  🎨 Modern, professional Gradio interface")
        print("=" * 80)
        print("Press Ctrl+C to stop")
        
        # Launch application
        app.launch(
            server_name="localhost",
            server_port=7860,
            share=False,
            inbrowser=True,
            show_error=True,
            favicon_path=None,
            ssl_verify=False,
            quiet=False
        )
        
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Install required packages: pip install transformers ultralytics gradio")
        print("2. Verify GPU availability and CUDA installation")
        print("3. Check model file paths and permissions")

if __name__ == "__main__":
    # Create required directories
    try:
        os.makedirs("models", exist_ok=True)
        os.makedirs(os.path.join("models", "gemma-3n-e4b-it"), exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create directories: {e}")
    
    main()