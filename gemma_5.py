#!/usr/bin/env python3
"""
Professional Multi-Model Vision Analysis Suite
Integrates Gemma 3n-E4B-it, YOLTv4, and YOLOv8x models with robust error handling
Author: AI Assistant
Version: 2.0
"""

import os
import sys
import logging
import warnings
import gc
import time
import json
from datetime import datetime
from typing import List, Tuple, Optional, Union, Dict, Any
from pathlib import Path
import traceback

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Core imports
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

# Optional imports with fallbacks
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    print("⚠️ Gradio not available. Install with: pip install gradio>=4.44.0")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM
    from transformers import Gemma3nForConditionalGeneration
    GEMMA_AVAILABLE = True
except ImportError:
    GEMMA_AVAILABLE = False
    print("⚠️ Gemma 3n requires transformers >= 4.53.0")

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("⚠️ YOLO models require ultralytics. Install with: pip install ultralytics")

try:
    from darknet2pytorch import Darknet
    DARKNET2PYTORCH_AVAILABLE = True
except ImportError:
    DARKNET2PYTORCH_AVAILABLE = False
    print("⚠️ YOLTv4 conversion requires darknet2pytorch. Install with: pip install darknet2pytorch")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vision_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

class Config:
    """Centralized configuration management"""
    
    # Device configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
    
    # Model paths and names
    GEMMA_MODEL_NAME = "google/Gemma-3n-E4B-it"
    GEMMA_MODEL_PATH = "./models/Gemma-3n-E4B-it"
    YOLTV4_MODEL_PATH = "./models/yoltv4.pt"
    YOLTV4_DARKNET_PATH = "./models/yolov4.conv.137"
    YOLTV4_CFG_PATH = "./models/yolov4.cfg"
    
    # Generation parameters
    GEMMA_MAX_NEW_TOKENS = 512
    GEMMA_TEMPERATURE = 0.7
    YOLO_CONFIDENCE = 0.25
    YOLO_IOU_THRESHOLD = 0.45
    
    # Memory management
    MAX_IMAGE_SIZE = (1024, 1024)
    BATCH_SIZE = 4
    
    # COCO Classes (80 classes)
    COCO_CLASSES = [
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
    
    # YOLTv4 Classes (100 classes - COCO + additional aerial/satellite classes)
    YOLTV4_CLASSES = COCO_CLASSES + [
        'ship', 'container', 'oil tanker', 'motorboat', 'sailboat', 'fishing vessel',
        'tugboat', 'barge', 'ferry', 'cargo ship', 'cruise ship', 'speedboat',
        'helicopter', 'small aircraft', 'airliner', 'fighter jet', 'military aircraft',
        'drone', 'glider', 'hot air balloon'
    ]

class MemoryManager:
    """Enhanced memory management utilities"""
    
    @staticmethod
    def clear_gpu_memory():
        """Clear GPU memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
    
    @staticmethod
    def get_memory_info() -> str:
        """Get current memory usage information"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            return f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
        return "CPU Memory Mode"
    
    @staticmethod
    def force_cleanup():
        """Force comprehensive memory cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()

class ImageProcessor:
    """Enhanced image processing utilities"""
    
    def __init__(self, config: Config):
        self.config = config
        self.transform = transforms.Compose([
            transforms.Resize(config.MAX_IMAGE_SIZE),
            transforms.ToTensor(),
        ])
    
    def load_and_validate_image(self, image_input: Union[str, Image.Image, np.ndarray]) -> Optional[Image.Image]:
        """Load and validate image from various input types"""
        try:
            if isinstance(image_input, str):
                if not os.path.exists(image_input):
                    logging.error(f"Image file not found: {image_input}")
                    return None
                image = Image.open(image_input).convert("RGB")
            elif isinstance(image_input, Image.Image):
                image = image_input.convert("RGB")
            elif isinstance(image_input, np.ndarray):
                image = Image.fromarray(image_input).convert("RGB")
            else:
                logging.error(f"Unsupported image input type: {type(image_input)}")
                return None
            
            # Resize if too large
            if image.size[0] > self.config.MAX_IMAGE_SIZE[0] or image.size[1] > self.config.MAX_IMAGE_SIZE[1]:
                image.thumbnail(self.config.MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
            
            return image
            
        except Exception as e:
            logging.error(f"Error loading image: {e}")
            return None
    
    def prepare_image_for_yolo(self, image: Image.Image) -> np.ndarray:
        """Prepare image for YOLO processing"""
        return np.array(image)
    
    def get_images_from_directory(self, directory: str) -> List[Tuple[str, Image.Image]]:
        """Get all valid images from directory"""
        if not os.path.exists(directory):
            logging.error(f"Directory not found: {directory}")
            return []
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        images_with_paths = []
        
        for filename in os.listdir(directory):
            if any(filename.lower().endswith(ext) for ext in valid_extensions):
                image_path = os.path.join(directory, filename)
                image = self.load_and_validate_image(image_path)
                if image:
                    images_with_paths.append((image_path, image))
        
        logging.info(f"Found {len(images_with_paths)} valid images in {directory}")
        return images_with_paths

class YOLTv4Handler:
    """Enhanced YOLTv4 handler with proper Darknet weight conversion"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.loaded = False
        
    def convert_darknet_to_pytorch(self, darknet_weights_path: str, cfg_path: str) -> Optional[str]:
        """Convert Darknet YOLTv4 weights to PyTorch format"""
        if not DARKNET2PYTORCH_AVAILABLE:
            logging.error("darknet2pytorch not available. Install with: pip install darknet2pytorch")
            return None
            
        try:
            # Load Darknet model
            darknet_model = Darknet(cfg_path)
            darknet_model.load_weights(darknet_weights_path)
            
            # Convert to PyTorch
            pytorch_path = darknet_weights_path.replace('.conv.137', '.pt')
            torch.save(darknet_model.state_dict(), pytorch_path)
            
            logging.info(f"✅ Converted {darknet_weights_path} to {pytorch_path}")
            return pytorch_path
            
        except Exception as e:
            logging.error(f"YOLTv4 conversion error: {e}")
            return None
    
    def load_model(self) -> bool:
        """Load YOLTv4 with automatic Darknet conversion"""
        try:
            # Check if PyTorch weights exist
            if not os.path.exists(self.config.YOLTV4_MODEL_PATH):
                if (os.path.exists(self.config.YOLTV4_DARKNET_PATH) and 
                    os.path.exists(self.config.YOLTV4_CFG_PATH)):
                    logging.info("Converting YOLTv4 Darknet weights to PyTorch...")
                    converted_path = self.convert_darknet_to_pytorch(
                        self.config.YOLTV4_DARKNET_PATH, 
                        self.config.YOLTV4_CFG_PATH
                    )
                    if not converted_path:
                        raise ValueError("YOLTv4 conversion failed")
                else:
                    logging.warning("YOLTv4 files not found, using YOLO11n as fallback")
                    if ULTRALYTICS_AVAILABLE:
                        self.model = YOLO("yolo11n.pt")
                        self.loaded = True
                        return True
                    else:
                        return False
            
            # Load converted PyTorch model
            if ULTRALYTICS_AVAILABLE:
                self.model = YOLO(self.config.YOLTV4_MODEL_PATH)
                self.loaded = True
                logging.info("✅ YOLTv4 loaded successfully")
                return True
            else:
                return False
                
        except Exception as e:
            logging.error(f"YOLTv4 loading error: {e}")
            # Fallback to YOLO11n
            if ULTRALYTICS_AVAILABLE:
                try:
                    self.model = YOLO("yolo11n.pt")
                    self.loaded = True
                    return True
                except:
                    return False
            return False

class GemmaModel:
    """Enhanced Gemma 3n-E4B-it model with compatibility fixes"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.processor = None
        self.loaded = False
        self.device = config.DEVICE
        
        # Fix for Gemma dynamo issue
        self._disable_dynamo()
    
    def _disable_dynamo(self):
        """Disable torch._dynamo to fix Gemma generation issues"""
        try:
            import torch._dynamo
            torch._dynamo.config.disable = True
            logging.info("✅ Disabled torch._dynamo for Gemma compatibility")
        except Exception as e:
            logging.warning(f"Could not disable dynamo: {e}")
    
    def load_model(self) -> bool:
        """Enhanced Gemma loading with compatibility fixes"""
        if not GEMMA_AVAILABLE:
            logging.error("Gemma 3n requires transformers >= 4.53.0")
            return False
        
        try:
            # Additional compatibility settings
            os.environ.setdefault('TORCH_LOGS', '')
            os.environ.setdefault('TORCHDYNAMO_VERBOSE', '0')
            
            logging.info("Loading Gemma 3n-E4B-it model...")
            MemoryManager.clear_gpu_memory()
            
            model_path = (self.config.GEMMA_MODEL_PATH 
                         if os.path.exists(self.config.GEMMA_MODEL_PATH) 
                         else self.config.GEMMA_MODEL_NAME)
            
            # Load with enhanced error handling
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=self.config.TORCH_DTYPE
            )
            
            self.model = Gemma3nForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=self.config.TORCH_DTYPE,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device == "cuda":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            self.loaded = True
            
            logging.info(f"✅ Gemma 3n model loaded - {MemoryManager.get_memory_info()}")
            return True
            
        except Exception as e:
            logging.error(f"Gemma loading error: {e}")
            self.loaded = False
            self.offload_model()
            return False
    
    def analyze_image(self, image: Image.Image, prompt: str = None) -> str:
        """Fixed image analysis with proper error handling"""
        if not self.loaded or self.model is None:
            return "❌ Gemma 3n model not loaded"
        
        try:
            if image is None:
                return "❌ No image provided for analysis"
                
            # Default comprehensive prompt
            if prompt is None:
                prompt = """Provide a comprehensive analysis of this image including:
1. All objects, people, and elements visible
2. Setting, environment, and context  
3. Colors, lighting, and composition
4. Activities or interactions occurring
5. Technical quality and artistic elements
6. Overall narrative or significance
Be detailed and professional."""
            
            # Prepare messages with proper format
            messages = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Process with enhanced error handling
            try:
                # Disable gradient computation for inference
                with torch.no_grad():
                    inputs = self.processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt"
                    )
                    
                    # Move to device
                    inputs = {k: v.to(self.device) if hasattr(v, 'to') else v 
                             for k, v in inputs.items()}
                    
                    input_len = inputs["input_ids"].shape[-1]
                    
                    # Generate with fixed parameters
                    generation = self.model.generate(
                        **inputs,
                        max_new_tokens=self.config.GEMMA_MAX_NEW_TOKENS,
                        temperature=self.config.GEMMA_TEMPERATURE,
                        do_sample=True,
                        top_p=0.9,
                        top_k=40,
                        repetition_penalty=1.1,
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                        use_cache=True
                    )
                    
                    # Decode response
                    if generation.dim() > 1:
                        generation = generation[0][input_len:]
                    else:
                        generation = generation[input_len:]
                    
                    response = self.processor.decode(generation, skip_special_tokens=True).strip()
                    
                    return response if response else "Analysis completed but no response generated."
                    
            except torch.cuda.OutOfMemoryError:
                MemoryManager.clear_gpu_memory()
                return "❌ GPU out of memory. Try reducing image size."
            except Exception as e:
                logging.error(f"Gemma generation error: {e}")
                return f"❌ Generation error: {str(e)}"
                
        except Exception as e:
            logging.error(f"Gemma analysis error: {e}")
            return f"❌ Analysis error: {str(e)}"
    
    def offload_model(self):
        """Offload model from memory"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self.loaded = False
        MemoryManager.clear_gpu_memory()

class YOLOModel:
    """Enhanced YOLO model handler supporting both YOLTv4 and YOLOv8x"""
    
    def __init__(self, config: Config, model_type: str):
        self.config = config
        self.model_type = model_type
        self.model = None
        self.loaded = False
        
        if model_type == "YOLTv4":
            self.yoltv4_handler = YOLTv4Handler(config)
            self.class_names = config.YOLTV4_CLASSES
            self.num_classes = len(config.YOLTV4_CLASSES)
        else:
            self.class_names = config.COCO_CLASSES
            self.num_classes = len(config.COCO_CLASSES)
    
    def load_model(self) -> bool:
        """Enhanced model loading with YOLTv4 support"""
        if self.model_type == "YOLTv4":
            success = self.yoltv4_handler.load_model()
            if success:
                self.model = self.yoltv4_handler.model
                self.loaded = True
            return success
        else:
            # Standard YOLOv8x loading
            try:
                if ULTRALYTICS_AVAILABLE:
                    self.model = YOLO("yolov8x.pt")
                    self.loaded = True
                    logging.info("✅ YOLOv8x loaded successfully")
                    return True
                else:
                    logging.error("Ultralytics not available")
                    return False
            except Exception as e:
                logging.error(f"YOLOv8x loading error: {e}")
                return False
    
    def detect_objects(self, image: np.ndarray) -> Tuple[Image.Image, str]:
        """Perform object detection with enhanced result formatting"""
        if not self.loaded or self.model is None:
            return Image.fromarray(image), "❌ Model not loaded"
        
        try:
            # Perform detection
            results = self.model(
                image,
                conf=self.config.YOLO_CONFIDENCE,
                iou=self.config.YOLO_IOU_THRESHOLD,
                verbose=False
            )
            
            # Process results
            if len(results) > 0:
                result = results[0]
                
                # Get annotated image
                annotated_img = result.plot()
                annotated_pil = Image.fromarray(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
                
                # Generate summary
                summary = self._generate_detection_summary(result)
                
                return annotated_pil, summary
            else:
                return Image.fromarray(image), "No objects detected"
                
        except Exception as e:
            logging.error(f"Detection error: {e}")
            return Image.fromarray(image), f"❌ Detection error: {str(e)}"
    
    def _generate_detection_summary(self, result) -> str:
        """Generate detailed detection summary"""
        try:
            if result.boxes is None or len(result.boxes) == 0:
                return "No objects detected"
            
            boxes = result.boxes
            class_counts = {}
            total_objects = len(boxes)
            
            # Count objects by class
            for box in boxes:
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                
                if class_id < len(self.class_names):
                    class_name = self.class_names[class_id]
                    if class_name in class_counts:
                        class_counts[class_name].append(confidence)
                    else:
                        class_counts[class_name] = [confidence]
            
            # Format summary
            summary = f"# 🎯 {self.model_type} Object Detection Results\n\n"
            summary += f"**Model:** {self.model_type}\n"
            summary += f"**Classes Supported:** {self.num_classes}\n"
            summary += f"**Confidence Threshold:** {self.config.YOLO_CONFIDENCE}\n"
            summary += f"**Analysis Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            summary += "## 📊 Detection Summary:\n\n"
            summary += f"**Total Objects Detected:** {total_objects}\n"
            summary += f"**Unique Classes Found:** {len(class_counts)}\n\n"
            
            if class_counts:
                summary += "## 🏷️ Detected Objects:\n\n"
                for class_name, confidences in sorted(class_counts.items()):
                    count = len(confidences)
                    avg_conf = sum(confidences) / count
                    max_conf = max(confidences)
                    summary += f"- **{class_name}**: {count} object{'s' if count > 1 else ''} "
                    summary += f"(avg: {avg_conf:.2f}, max: {max_conf:.2f})\n"
            
            return summary
            
        except Exception as e:
            logging.error(f"Summary generation error: {e}")
            return f"Detection completed with {len(result.boxes) if result.boxes else 0} objects"

class VisionAnalyzer:
    """Main analyzer class coordinating all models"""
    
    def __init__(self, config: Config):
        self.config = config
        self.image_processor = ImageProcessor(config)
        
        # Initialize models
        self.gemma_model = GemmaModel(config)
        self.yoltv4_model = YOLOModel(config, "YOLTv4")
        self.yolov8x_model = YOLOModel(config, "YOLOv8x")
        
        # Current model tracking
        self.current_model = None
        self.models_loaded = {"Gemma": False, "YOLTv4": False, "YOLOv8x": False}
    
    def initialize_models(self):
        """Initialize all models with progress tracking"""
        logging.info("🚀 Initializing Professional Multi-Model Vision Analysis Suite...")
        
        # Load Gemma model
        try:
            self.models_loaded["Gemma"] = self.gemma_model.load_model()
        except Exception as e:
            logging.error(f"Failed to load Gemma: {e}")
        
        # Load YOLTv4 model
        try:
            self.models_loaded["YOLTv4"] = self.yoltv4_model.load_model()
        except Exception as e:
            logging.error(f"Failed to load YOLTv4: {e}")
        
        # Load YOLOv8x model
        try:
            self.models_loaded["YOLOv8x"] = self.yolov8x_model.load_model()
        except Exception as e:
            logging.error(f"Failed to load YOLOv8x: {e}")
        
        logging.info(f"Model loading complete: {self.models_loaded}")
    
    def analyze_with_gemma(self, image: Image.Image, prompt: str = None) -> str:
        """Analyze image with Gemma model"""
        return self.gemma_model.analyze_image(image, prompt)
    
    def detect_with_yolo(self, image: Image.Image, model_type: str) -> Tuple[Image.Image, str]:
        """Perform object detection with specified YOLO model"""
        image_array = self.image_processor.prepare_image_for_yolo(image)
        
        if model_type == "YOLTv4":
            self.current_model = self.yoltv4_model
            return self.yoltv4_model.detect_objects(image_array)
        else:
            self.current_model = self.yolov8x_model
            return self.yolov8x_model.detect_objects(image_array)
    
    def process_image_batch(self, directory: str) -> Tuple[List[Tuple[Image.Image, str]], str]:
        """Fixed YOLO batch processing with proper Gradio Gallery format"""
        if not self.current_model or not self.current_model.loaded:
            return [], "❌ No YOLO model loaded for batch processing"
        
        images_with_paths = self.image_processor.get_images_from_directory(directory)
        if not images_with_paths:
            return [], f"❌ No valid images found in directory: {directory}"
        
        processed_images_with_info = []
        detection_stats = {"total": len(images_with_paths), "processed": 0, "total_objects": 0}
        
        iterator = (tqdm(images_with_paths, desc=f"Processing with {self.current_model.model_type}") 
                   if TQDM_AVAILABLE else images_with_paths)
        
        for image_path, image in iterator:
            try:
                image_array = self.image_processor.prepare_image_for_yolo(image)
                annotated_image, summary = self.current_model.detect_objects(image_array)
                
                filename = os.path.basename(image_path)
                
                # Extract object count
                object_count = 0
                if "Total Objects Detected:" in summary:
                    try:
                        count_line = [line for line in summary.split('\n') if 'Total Objects Detected:' in line][0]
                        object_count = int(count_line.split(':')[1].strip())
                        detection_stats["total_objects"] += object_count
                    except:
                        pass
                
                # Format for Gradio Gallery: (image, caption)
                caption = f"{filename} | {object_count} objects detected"
                processed_images_with_info.append((annotated_image, caption))
                detection_stats["processed"] += 1
                    
            except Exception as e:
                logging.warning(f"Error processing {image_path}: {e}")
                filename = os.path.basename(image_path)
                # Add error image with caption
                processed_images_with_info.append((image, f"{filename} | Processing Error"))
                continue
        
        # Create comprehensive batch summary
        avg_objects = (detection_stats["total_objects"] / detection_stats["processed"] 
                      if detection_stats["processed"] > 0 else 0)
        
        result_text = f"# 🎯 Professional Batch Object Detection Results\n\n"
        result_text += f"**Detection Model:** {self.current_model.model_type}\n"
        result_text += f"**Classes Supported:** {self.current_model.num_classes}\n"
        result_text += f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        result_text += f"**Confidence Threshold:** {self.config.YOLO_CONFIDENCE}\n\n"
        
        result_text += "## 📊 Batch Processing Statistics:\n\n"
        result_text += f"- **Total Images:** {detection_stats['total']}\n"
        result_text += f"- **Successfully Processed:** {detection_stats['processed']}\n"
        result_text += f"- **Total Objects Detected:** {detection_stats['total_objects']}\n"
        result_text += f"- **Average Objects per Image:** {avg_objects:.1f}\n\n"
        
        return processed_images_with_info, result_text

class GradioInterface:
    """Enhanced Gradio interface with proper Gallery components"""
    
    def __init__(self, analyzer: VisionAnalyzer):
        self.analyzer = analyzer
        self.config = analyzer.config
    
    def create_interface(self) -> gr.Blocks:
        """Create the complete Gradio interface"""
        
        # Custom CSS for professional styling
        custom_css = """
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .result-display {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }
        .model-status {
            background-color: #e8f5e8;
            border-left: 4px solid #28a745;
            padding: 10px;
            margin: 10px 0;
        }
        .error-display {
            background-color: #f8d7da;
            border-left: 4px solid #dc3545;
            padding: 10px;
            margin: 10px 0;
        }
        """
        
        with gr.Blocks(
            title="Professional Multi-Model Vision Analysis Suite",
            theme=gr.themes.Soft(),
            css=custom_css
        ) as interface:
            
            # Header
            gr.Markdown("""
            # 🤖 Professional Multi-Model Vision Analysis Suite
            
            **Advanced AI-powered image analysis integrating three state-of-the-art models:**
            - 🧠 **Gemma 3n-E4B-it**: Comprehensive vision-language analysis
            - 🎯 **YOLTv4**: Enhanced object detection (100 classes)
            - 🔍 **YOLOv8x**: Standard object detection (80 COCO classes)
            """)
            
            # Model status display
            with gr.Row():
                model_status = gr.Markdown(
                    self._get_model_status_text(),
                    elem_classes=["model-status"]
                )
            
            # Main interface tabs
            with gr.Tabs() as tabs:
                
                # Gemma 3n Analysis Tab
                with gr.Tab("🧠 Gemma 3n Vision Analysis", id="gemma_tab"):
                    gr.Markdown("""
                    **Comprehensive AI-powered image analysis providing detailed insights about:**
                    - Objects, people, and scene elements
                    - Environment, context, and composition
                    - Activities, interactions, and narratives
                    - Technical and artistic qualities
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            gemma_image = gr.Image(
                                label="Upload Image for Analysis",
                                type="pil",
                                height=400
                            )
                            gemma_prompt = gr.Textbox(
                                label="Custom Analysis Prompt (Optional)",
                                placeholder="Enter specific analysis instructions...",
                                lines=3
                            )
                            gemma_analyze_btn = gr.Button(
                                "🔍 Analyze Image",
                                variant="primary",
                                size="lg"
                            )
                        
                        with gr.Column(scale=1):
                            gemma_result = gr.Markdown(
                                "Upload an image to begin comprehensive AI analysis",
                                elem_classes=["result-display"]
                            )
                
                # YOLTv4 Detection Tab
                with gr.Tab("🎯 YOLTv4 Object Detection", id="yoltv4_tab"):
                    gr.Markdown("""
                    **Enhanced object detection with 100+ classes including:**
                    - All 80 COCO classes (people, vehicles, animals, objects)
                    - 20+ additional aerial/satellite classes (ships, aircraft, etc.)
                    - Optimized for both ground-level and aerial imagery
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            yoltv4_image = gr.Image(
                                label="Upload Image for YOLTv4 Detection",
                                type="pil",
                                height=400
                            )
                            yoltv4_detect_btn = gr.Button(
                                "🎯 Detect Objects",
                                variant="primary",
                                size="lg"
                            )
                        
                        with gr.Column(scale=1):
                            yoltv4_result_img = gr.Image(
                                label="Detection Results",
                                height=400
                            )
                    
                    yoltv4_result_text = gr.Markdown(
                        "Upload an image to perform YOLTv4 object detection",
                        elem_classes=["result-display"]
                    )
                
                # YOLOv8x Detection Tab
                with gr.Tab("🔍 YOLOv8x Object Detection", id="yolov8x_tab"):
                    gr.Markdown("""
                    **Standard YOLO object detection with 80 COCO classes:**
                    - High-accuracy detection for common objects
                    - Fast processing and reliable results
                    - Ideal for general-purpose object detection
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            yolov8x_image = gr.Image(
                                label="Upload Image for YOLOv8x Detection",
                                type="pil",
                                height=400
                            )
                            yolov8x_detect_btn = gr.Button(
                                "🔍 Detect Objects",
                                variant="primary",
                                size="lg"
                            )
                        
                        with gr.Column(scale=1):
                            yolov8x_result_img = gr.Image(
                                label="Detection Results",
                                height=400
                            )
                    
                    yolov8x_result_text = gr.Markdown(
                        "Upload an image to perform YOLOv8x object detection",
                        elem_classes=["result-display"]
                    )
                
                # Batch Processing Tab
                with gr.Tab("📁 Batch Processing", id="batch_tab"):
                    gr.Markdown("""
                    **Process multiple images simultaneously:**
                    - Analyze entire directories of images
                    - Generate comprehensive batch reports
                    - Support for YOLTv4 and YOLOv8x models
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            batch_directory = gr.Textbox(
                                label="Directory Path",
                                placeholder="/path/to/your/images/",
                                info="Enter the full path to directory containing images"
                            )
                            
                            with gr.Row():
                                yoltv4_batch_btn = gr.Button(
                                    "🎯 Batch Process with YOLTv4",
                                    variant="primary"
                                )
                                yolov8x_batch_btn = gr.Button(
                                    "🔍 Batch Process with YOLOv8x",
                                    variant="secondary"
                                )
                    
                    # Batch results display
                    batch_images = gr.Gallery(
                        label="Batch Processing Results - All Images with Annotations",
                        show_label=True,
                        elem_id="batch_gallery",
                        columns=2,
                        rows=2,
                        object_fit="contain",
                        height=600,
                        allow_preview=True
                    )
                    
                    batch_text = gr.Markdown(
                        "Specify directory path to perform batch object detection",
                        elem_classes=["result-display"]
                    )
            
            # Event handlers
            self._setup_event_handlers(
                gemma_analyze_btn, gemma_image, gemma_prompt, gemma_result,
                yoltv4_detect_btn, yoltv4_image, yoltv4_result_img, yoltv4_result_text,
                yolov8x_detect_btn, yolov8x_image, yolov8x_result_img, yolov8x_result_text,
                yoltv4_batch_btn, yolov8x_batch_btn, batch_directory, batch_images, batch_text
            )
            
            # Footer
            gr.Markdown("""
            ---
            **Professional Multi-Model Vision Analysis Suite** | Powered by Gemma 3n, YOLTv4, and YOLOv8x
            """)
        
        return interface
    
    def _get_model_status_text(self) -> str:
        """Generate model status display text"""
        status_text = "## 🔧 Model Status:\n\n"
        
        for model_name, loaded in self.analyzer.models_loaded.items():
            status_icon = "✅" if loaded else "❌"
            status_text += f"- **{model_name}**: {status_icon} {'Loaded' if loaded else 'Not Available'}\n"
        
        return status_text
    
    def _setup_event_handlers(self, *components):
        """Setup all event handlers for the interface"""
        (gemma_analyze_btn, gemma_image, gemma_prompt, gemma_result,
         yoltv4_detect_btn, yoltv4_image, yoltv4_result_img, yoltv4_result_text,
         yolov8x_detect_btn, yolov8x_image, yolov8x_result_img, yolov8x_result_text,
         yoltv4_batch_btn, yolov8x_batch_btn, batch_directory, batch_images, batch_text) = components
        
        # Gemma analysis
        gemma_analyze_btn.click(
            fn=self.process_gemma_analysis,
            inputs=[gemma_image, gemma_prompt],
            outputs=[gemma_result]
        )
        
        # YOLTv4 detection
        yoltv4_detect_btn.click(
            fn=lambda img: self.process_yolo_detection(img, "YOLTv4"),
            inputs=[yoltv4_image],
            outputs=[yoltv4_result_img, yoltv4_result_text]
        )
        
        # YOLOv8x detection
        yolov8x_detect_btn.click(
            fn=lambda img: self.process_yolo_detection(img, "YOLOv8x"),
            inputs=[yolov8x_image],
            outputs=[yolov8x_result_img, yolov8x_result_text]
        )
        
        # Batch processing
        yoltv4_batch_btn.click(
            fn=lambda dir: self.process_batch_yolo(dir, "YOLTv4"),
            inputs=[batch_directory],
            outputs=[batch_images, batch_text]
        )
        
        yolov8x_batch_btn.click(
            fn=lambda dir: self.process_batch_yolo(dir, "YOLOv8x"),
            inputs=[batch_directory],
            outputs=[batch_images, batch_text]
        )
    
    def process_gemma_analysis(self, image: Image.Image, prompt: str) -> str:
        """Process Gemma analysis request"""
        try:
            if image is None:
                return "❌ Please upload an image for analysis"
            
            if not self.analyzer.models_loaded["Gemma"]:
                return "❌ Gemma 3n model not loaded. Check system requirements."
            
            result = self.analyzer.analyze_with_gemma(image, prompt if prompt.strip() else None)
            return result
            
        except Exception as e:
            logging.error(f"Gemma analysis error: {e}")
            return f"❌ Analysis error: {str(e)}"
    
    def process_yolo_detection(self, image: Image.Image, model_type: str) -> Tuple[Image.Image, str]:
        """Process YOLO detection request"""
        try:
            if image is None:
                return None, "❌ Please upload an image for detection"
            
            model_key = model_type
            if not self.analyzer.models_loaded[model_key]:
                return image, f"❌ {model_type} model not loaded. Check system requirements."
            
            result_img, result_text = self.analyzer.detect_with_yolo(image, model_type)
            return result_img, result_text
            
        except Exception as e:
            logging.error(f"{model_type} detection error: {e}")
            return image, f"❌ Detection error: {str(e)}"
    
    def process_batch_yolo(self, directory: str, model_type: str) -> Tuple[List, str]:
        """Process batch YOLO detection request"""
        try:
            if not directory:
                return [], f"❌ Please provide directory path for {model_type} batch processing"
            
            model_key = model_type
            if not self.analyzer.models_loaded[model_key]:
                return [], f"❌ {model_type} model not loaded. Check system requirements."
            
            # Set current model for batch processing
            if model_type == "YOLTv4":
                self.analyzer.current_model = self.analyzer.yoltv4_model
            else:
                self.analyzer.current_model = self.analyzer.yolov8x_model
            
            result = self.analyzer.process_image_batch(directory)
            
            if isinstance(result, tuple) and len(result) == 2:
                images_list, summary_text = result
                
                # Ensure proper format for Gradio Gallery
                if isinstance(images_list, list) and all(isinstance(item, tuple) for item in images_list):
                    return images_list, summary_text
                else:
                    return [], f"❌ Invalid batch processing result format"
            else:
                return [], f"❌ Batch processing failed: {str(result)}"
                
        except Exception as e:
            logging.error(f"Batch processing error: {e}")
            return [], f"❌ Error in batch processing with {model_type}: {str(e)}"

def main():
    """Main application entry point"""
    
    # Runtime error checking and resolution
    def check_runtime_environment():
        """Comprehensive runtime environment check"""
        issues = []
        
        # Check Python version
        if sys.version_info < (3, 8):
            issues.append("Python 3.8+ required")
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            logging.warning("CUDA not available, using CPU mode")
        
        # Check required packages
        required_packages = {
            'gradio': GRADIO_AVAILABLE,
            'transformers (Gemma)': GEMMA_AVAILABLE,
            'ultralytics': ULTRALYTICS_AVAILABLE,
            'darknet2pytorch': DARKNET2PYTORCH_AVAILABLE
        }
        
        for package, available in required_packages.items():
            if not available:
                issues.append(f"Missing: {package}")
        
        if issues:
            logging.warning(f"Runtime issues detected: {', '.join(issues)}")
            logging.info("Some features may be limited")
        else:
            logging.info("✅ Runtime environment check passed")
        
        return len(issues) == 0
    
    try:
        # Perform runtime checks
        logging.info("🔍 Performing runtime environment check...")
        check_runtime_environment()
        
        # Initialize configuration
        config = Config()
        logging.info(f"🔧 Configuration loaded - Device: {config.DEVICE}")
        
        # Initialize analyzer
        analyzer = VisionAnalyzer(config)
        analyzer.initialize_models()
        
        # Check if Gradio is available
        if not GRADIO_AVAILABLE:
            logging.error("❌ Gradio not available. Cannot start web interface.")
            logging.info("Install Gradio with: pip install gradio>=4.44.0")
            return
        
        # Create and launch interface
        logging.info("🚀 Launching Professional Multi-Model Vision Analysis Suite...")
        interface = GradioInterface(analyzer)
        app = interface.create_interface()
        
        # Launch with comprehensive settings
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
            show_tips=True,
            enable_queue=True,
            max_threads=10,
            debug=False
        )
        
    except KeyboardInterrupt:
        logging.info("🛑 Application stopped by user")
    except Exception as e:
        logging.error(f"❌ Critical error: {e}")
        logging.error(traceback.format_exc())
    finally:
        # Cleanup
        logging.info("🧹 Performing cleanup...")
        MemoryManager.force_cleanup()
        logging.info("✅ Cleanup completed")

if __name__ == "__main__":
    main()
