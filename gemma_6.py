#!/usr/bin/env python3
"""
Offline-Only Professional Vision Analysis Suite
===============================================

Models (all from local paths):
- YOLOv5 (local PyTorch Hub repo + local .pt checkpoint)
- YOLOv8x (Ultralytics, local .pt)
- Optional: Gemma 3n-E4B-it (Transformers, local directory)

Features:
- Fully offline: no downloads, no torch.hub remote fetching
- Professional Gradio UI
- Robust model switching with memory cleanup
- Single-image analysis/detection
- Clear summaries
- GPU-friendly patterns for RTX A4000

Author: AI Assistant
Version: 4.1 (Offline paths)
"""

import os
import sys
import gc
import time
import logging
from datetime import datetime
from typing import Optional, Tuple, List, Dict

import numpy as np
import cv2
from PIL import Image

# =========================
# Global Local Model Paths
# =========================
# Set these to valid local paths before running.
# 1) YOLOv5:
#    - YOLOV5_REPO_DIR should point to a local clone of ultralytics/yolov5 (contains hubconf.py)
#    - YOLOV5_MODEL_PATH should point to a local YOLOv5 .pt checkpoint (trained with YOLOv5 repo)
YOLOV5_REPO_DIR = r"/path/to/local/yolov5_repo"  # e.g., "/data/models/yolov5" (must contain hubconf.py)
YOLOV5_MODEL_PATH = r"/path/to/local/yolov5s.pt" # e.g., "/data/models/yolov5s.pt"

# 2) YOLOv8x:
YOLOV8X_MODEL_PATH = r"/path/to/local/yolov8x.pt" # e.g., "/data/models/yolov8x.pt"

# 3) Gemma 3n (optional):
GEMMA_MODEL_PATH = r"/path/to/local/gemma-3n-e4b-it" # local dir containing model files

# =========================

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Torch
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    print("Error: PyTorch is required. Install with: pip install torch")
    TORCH_AVAILABLE = False
    sys.exit(1)

# Ultralytics (for YOLOv8)
try:
    from ultralytics import YOLO as UltralyticsYOLO
    ULTRALYTICS_AVAILABLE = True
except Exception:
    ULTRALYTICS_AVAILABLE = False

# Gradio
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except Exception:
    print("Error: Gradio is required. Install with: pip install gradio")
    GRADIO_AVAILABLE = False
    sys.exit(1)

# Transformers (optional for Gemma)
try:
    from transformers import AutoProcessor
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

# Optional Gemma class import guarded
GEMMA_AVAILABLE = False
GemmaModelClass = None
if TRANSFORMERS_AVAILABLE:
    try:
        from transformers import Gemma3nForConditionalGeneration as _GemmaClass
        GemmaModelClass = _GemmaClass
        GEMMA_AVAILABLE = True
    except Exception:
        GEMMA_AVAILABLE = False
        GemmaModelClass = None

class Config:
    INTERFACE_TITLE = "🔬 Offline Vision Analysis Suite"
    INTERFACE_DESCRIPTION = """
    All models are loaded strictly from local directories/files.
    Provide valid local paths in the global variables at the top of this script.
    """

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    YOLO_CONFIDENCE = 0.25
    YOLO_IOU = 0.45
    YOLO_MAX_DETECTIONS = 300
    YOLO_LINE_WIDTH = 3
    YOLO_FONT_SIZE = 14

    GEMMA_MAX_NEW_TOKENS = 256
    GEMMA_TEMPERATURE = 0.7

class MemoryManager:
    @staticmethod
    def clear_gpu_memory():
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        except Exception as e:
            logging.warning(f"Memory clearing warning: {e}")
            return False

    @staticmethod
    def get_memory_info():
        try:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                return f"Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
            return "CUDA not available"
        except Exception as e:
            return f"Memory info error: {e}"

class ImageProcessor:
    @staticmethod
    def to_pil_rgb(image: Image.Image) -> Image.Image:
        if image is None:
            raise ValueError("Input image is None")
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    @staticmethod
    def to_numpy_rgb(image: Image.Image) -> np.ndarray:
        image = ImageProcessor.to_pil_rgb(image)
        arr = np.array(image)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        return arr

# -----------------------------------------
# YOLOv5 (Offline via local torch.hub repo)
# -----------------------------------------
class YOLOv5LocalHubModel:
    """
    Offline loading of YOLOv5 using a local torch.hub repo directory and local .pt weights.
    Requirements:
      - YOLOV5_REPO_DIR: local clone of ultralytics/yolov5 (contains hubconf.py)
      - YOLOV5_MODEL_PATH: path to a local YOLOv5 .pt checkpoint
    """
    def __init__(self, conf=0.25, iou=0.45, max_det=300, device=Config.DEVICE):
        self.model = None
        self.device = device
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.names = None

    def _validate_paths(self, repo_dir: str, weights_path: str):
        if not os.path.isdir(repo_dir):
            raise FileNotFoundError(f"YOLOv5 repo dir not found: {repo_dir}")
        hubconf = os.path.join(repo_dir, "hubconf.py")
        if not os.path.isfile(hubconf):
            raise FileNotFoundError(f"hubconf.py not found in repo dir: {repo_dir}")
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"YOLOv5 weights not found: {weights_path}")

    def load(self, repo_dir: str, weights_path: str):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available; cannot load YOLOv5.")
        self._validate_paths(repo_dir, weights_path)

        MemoryManager.clear_gpu_memory()
        logging.info(f"Loading YOLOv5 from local repo '{repo_dir}' with weights '{weights_path}' on {self.device} ...")

        # Use torch.hub with a local repo directory; no internet
        # The local repo must be a proper clone of ultralytics/yolov5 with hubconf.py
        self.model = torch.hub.load(repo_dir, 'custom', path=weights_path, source='local')
        self.model = self.model.autoshape()  # for PIL/np inputs
        if self.device == "cuda":
            self.model.to(self.device)

        # Set inference parameters
        self.model.conf = self.conf
        self.model.iou = self.iou
        self.model.max_det = self.max_det

        # Class names
        if hasattr(self.model, "names"):
            self.names = self.model.names
        logging.info(f"✅ YOLOv5 loaded offline. {MemoryManager.get_memory_info()}")
        return True

    def offload(self):
        try:
            self.model = None
            MemoryManager.clear_gpu_memory()
            logging.info("✅ YOLOv5 offloaded.")
        except Exception as e:
            logging.warning(f"YOLOv5 offload warning: {e}")

    def predict(self, image_pil: Image.Image) -> Tuple[Image.Image, str]:
        if self.model is None:
            return image_pil, "❌ YOLOv5 not loaded."

        img_np = ImageProcessor.to_numpy_rgb(image_pil)
        try:
            results = self.model(img_np, size=640)
        except Exception as e:
            return image_pil, f"❌ Detection failed: {e}"

        # Render annotated image
        try:
            results.render()  # updates results.imgs in-place
            annotated = results.imgs[0]
            if isinstance(annotated, np.ndarray):
                annotated_pil = Image.fromarray(annotated)
            else:
                annotated_pil = annotated
        except Exception as e:
            logging.warning(f"Annotation render error: {e}")
            annotated_pil = image_pil

        # Build summary
        try:
            summary = self._build_summary(results)
        except Exception as e:
            summary = f"Detection completed, but summary failed: {e}"

        return annotated_pil, summary

    def _build_summary(self, results) -> str:
        if results is None or len(results.xyxy) == 0:
            return "No results returned."

        det = results.xyxy[0]
        if det is None or det.shape[0] == 0:
            return "No objects detected."

        names = self.names if self.names is not None else getattr(results, "names", {})

        class_counts: Dict[str, List[float]] = {}
        for row in det.cpu().numpy():
            x1, y1, x2, y2, conf, cls_id = row
            cls_id = int(cls_id)
            cls_name = names.get(cls_id, f"Class_{cls_id}") if isinstance(names, dict) else (
                names[cls_id] if isinstance(names, (list, tuple)) and cls_id < len(names) else f"Class_{cls_id}"
            )
            class_counts.setdefault(cls_name, []).append(float(conf))

        total = int(det.shape[0])
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = []
        lines.append("# 🎯 YOLOv5 Object Detection Analysis (Offline)")
        lines.append(f"Model: YOLOv5 (Local Hub)")
        lines.append(f"Total Objects Detected: {total}")
        lines.append(f"Confidence Threshold: {self.conf:.2f}")
        lines.append(f"IoU Threshold: {self.iou:.2f}")
        lines.append(f"Analysis Timestamp: {ts}")
        lines.append("")
        lines.append("## 📊 Detection Summary by Class:")
        for cls_name, confs in sorted(class_counts.items(), key=lambda kv: len(kv[1]), reverse=True):
            count = len(confs)
            avg_conf = sum(confs) / count
            lines.append(f"- {cls_name}: count={count}, avg_conf={avg_conf:.3f}")
        return "\n".join(lines)

# -----------------------------
# YOLOv8x (Offline, Ultralytics)
# -----------------------------
class YOLOv8LocalModel:
    def __init__(self, conf=0.25, iou=0.45, max_det=300, device=Config.DEVICE):
        self.model = None
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.device = device
        self.names = None

    def load(self, weights_path: str):
        if not ULTRALYTICS_AVAILABLE:
            raise RuntimeError("Ultralytics not available; pip install ultralytics.")
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"YOLOv8 weights not found: {weights_path}")

        MemoryManager.clear_gpu_memory()
        logging.info(f"Loading YOLOv8x from local weights '{weights_path}' on {self.device} ...")
        self.model = UltralyticsYOLO(weights_path)

        try:
            self.names = self.model.model.names
        except Exception:
            self.names = None

        logging.info(f"✅ YOLOv8x loaded offline. {MemoryManager.get_memory_info()}")
        return True

    def offload(self):
        try:
            self.model = None
            MemoryManager.clear_gpu_memory()
            logging.info("✅ YOLOv8x offloaded.")
        except Exception as e:
            logging.warning(f"YOLOv8x offload warning: {e}")

    def predict(self, image_pil: Image.Image) -> Tuple[Image.Image, str]:
        if self.model is None:
            return image_pil, "❌ YOLOv8x not loaded."

        img_np = ImageProcessor.to_numpy_rgb(image_pil)
        try:
            results = self.model.predict(
                source=img_np,
                conf=self.conf,
                iou=self.iou,
                max_det=self.max_det,
                verbose=False
            )
        except Exception as e:
            return image_pil, f"❌ Detection failed: {e}"

        if not results or len(results) == 0:
            return image_pil, "No results returned."

        result = results[0]
        try:
            annotated = result.plot(
                conf=True,
                labels=True,
                boxes=True,
                line_width=Config.YOLO_LINE_WIDTH,
                font_size=Config.YOLO_FONT_SIZE
            )
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            annotated_pil = Image.fromarray(annotated)
        except Exception as e:
            logging.warning(f"Annotation render error: {e}")
            annotated_pil = image_pil

        try:
            summary = self._build_summary(result)
        except Exception as e:
            summary = f"Detection completed, but summary failed: {e}"

        return annotated_pil, summary

    def _build_summary(self, result) -> str:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return "No objects detected."

        names = getattr(result, "names", None)
        if names is None:
            names = self.names

        class_counts: Dict[str, List[float]] = {}
        try:
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
        except Exception as e:
            return f"Failed to parse detections: {e}"

        for conf, cls_id in zip(confs, clss):
            if isinstance(names, dict):
                cls_name = names.get(int(cls_id), f"Class_{int(cls_id)}")
            elif isinstance(names, (list, tuple)) and int(cls_id) < len(names):
                cls_name = names[int(cls_id)]
            else:
                cls_name = f"Class_{int(cls_id)}"
            class_counts.setdefault(cls_name, []).append(float(conf))

        total = len(confs)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = []
        lines.append("# 🎯 YOLOv8x Object Detection Analysis (Offline)")
        lines.append(f"Model: YOLOv8x (Local)")
        lines.append(f"Total Objects Detected: {total}")
        lines.append(f"Confidence Threshold: {self.conf:.2f}")
        lines.append(f"IoU Threshold: {self.iou:.2f}")
        lines.append(f"Analysis Timestamp: {ts}")
        lines.append("")
        lines.append("## 📊 Detection Summary by Class:")
        for cls_name, confs_list in sorted(class_counts.items(), key=lambda kv: len(kv[1]), reverse=True):
            count = len(confs_list)
            avg_conf = sum(confs_list) / count
            lines.append(f"- {cls_name}: count={count}, avg_conf={avg_conf:.3f}")
        return "\n".join(lines)

# ----------------------------
# Optional Gemma (Offline)
# ----------------------------
class GemmaLocalModel:
    def __init__(self, device=Config.DEVICE, dtype=Config.TORCH_DTYPE):
        self.device = device
        self.dtype = dtype
        self.model = None
        self.processor = None
        self.loaded = False

    def load(self, local_dir: str):
        if not (TRANSFORMERS_AVAILABLE and GEMMA_AVAILABLE and GemmaModelClass is not None):
            raise RuntimeError("Gemma not available in this environment.")
        if not os.path.isdir(local_dir):
            raise FileNotFoundError(f"Gemma local directory not found: {local_dir}")

        MemoryManager.clear_gpu_memory()
        logging.info(f"Loading Gemma from local directory '{local_dir}' on {self.device} ...")
        self.processor = AutoProcessor.from_pretrained(local_dir, trust_remote_code=True, local_files_only=True)
        self.model = GemmaModelClass.from_pretrained(
            local_dir,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            local_files_only=True
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        self.loaded = True
        logging.info(f"✅ Gemma loaded offline. {MemoryManager.get_memory_info()}")
        return True

    def offload(self):
        try:
            self.model = None
            self.processor = None
            self.loaded = False
            MemoryManager.clear_gpu_memory()
            logging.info("✅ Gemma offloaded.")
        except Exception as e:
            logging.warning(f"Gemma offload warning: {e}")

    def analyze(self, image_pil: Image.Image, prompt: Optional[str]) -> str:
        if not self.loaded or self.model is None or self.processor is None:
            return "❌ Gemma not loaded."
        if image_pil is None:
            return "❌ No image provided."

        try:
            # Prefer processor(images=..., text=...) when chat template is unavailable offline
            if hasattr(self.processor, "apply_chat_template"):
                messages = [
                    {"role": "system", "content": [{"type": "text", "text": "You are a professional image analyst."}]},
                    {"role": "user", "content": [
                        {"type": "image", "image": ImageProcessor.to_pil_rgb(image_pil)},
                        {"type": "text", "text": prompt or "Describe the image in detail."}
                    ]}
                ]
                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
            else:
                inputs = self.processor(
                    images=ImageProcessor.to_pil_rgb(image_pil),
                    text=prompt or "Describe the image in detail.",
                    return_tensors="pt"
                )

            # Move tensors to device
            if isinstance(inputs, dict):
                for k, v in list(inputs.items()):
                    if hasattr(v, "to"):
                        inputs[k] = v.to(self.model.device)

            with torch.inference_mode():
                generation = self.model.generate(
                    **inputs,
                    max_new_tokens=Config.GEMMA_MAX_NEW_TOKENS,
                    temperature=Config.GEMMA_TEMPERATURE,
                    do_sample=True,
                    top_p=0.9,
                    top_k=40
                )

            tokenizer = getattr(self.processor, "tokenizer", None)
            if tokenizer is None:
                return "❌ Missing tokenizer for decoding."
            text = tokenizer.decode(generation[0], skip_special_tokens=True).strip()
            return text if text else "No response generated."
        except torch.cuda.OutOfMemoryError:
            MemoryManager.clear_gpu_memory()
            return "❌ GPU out of memory during Gemma generation."
        except Exception as e:
            return f"❌ Gemma analysis error: {e}"

# ---------------------------------
# Multi-model Controller and UI
# ---------------------------------
class MultiModelAnalyzer:
    def __init__(self):
        self.processor = ImageProcessor()

        self.yolov5 = YOLOv5LocalHubModel(
            conf=Config.YOLO_CONFIDENCE,
            iou=Config.YOLO_IOU,
            max_det=Config.YOLO_MAX_DETECTIONS,
            device=Config.DEVICE
        )
        self.yolov8x = YOLOv8LocalModel(
            conf=Config.YOLO_CONFIDENCE,
            iou=Config.YOLO_IOU,
            max_det=Config.YOLO_MAX_DETECTIONS,
            device=Config.DEVICE
        )
        self.gemma = GemmaLocalModel() if (TRANSFORMERS_AVAILABLE and GEMMA_AVAILABLE) else None

        self.current = None
        self.current_name = None

    def available_models(self) -> List[str]:
        models = ["YOLOv5 (Local)", "YOLOv8x (Local)"]
        if self.gemma is not None:
            models.append("Gemma 3n-E4B-it (Local)")
        return models

    def offload_all(self):
        try:
            if self.yolov5 is not None:
                self.yolov5.offload()
            if self.yolov8x is not None:
                self.yolov8x.offload()
            if self.gemma is not None:
                self.gemma.offload()
            self.current = None
            self.current_name = None
            MemoryManager.clear_gpu_memory()
        except Exception as e:
            logging.warning(f"Offload warning: {e}")

    def load_model(self, name: str) -> str:
        self.offload_all()
        time.sleep(0.2)
        try:
            if name == "YOLOv5 (Local)":
                self.yolov5.load(YOLOV5_REPO_DIR, YOLOV5_MODEL_PATH)
                self.current = self.yolov5
                self.current_name = name
                return "✅ YOLOv5 loaded (offline)."
            elif name == "YOLOv8x (Local)":
                self.yolov8x.load(YOLOV8X_MODEL_PATH)
                self.current = self.yolov8x
                self.current_name = name
                return "✅ YOLOv8x loaded (offline)."
            elif name == "Gemma 3n-E4B-it (Local)":
                if self.gemma is None:
                    return "❌ Gemma not available in this environment."
                self.gemma.load(GEMMA_MODEL_PATH)
                self.current = self.gemma
                self.current_name = name
                return "✅ Gemma 3n-E4B-it loaded (offline)."
            else:
                return "❌ Unknown model selection."
        except Exception as e:
            self.current = None
            self.current_name = None
            return f"❌ Failed to load {name}: {e}"

    def analyze_single(self, model_name: str, image: Image.Image, prompt: Optional[str]) -> Tuple[Optional[Image.Image], str]:
        if image is None:
            return None, "❌ Please provide an image."

        status = self.load_model(model_name)
        if status.startswith("❌"):
            return None, status

        try:
            if model_name == "YOLOv5 (Local)":
                annotated, summary = self.yolov5.predict(image)
                return annotated, summary
            elif model_name == "YOLOv8x (Local)":
                annotated, summary = self.yolov8x.predict(image)
                return annotated, summary
            elif model_name == "Gemma 3n-E4B-it (Local)":
                text = self.gemma.analyze(image, prompt)
                return image, text
            else:
                return None, "❌ Unsupported model."
        except Exception as e:
            return None, f"❌ Inference error: {e}"

    def build_ui(self):
        with gr.Blocks(title=Config.INTERFACE_TITLE) as demo:
            gr.Markdown(Config.INTERFACE_DESCRIPTION)

            with gr.Row():
                model_choice = gr.Dropdown(choices=self.available_models(), value=self.available_models()[0], label="Model")
                prompt = gr.Textbox(label="Prompt (Gemma only)", value="Provide a comprehensive analysis of this image.", lines=3)

            # Show current configured local paths (read-only)
            gr.Markdown("### Current Local Model Paths")
            gr.Markdown(f"- YOLOv5 repo: {YOLOV5_REPO_DIR}")
            gr.Markdown(f"- YOLOv5 weights: {YOLOV5_MODEL_PATH}")
            gr.Markdown(f"- YOLOv8x weights: {YOLOV8X_MODEL_PATH}")
            gr.Markdown(f"- Gemma directory: {GEMMA_MODEL_PATH}")

            input_image = gr.Image(type="pil", label="Input Image")
            run_btn = gr.Button("Run Analysis/Detection")
            output_image = gr.Image(label="Annotated/Processed Image")
            output_text = gr.Textbox(label="Analysis / Detection Summary", lines=16)

            def _run(model, img, p):
                annotated, text = self.analyze_single(model, img, p)
                return annotated, text

            run_btn.click(_run, inputs=[model_choice, input_image, prompt], outputs=[output_image, output_text])

        return demo

def main():
    logging.info(f"Device: {Config.DEVICE}")
    logging.info(MemoryManager.get_memory_info())

    # Validate critical paths early and warn if missing
    warnings = []
    if not os.path.isdir(YOLOV5_REPO_DIR):
        warnings.append(f"YOLOv5 repo directory not found: {YOLOV5_REPO_DIR}")
    if not os.path.isfile(YOLOV5_MODEL_PATH):
        warnings.append(f"YOLOv5 weights not found: {YOLOV5_MODEL_PATH}")
    if not os.path.isfile(YOLOV8X_MODEL_PATH):
        warnings.append(f"YOLOv8x weights not found: {YOLOV8X_MODEL_PATH}")
    if GEMMA_AVAILABLE and (not os.path.isdir(GEMMA_MODEL_PATH)):
        warnings.append(f"Gemma directory not found: {GEMMA_MODEL_PATH}")

    for w in warnings:
        logging.warning(w)

    analyzer = MultiModelAnalyzer()
    ui = analyzer.build_ui()
    ui.launch()

if __name__ == "__main__":
    main()
