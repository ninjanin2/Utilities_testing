#!/usr/bin/env python3
"""
Offline-Only Professional Vision Analysis Suite
- YOLOv5 via local PyTorch Hub repo + local .pt
- YOLOv8x via Ultralytics local .pt
- Optional: Gemma 3n-E4B-it via local Transformers directory

Fixes:
- Gemma 3n-E4B-it device mismatch resolved by strictly following HF model card usage:
  - processor.apply_chat_template(...).to(model.device)
  - Generate with inputs on same device, slice generation using input_len, decode via processor.decode
  - Optional do_pan_and_scan for better vision quality on high-res images
- YOLOv5 autoshape guard to avoid "'AutoShape' object has no attribute 'autoshape'"
- Modern, professional Gradio UI with tabs, cards, theme, sticky actions, pagination

References consulted:
- Hugging Face model card for google/gemma-3n-E4B-it (usage, chat template, device)[9]
- HF transformers docs for Gemma 3/Gemma3n (processor.apply_chat_template, do_pan_and_scan)[18][13]
- Google AI developer docs for Gemma 3n overview (capabilities)[17]
- HF blog about Gemma 3n availability (ecosystem support)[10]
- Hugging Face guidance on running Gemma with Transformers (vision-text flow)[11]

Author: AI Assistant
Version: 4.4 (Gemma device fix + Pro UI)
"""

import os
import sys
import gc
import time
import math
import logging
from datetime import datetime
from typing import Optional, Tuple, List, Dict

import numpy as np
import cv2
from PIL import Image

# =========================
# Global Local Model Paths
# =========================
# 1) YOLOv5 offline:
#    - YOLOV5_REPO_DIR: local clone of ultralytics/yolov5 (must contain hubconf.py, models/, utils/, etc.)
#    - YOLOV5_MODEL_PATH: YOLOv5 .pt checkpoint trained with YOLOv5 repo
YOLOV5_REPO_DIR = r"/replace/with/local/yolov5_repo"
YOLOV5_MODEL_PATH = r"/replace/with/local/yolov5s.pt"

# 2) YOLOv8 offline:
YOLOV8X_MODEL_PATH = r"/replace/with/local/yolov8x.pt"

# 3) Gemma 3n (optional, offline Transformers dir):
#    Must be a local folder containing config, tokenizer, processor, and model weights for google/gemma-3n-E4B-it
GEMMA_MODEL_PATH = r"/replace/with/local/gemma-3n-E4B-it"

# 4) Batch outputs directory
BATCH_OUTPUT_DIR = r"./batch_outputs"
# =========================

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Torch
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    print("Error: PyTorch is required. Please install offline.")
    TORCH_AVAILABLE = False
    sys.exit(1)

# Ultralytics (YOLOv8)
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
    print("Error: Gradio is required. Please install offline.")
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
        # As per HF card, Gemma3nForConditionalGeneration is the correct class name[9].
        from transformers import Gemma3nForConditionalGeneration as _GemmaClass
        GemmaModelClass = _GemmaClass
        GEMMA_AVAILABLE = True
    except Exception:
        GEMMA_AVAILABLE = False
        GemmaModelClass = None


class Config:
    INTERFACE_TITLE = "Vision Analysis Suite (Offline, Pro)"
    INTERFACE_DESCRIPTION = """
    Enterprise-grade, fully offline multi-model vision analysis.
    - YOLOv5 (local repo + local weights)
    - YOLOv8x (local weights)
    - Optional: Gemma 3n-E4B-it (local Transformers dir)
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
    GEMMA_DO_PAN_AND_SCAN = False  # Set True to improve large image handling per HF docs[18]

    SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


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

    @staticmethod
    def list_images_in_dir(dir_path: str) -> List[str]:
        if not dir_path or not os.path.isdir(dir_path):
            return []
        files = []
        for name in os.listdir(dir_path):
            p = os.path.join(dir_path, name)
            if os.path.isfile(p):
                ext = os.path.splitext(p)[1].lower()
                if ext in Config.SUPPORTED_FORMATS:
                    files.append(p)
        files.sort()
        return files

    @staticmethod
    def save_image(img: Image.Image, out_path: str):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        img.save(out_path)


# -----------------------------------------
# YOLOv5 (Offline via local torch.hub repo)
# -----------------------------------------
class YOLOv5LocalHubModel:
    """
    Offline YOLOv5 using a local torch.hub repo + local weights.
    Guard autoshape usage to avoid "'AutoShape' object has no attribute 'autoshape'".
    """
    def __init__(self, conf=0.25, iou=0.45, max_det=300):
        self.model = None
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
        self._validate_paths(repo_dir, weights_path)
        MemoryManager.clear_gpu_memory()
        logging.info(f"Loading YOLOv5 (offline) from repo '{repo_dir}' weights '{weights_path}' on {Config.DEVICE} ...")

        mdl = torch.hub.load(repo_dir, 'custom', path=weights_path, source='local')  # offline local hub

        # Guard autoshape call
        try:
            if hasattr(mdl, 'autoshape') and callable(getattr(mdl, 'autoshape')):
                if mdl.__class__.__name__.lower() not in ("autoshape",):
                    mdl = mdl.autoshape()
        except Exception:
            pass

        # Device (AutoShape can manage device; ignore failures)
        try:
            if Config.DEVICE == "cuda":
                mdl.to(Config.DEVICE)
        except Exception:
            pass

        # Inference params if supported
        for attr, val in (("conf", self.conf), ("iou", self.iou), ("max_det", self.max_det)):
            try:
                setattr(mdl, attr, val)
            except Exception:
                pass

        self.model = mdl
        try:
            self.names = getattr(self.model, "names", None)
        except Exception:
            self.names = None

        logging.info(f"✅ YOLOv5 loaded. {MemoryManager.get_memory_info()}")
        return True

    def offload(self):
        self.model = None
        MemoryManager.clear_gpu_memory()
        logging.info("✅ YOLOv5 offloaded.")

    def predict(self, image_pil: Image.Image) -> Tuple[Image.Image, str]:
        if self.model is None:
            return image_pil, "❌ YOLOv5 not loaded."

        img_np = ImageProcessor.to_numpy_rgb(image_pil)
        try:
            results = self.model(img_np)  # AutoShape supports np/PIL inputs
        except Exception as e:
            return image_pil, f"❌ Detection failed: {e}"

        # Render
        annotated_pil = image_pil
        try:
            if hasattr(results, "render"):
                results.render()  # updates results.ims/imgs in-place
                annotated_arrs = getattr(results, "ims", None) or getattr(results, "imgs", None)
                if isinstance(annotated_arrs, list) and len(annotated_arrs) > 0:
                    ann = annotated_arrs[0]
                    if isinstance(ann, np.ndarray):
                        annotated_pil = Image.fromarray(ann)
                elif isinstance(annotated_arrs, np.ndarray):
                    annotated_pil = Image.fromarray(annotated_arrs)
        except Exception as e:
            logging.warning(f"Annotation render error: {e}")

        # Summary
        try:
            summary = self._build_summary(results)
        except Exception as e:
            summary = f"Detection completed, but summary failed: {e}"

        return annotated_pil, summary

    def _build_summary(self, results) -> str:
        # results.xyxy may be list per image
        if not hasattr(results, "xyxy"):
            return "No results returned."
        xyxy = results.xyxy
        if isinstance(xyxy, list) and len(xyxy) > 0:
            det = xyxy[0]
        else:
            det = xyxy
        if det is None or (hasattr(det, "shape") and det.shape[0] == 0):
            return "No objects detected."

        names = self.names if self.names is not None else getattr(results, "names", {})
        class_counts: Dict[str, List[float]] = {}
        rows = det.detach().cpu().numpy()
        for row in rows:
            # x1,y1,x2,y2,conf,cls
            conf = float(row[4])
            cls_id = int(row[5])
            if isinstance(names, dict):
                cls_name = names.get(cls_id, f"Class_{cls_id}")
            elif isinstance(names, (list, tuple)) and cls_id < len(names):
                cls_name = names[cls_id]
            else:
                cls_name = f"Class_{cls_id}"
            class_counts.setdefault(cls_name, []).append(conf)

        total = len(rows)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            "# 🎯 YOLOv5 Object Detection Analysis (Offline)",
            "Model: YOLOv5 (Local Hub)",
            f"Total Objects Detected: {total}",
            f"Confidence Threshold: {self.conf:.2f}",
            f"IoU Threshold: {self.iou:.2f}",
            f"Analysis Timestamp: {ts}",
            "",
            "## 📊 Detection Summary by Class:"
        ]
        for cls_name, confs in sorted(class_counts.items(), key=lambda kv: len(kv[1]), reverse=True):
            count = len(confs)
            avg_conf = sum(confs) / count
            lines.append(f"- {cls_name}: count={count}, avg_conf={avg_conf:.3f}")
        return "\n".join(lines)


# -----------------------------
# YOLOv8x (Offline, Ultralytics)
# -----------------------------
class YOLOv8LocalModel:
    def __init__(self, conf=0.25, iou=0.45, max_det=300):
        self.model = None
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.names = None

    def load(self, weights_path: str):
        if not ULTRALYTICS_AVAILABLE:
            raise RuntimeError("Ultralytics not available; install offline.")
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"YOLOv8 weights not found: {weights_path}")

        MemoryManager.clear_gpu_memory()
        logging.info(f"Loading YOLOv8x offline from '{weights_path}'...")
        self.model = UltralyticsYOLO(weights_path)
        try:
            self.names = self.model.model.names
        except Exception:
            self.names = None
        logging.info(f"✅ YOLOv8x loaded. {MemoryManager.get_memory_info()}")
        return True

    def offload(self):
        self.model = None
        MemoryManager.clear_gpu_memory()
        logging.info("✅ YOLOv8x offloaded.")

    def predict(self, image_pil: Image.Image) -> Tuple[Image.Image, str]:
        if self.model is None:
            return image_pil, "❌ YOLOv8x not loaded."
        img_np = ImageProcessor.to_numpy_rgb(image_pil)
        try:
            results_list = self.model.predict(
                source=img_np,
                conf=self.conf,
                iou=self.iou,
                max_det=self.max_det,
                verbose=False
            )
        except Exception as e:
            return image_pil, f"❌ Detection failed: {e}"

        if not results_list:
            return image_pil, "No results returned."
        result = results_list[0]

        annotated_pil = image_pil
        try:
            im_bgr = result.plot()
            im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
            annotated_pil = Image.fromarray(im_rgb)
        except Exception as e:
            logging.warning(f"Annotation render error: {e}")

        try:
            summary = self._build_summary(result)
        except Exception as e:
            summary = f"Detection completed, but summary failed: {e}"

        return annotated_pil, summary

    def _build_summary(self, result) -> str:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return "No objects detected."

        names = getattr(result, "names", None) or self.names
        try:
            confs = boxes.conf.detach().cpu().numpy()
            clss = boxes.cls.detach().cpu().numpy().astype(int)
        except Exception as e:
            return f"Failed to parse detections: {e}"

        class_counts: Dict[str, List[float]] = {}
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
        lines = [
            "# 🎯 YOLOv8x Object Detection Analysis (Offline)",
            "Model: YOLOv8x (Local)",
            f"Total Objects Detected: {total}",
            f"Confidence Threshold: {self.conf:.2f}",
            f"IoU Threshold: {self.iou:.2f}",
            f"Analysis Timestamp: {ts}",
            "",
            "## 📊 Detection Summary by Class:"
        ]
        for cls_name, confs_list in sorted(class_counts.items(), key=lambda kv: len(kv[1]), reverse=True):
            count = len(confs_list)
            avg_conf = sum(confs_list) / count
            lines.append(f"- {cls_name}: count={count}, avg_conf={avg_conf:.3f}")
        return "\n".join(lines)


# ----------------------------
# Gemma 3n-E4B-it (Offline)
# ----------------------------
class GemmaLocalModel:
    """
    Implements HF-recommended usage to avoid device mismatch:
    - Build messages with image + text
    - inputs = processor.apply_chat_template(..., return_tensors="pt").to(model.device)[9]
    - Track input_len and slice generation[9]
    - Optional do_pan_and_scan per HF docs to improve visual quality[18]
    """
    def __init__(self):
        self.model = None
        self.processor = None
        self.loaded = False

    def load(self, local_dir: str):
        if not (TRANSFORMERS_AVAILABLE and GEMMA_AVAILABLE and GemmaModelClass is not None):
            raise RuntimeError("Gemma not available in this environment.")
        if not os.path.isdir(local_dir):
            raise FileNotFoundError(f"Gemma local directory not found: {local_dir}")

        MemoryManager.clear_gpu_memory()
        logging.info(f"Loading Gemma (offline) from '{local_dir}'...")
        # Processor and model as per HF model card; v4.53.0+ required[9]
        self.processor = AutoProcessor.from_pretrained(
            local_dir, trust_remote_code=True, local_files_only=True
        )
        self.model = GemmaModelClass.from_pretrained(
            local_dir,
            torch_dtype=Config.TORCH_DTYPE,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            local_files_only=True
        )
        # Move model to single device (no device_map=auto to keep full control offline)
        self.model = self.model.to(Config.DEVICE).eval()
        self.loaded = True
        logging.info(f"✅ Gemma loaded. {MemoryManager.get_memory_info()}")
        return True

    def offload(self):
        self.model = None
        self.processor = None
        self.loaded = False
        MemoryManager.clear_gpu_memory()
        logging.info("✅ Gemma offloaded.")

    def analyze(self, image_pil: Image.Image, prompt: Optional[str]) -> str:
        if not self.loaded or self.model is None or self.processor is None:
            return "❌ Gemma not loaded."
        if image_pil is None:
            return "❌ No image provided."

        try:
            # Build messages per HF card[9]; processor will inject required special tokens[18]
            messages = [
                {"role": "system", "content": [
                    {"type": "text", "text": "You are a professional image analyst."}
                ]},
                {"role": "user", "content": [
                    {"type": "image", "image": ImageProcessor.to_pil_rgb(image_pil)},
                    {"type": "text", "text": prompt or "Describe the image in detail."}
                ]}
            ]

            # Optional: pan-and-scan improves quality on non-square/high-res images[18]
            # Note: argument key follows HF docs for Gemma 3
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                do_pan_and_scan=Config.GEMMA_DO_PAN_AND_SCAN  # per docs[18]
            )

            # CRITICAL: move the entire inputs dict to model.device[9][11]
            # This avoids the cuda:0 vs cpu mismatch.
            device = next(self.model.parameters()).device
            if hasattr(inputs, "to"):
                inputs = inputs.to(device)  # when HF returns a BatchEncoding with .to()
            else:
                for k, v in list(inputs.items()):
                    if hasattr(v, "to"):
                        inputs[k] = v.to(device)

            input_len = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generation = self.model.generate(
                    **inputs,
                    max_new_tokens=Config.GEMMA_MAX_NEW_TOKENS,
                    temperature=Config.GEMMA_TEMPERATURE,
                    do_sample=False  # deterministic by default; change to True for variety
                )

            # Slice newly generated tokens per HF card[9]
            if generation.dim() > 1:
                generation = generation[0][input_len:]
            else:
                generation = generation[input_len:]

            # Decode using processor.decode per HF card[9]
            decoded = self.processor.decode(generation, skip_special_tokens=True)
            decoded = decoded.strip() if isinstance(decoded, str) else str(decoded)
            return decoded if decoded else "No response generated."
        except torch.cuda.OutOfMemoryError:
            MemoryManager.clear_gpu_memory()
            return "❌ GPU out of memory during Gemma generation."
        except Exception as e:
            return f"❌ Gemma analysis error: {e}"


# ---------------------------------
# Multi-model Controller with Batch
# ---------------------------------
class MultiModelAnalyzer:
    def __init__(self):
        self.proc = ImageProcessor()

        self.yolov5 = YOLOv5LocalHubModel(
            conf=Config.YOLO_CONFIDENCE,
            iou=Config.YOLO_IOU,
            max_det=Config.YOLO_MAX_DETECTIONS
        )
        self.yolov8x = YOLOv8LocalModel(
            conf=Config.YOLO_CONFIDENCE,
            iou=Config.YOLO_IOU,
            max_det=Config.YOLO_MAX_DETECTIONS
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
        time.sleep(0.1)
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

    # Single image
    def analyze_single(self, model_name: str, image: Image.Image, prompt: Optional[str]) -> Tuple[Optional[Image.Image], str]:
        if image is None:
            return None, "❌ Please provide an image."
        status = self.load_model(model_name)
        if status.startswith("❌"):
            return None, status
        try:
            if model_name.startswith("YOLOv5"):
                return self.yolov5.predict(image)
            elif model_name.startswith("YOLOv8x"):
                return self.yolov8x.predict(image)
            elif model_name.startswith("Gemma"):
                return image, self.gemma.analyze(image, prompt)
            else:
                return None, "❌ Unsupported model."
        except Exception as e:
            return None, f"❌ Inference error: {e}"

    # Batch processing with pagination
    def analyze_batch(
        self, model_name: str, directory: str, prompt: Optional[str],
        page: int, page_size: int, save_outputs: bool
    ):
        files = self.proc.list_images_in_dir(directory)
        total = len(files)
        if total == 0:
            return [], [], f"❌ No valid images found in: {directory}", 0, 0

        status = self.load_model(model_name)
        if status.startswith("❌"):
            return [], [], status, 0, 0

        # Pagination setup
        pages = max(1, math.ceil(total / page_size))
        page = max(1, min(page, pages))
        start = (page - 1) * page_size
        end = min(total, start + page_size)
        subset = files[start:end]

        images_out: List[Optional[Image.Image]] = []
        texts_out: List[str] = []
        errors = 0

        os.makedirs(BATCH_OUTPUT_DIR, exist_ok=True)
        out_dir = os.path.join(BATCH_OUTPUT_DIR, model_name.replace(" ", "_"))
        if save_outputs:
            os.makedirs(out_dir, exist_ok=True)

        for path in subset:
            try:
                with Image.open(path) as img:
                    img = img.convert("RGB")
                if model_name.startswith("YOLOv5"):
                    annotated, summary = self.yolov5.predict(img)
                elif model_name.startswith("YOLOv8x"):
                    annotated, summary = self.yolov8x.predict(img)
                elif model_name.startswith("Gemma"):
                    annotated, summary = img, self.gemma.analyze(img, prompt)
                else:
                    annotated, summary = img, "❌ Unsupported model."

                images_out.append(annotated)
                fname = os.path.basename(path)
                texts_out.append(f"[{fname}]\n{summary}")

                if save_outputs:
                    base, _ = os.path.splitext(os.path.basename(path))
                    out_img = os.path.join(out_dir, f"{base}_annotated.png")
                    if annotated is not None:
                        ImageProcessor.save_image(annotated, out_img)
                    out_txt = os.path.join(out_dir, f"{base}_summary.txt")
                    with open(out_txt, "w", encoding="utf-8") as f:
                        f.write(summary)

            except Exception as e:
                errors += 1
                texts_out.append(f"[{os.path.basename(path)}] Error: {e}")
                images_out.append(None)

        status_text = f"Processed {len(subset)} images on page {page}/{pages}. Errors: {errors}. Total images: {total}."
        return images_out, texts_out, status_text, pages, total

    # UI
    def build_ui(self):
        # A modern theme with custom CSS for elevated, professional look
        theme = gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="slate",
            neutral_hue="slate",
            radius_size=gr.themes.sizes.radius_md,
        ).set(
            body_background_fill="#0f172a",  # slate-900
            block_title_text_color="#e2e8f0",  # slate-200
            block_label_text_color="#94a3b8",  # slate-400
            input_background_fill="#111827",  # gray-900
            input_border_color="#334155",     # slate-700
            button_primary_background_fill="#4f46e5",  # indigo-600
            button_primary_background_fill_hover="#4338ca",  # indigo-700
            border_color_primary="#1e293b",   # slate-800
        )

        custom_css = """
        .app-title { font-size: 26px; font-weight: 700; color: #e2e8f0; margin-bottom: 8px; }
        .app-subtitle { color: #94a3b8; margin-bottom: 18px; }
        .card { background: #0b1220; border: 1px solid #1e293b; border-radius: 14px; padding: 14px; }
        .sticky-actions { position: sticky; top: 8px; z-index: 5; }
        .status-box { background:#0b1020; color:#bfdbfe; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono"; padding:12px; border-radius:10px; }
        .gallery .container { background: transparent !important; }
        """

        with gr.Blocks(theme=theme, title=Config.INTERFACE_TITLE, css=custom_css) as demo:
            gr.Markdown("<div class='app-title'>🔭 Vision Analysis Suite</div>", elem_id="title")
            gr.Markdown(f"<div class='app-subtitle'>{Config.INTERFACE_DESCRIPTION}</div>")

            with gr.Tabs():
                with gr.Tab("Single Image"):
                    with gr.Row():
                        with gr.Column(scale=1, min_width=340, elem_classes=["card", "sticky-actions"]):
                            model_choice = gr.Dropdown(choices=self.available_models(), value=self.available_models()[0], label="Model")
                            prompt = gr.Textbox(label="Prompt (Gemma only)", value="Provide a comprehensive analysis of this image.", lines=4, placeholder="Describe the image focusing on key objects, context, composition, and technical qualities.")
                            input_image = gr.Image(type="pil", label="Input Image")
                            with gr.Row():
                                run_btn = gr.Button("Run Analysis/Detection", variant="primary")
                                clear_btn = gr.Button("Clear")
                            sys_box = gr.Markdown(value=f"Device: {Config.DEVICE}<br>{MemoryManager.get_memory_info()}", elem_classes=["status-box"])

                        with gr.Column(scale=2, elem_classes=["card"]):
                            output_image = gr.Image(label="Annotated/Processed Image", height=520)
                            output_text = gr.Textbox(label="Analysis / Detection Summary", lines=20)

                    def run_single(model, img, p):
                        annotated, text = self.analyze_single(model, img, p)
                        return annotated, text, f"Device: {Config.DEVICE}<br>{MemoryManager.get_memory_info()}"

                    run_btn.click(run_single, inputs=[model_choice, input_image, prompt], outputs=[output_image, output_text, sys_box])

                    def clear_all():
                        return None, "", f"Device: {Config.DEVICE}<br>{MemoryManager.get_memory_info()}"

                    clear_btn.click(clear_all, inputs=None, outputs=[output_image, output_text, sys_box])

                with gr.Tab("Batch Processing"):
                    with gr.Row():
                        with gr.Column(scale=1, min_width=340, elem_classes=["card", "sticky-actions"]):
                            b_model = gr.Dropdown(choices=self.available_models(), value=self.available_models()[0], label="Model")
                            b_dir = gr.Textbox(label="Input Directory (Images)", placeholder="/path/to/images")
                            b_prompt = gr.Textbox(label="Prompt (Gemma only)", value="Provide a comprehensive analysis of each image.", lines=4)
                            b_save = gr.Checkbox(label="Save outputs to disk", value=True)
                            with gr.Row():
                                b_page = gr.Number(label="Page", value=1, precision=0)
                                b_page_size = gr.Slider(label="Page size", minimum=1, maximum=20, value=6, step=1)
                            b_run = gr.Button("Run Batch", variant="primary")
                            b_status = gr.Markdown(elem_classes=["status-box"])
                        with gr.Column(scale=2, elem_classes=["card"]):
                            b_gallery = gr.Gallery(label="Annotated Images (Paginated)", columns=3, height=520, preview=True, allow_preview=True)
                            b_texts = gr.Textbox(label="Summaries (for current page)", lines=24)

                    def run_batch(model, directory, prompt_text, save_outputs, page, page_size):
                        try:
                            page = int(page) if page is not None else 1
                            page_size = int(page_size) if page_size is not None else 6
                        except Exception:
                            page, page_size = 1, 6
                        imgs, texts, status, pages, total = self.analyze_batch(model, directory, prompt_text, page, page_size, save_outputs)
                        gallery_items = [img for img in imgs if img is not None]
                        combined_text = "\n\n".join(texts[:page_size])
                        status_line = f"{status} | Pages: {pages}, Total images: {total}. Adjust Page to navigate."
                        return gallery_items, combined_text, status_line

                    b_run.click(run_batch, inputs=[b_model, b_dir, b_prompt, b_save, b_page, b_page_size], outputs=[b_gallery, b_texts, b_status])

                with gr.Tab("System & Paths"):
                    with gr.Row():
                        with gr.Column(elem_classes=["card"]):
                            gr.Markdown("### Environment")
                            gr.Markdown(f"- Device: {Config.DEVICE}<br>- CUDA: {'Yes' if torch.cuda.is_available() else 'No'}<br>- Memory: {MemoryManager.get_memory_info()}")
                        with gr.Column(elem_classes=["card"]):
                            gr.Markdown("### Local Paths in Use")
                            gr.Markdown(f"- YOLOv5 repo: {YOLOV5_REPO_DIR}<br>- YOLOv5 weights: {YOLOV5_MODEL_PATH}<br>- YOLOv8x weights: {YOLOV8X_MODEL_PATH}<br>- Gemma directory: {GEMMA_MODEL_PATH}<br>- Batch output dir: {BATCH_OUTPUT_DIR}")

        return demo


def main():
    logging.info(f"Device: {Config.DEVICE}")
    logging.info(MemoryManager.get_memory_info())

    # Early warnings if paths are invalid
    if not os.path.isdir(YOLOV5_REPO_DIR):
        logging.warning(f"YOLOv5 repo directory not found: {YOLOV5_REPO_DIR}")
    if not os.path.isfile(YOLOV5_MODEL_PATH):
        logging.warning(f"YOLOv5 weights not found: {YOLOV5_MODEL_PATH}")
    if not os.path.isfile(YOLOV8X_MODEL_PATH):
        logging.warning(f"YOLOv8x weights not found: {YOLOV8X_MODEL_PATH}")
    if GEMMA_AVAILABLE and (not os.path.isdir(GEMMA_MODEL_PATH)):
        logging.warning(f"Gemma directory not found: {GEMMA_MODEL_PATH}")
    os.makedirs(BATCH_OUTPUT_DIR, exist_ok=True)

    analyzer = MultiModelAnalyzer()
    ui = analyzer.build_ui()
    # Queue to improve responsiveness; keep concurrency modest for offline GPU
    ui.queue(concurrency_count=2, max_size=32).launch(show_error=True, inbrowser=False)


if __name__ == "__main__":
    main()
