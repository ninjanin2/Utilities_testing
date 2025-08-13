#!/usr/bin/env python3
"""
Offline-Only Professional Vision Analysis Suite
- YOLOv5 via local PyTorch Hub repo + local .pt
- YOLOv8x via Ultralytics local .pt
- Optional: Gemma 3n-E4B-it via local Transformers directory

Features:
- Single image and batch (directory) processing with pagination
- Professional Gradio UI: tabs, status console, environment info
- Robust memory handling and device alignment
- Fully offline: no network calls

Author: AI Assistant
Version: 4.3 (Revalidated)
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
#    - YOLOV5_MODEL_PATH: YOLOv5 .pt checkpoint trained with the YOLOv5 repo
YOLOV5_REPO_DIR = r"/replace/with/local/yolov5_repo"
YOLOV5_MODEL_PATH = r"/replace/with/local/yolov5s.pt"

# 2) YOLOv8 offline:
YOLOV8X_MODEL_PATH = r"/replace/with/local/yolov8x.pt"

# 3) Gemma 3n (optional, offline Transformers dir):
GEMMA_MODEL_PATH = r"/replace/with/local/gemma-3n-e4b-it"

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
        from transformers import Gemma3nForConditionalGeneration as _GemmaClass
        GemmaModelClass = _GemmaClass
        GEMMA_AVAILABLE = True
    except Exception:
        GEMMA_AVAILABLE = False
        GemmaModelClass = None


class Config:
    INTERFACE_TITLE = "🔬 Vision Analysis Suite (Offline, Pro)"
    INTERFACE_DESCRIPTION = """
    Professional multi-model vision analysis, fully offline.
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
    Fix: guard autoshape usage to avoid "'AutoShape' object has no attribute 'autoshape'"[9][10][16].
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

        mdl = torch.hub.load(repo_dir, 'custom', path=weights_path, source='local')  # local hub load[9][16]

        # Guard autoshape call: some hub returns already AutoShape-wrapped models[10].
        try:
            if hasattr(mdl, 'autoshape') and callable(getattr(mdl, 'autoshape')):
                # If class name suggests it's already autoshape, skip
                if mdl.__class__.__name__.lower() not in ("autoshape",):
                    mdl = mdl.autoshape()
        except Exception:
            pass

        # Device transfer (AutoShape manages device internally; safe to skip if fails)
        try:
            if Config.DEVICE == "cuda":
                mdl.to(Config.DEVICE)
        except Exception:
            pass

        # Inference params when supported by model wrapper
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
            results = self.model(img_np)  # batched or single works with AutoShape[9]
        except Exception as e:
            return image_pil, f"❌ Detection failed: {e}"

        # Render annotated output
        annotated_pil = image_pil
        try:
            if hasattr(results, "render"):
                results.render()  # updates results.ims/imgs in-place[9]
                # YOLOv5 results store annotated images in results.ims or results.imgs (depending on version)
                annotated_arrs = getattr(results, "ims", None) or getattr(results, "imgs", None)
                if isinstance(annotated_arrs, list) and len(annotated_arrs) > 0:
                    ann = annotated_arrs[0]
                    if isinstance(ann, np.ndarray):
                        annotated_pil = Image.fromarray(ann)
                elif isinstance(annotated_arrs, np.ndarray):
                    annotated_pil = Image.fromarray(annotated_arrs)
        except Exception as e:
            logging.warning(f"Annotation render error: {e}")

        # Build summary
        try:
            summary = self._build_summary(results)
        except Exception as e:
            summary = f"Detection completed, but summary failed: {e}"

        return annotated_pil, summary

    def _build_summary(self, results) -> str:
        # As per YOLOv5 hub docs, results.xyxy[i] holds tensor predictions per image[9]
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

        # Plot per Ultralytics docs: result.plot() returns BGR array[14][17][23]
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
# Optional Gemma (Offline)
# ----------------------------
class GemmaLocalModel:
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
        self.processor = AutoProcessor.from_pretrained(local_dir, trust_remote_code=True, local_files_only=True)
        self.model = GemmaModelClass.from_pretrained(
            local_dir,
            torch_dtype=Config.TORCH_DTYPE,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            local_files_only=True
        ).to(Config.DEVICE)
        self.model.eval()
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
            # Build inputs. If chat template available, use it; else standard processor.
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

            # CRITICAL: move all tensors to the same device as model to avoid cuda/cpu mismatch
            device = next(self.model.parameters()).device
            if isinstance(inputs, dict):
                for k, v in list(inputs.items()):
                    if hasattr(v, "to"):
                        inputs[k] = v.to(device)

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
            # Handle both 1D and 2D outputs
            gen = generation[0] if generation.ndim > 1 else generation
            text = tokenizer.decode(gen, skip_special_tokens=True).strip()
            return text if text else "No response generated."
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
        with gr.Blocks(
            theme=gr.themes.Soft(font=["Inter", "Source Sans Pro"]),
            title=Config.INTERFACE_TITLE,
            css="""
            .content-card {border: 1px solid #e5e7eb; border-radius: 10px; padding: 12px; background: #ffffff;}
            .status-box {background:#0b1020; color:#d1e3ff; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono"; padding:10px; border-radius:8px;}
            """
        ) as demo:
            gr.Markdown(f"# {Config.INTERFACE_TITLE}")
            gr.Markdown(Config.INTERFACE_DESCRIPTION)

            with gr.Tabs():
                with gr.Tab("Single Image"):
                    with gr.Row():
                        with gr.Column(scale=1, min_width=320):
                            model_choice = gr.Dropdown(choices=self.available_models(), value=self.available_models()[0], label="Model")
                            prompt = gr.Textbox(label="Prompt (Gemma only)", value="Provide a comprehensive analysis of this image.", lines=3)
                            input_image = gr.Image(type="pil", label="Input Image")
                            run_btn = gr.Button("Run Analysis/Detection", variant="primary")
                            sys_box = gr.Markdown(value=f"Device: {Config.DEVICE}<br>{MemoryManager.get_memory_info()}", elem_classes=["content-card"])
                        with gr.Column(scale=2):
                            output_image = gr.Image(label="Annotated/Processed Image", height=480)
                            output_text = gr.Textbox(label="Analysis / Detection Summary", lines=18)

                    def run_single(model, img, p):
                        annotated, text = self.analyze_single(model, img, p)
                        return annotated, text, f"Device: {Config.DEVICE}<br>{MemoryManager.get_memory_info()}"

                    run_btn.click(run_single, inputs=[model_choice, input_image, prompt], outputs=[output_image, output_text, sys_box])

                with gr.Tab("Batch Processing"):
                    with gr.Row():
                        with gr.Column(scale=1, min_width=320):
                            b_model = gr.Dropdown(choices=self.available_models(), value=self.available_models()[0], label="Model")
                            b_dir = gr.Textbox(label="Input Directory (Images)", placeholder="/path/to/images")
                            b_prompt = gr.Textbox(label="Prompt (Gemma only)", value="Provide a comprehensive analysis of this image.", lines=3)
                            b_save = gr.Checkbox(label="Save outputs to disk", value=True)
                            with gr.Row():
                                b_page = gr.Number(label="Page", value=1, precision=0)
                                b_page_size = gr.Slider(label="Page size", minimum=1, maximum=20, value=6, step=1)
                            b_run = gr.Button("Run Batch", variant="primary")
                            b_status = gr.Markdown(elem_classes=["status-box"])
                        with gr.Column(scale=2):
                            b_gallery = gr.Gallery(label="Annotated Images (Paginated)", columns=3, height=480, preview=True)
                            b_texts = gr.Textbox(label="Summaries (for current page)", lines=20)

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
                    gr.Markdown("### Environment")
                    gr.Markdown(f"- Device: {Config.DEVICE}<br>- CUDA: {'Yes' if torch.cuda.is_available() else 'No'}<br>- Memory: {MemoryManager.get_memory_info()}", elem_classes=["content-card"])
                    gr.Markdown("### Local Paths in Use")
                    gr.Markdown(f"- YOLOv5 repo: {YOLOV5_REPO_DIR}<br>- YOLOv5 weights: {YOLOV5_MODEL_PATH}<br>- YOLOv8x weights: {YOLOV8X_MODEL_PATH}<br>- Gemma directory: {GEMMA_MODEL_PATH}<br>- Batch output dir: {BATCH_OUTPUT_DIR}", elem_classes=["content-card"])

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
    # Queue to improve responsiveness and avoid blocking; modest concurrency for offline machines
    ui.queue(concurrency_count=2, max_size=32).launch(show_error=True, inbrowser=False)


if __name__ == "__main__":
    main()
