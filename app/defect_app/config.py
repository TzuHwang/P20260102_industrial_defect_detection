"""Model registry for the front / back RF-DETR defect detectors.

`checkpoint` / `dataset_dir` are dev-only (source-tree ONNX export/parity) and stay
repository-relative. The runtime model artifacts live under MODELS_DIR, which
resolves to `app/models` when running from source and to `<exe_dir>/models` when
frozen by PyInstaller, so the packaged app finds its models beside the .exe.
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path


def _models_dir() -> str:
    if getattr(sys, "frozen", False):          # PyInstaller: ship a models/ folder beside the exe
        return os.path.join(os.path.dirname(sys.executable), "models")
    return os.path.join(os.path.dirname(__file__), "..", "models")   # source: app/models


MODELS_DIR = os.path.abspath(_models_dir())


def _cache_dir() -> str:
    """Writable dir for the per-machine TensorRT engine cache. Beside the models
    when running from source; under %LOCALAPPDATA% when frozen, so the app works
    even if installed to a read-only location like Program Files."""
    if getattr(sys, "frozen", False):
        base = os.environ.get("LOCALAPPDATA") or os.path.dirname(sys.executable)
        return os.path.join(base, "DefectDetection", "cache")
    return MODELS_DIR


CACHE_DIR = os.path.abspath(_cache_dir())

# RF-DETR preprocessing: resize to a square `resolution`, scale to [0, 1],
# then normalize with ImageNet statistics (matches rfdetr.RFDETR.predict).
MEANS = (0.485, 0.456, 0.406)
STDS = (0.229, 0.224, 0.225)

# Class names as ordered by COCO category_id (1-based); the ONNX `labels` output
# has len(class_names) + 1 columns, the extra last column being the no-object slot.
FRONT_CLASSES = [
    "表面脏污", "波浪", "残缺", "表面损伤", "烤焦起泡", "烤漆色差_青线割线",
    "空白接头", "黑块", "前工序接头", "印刷重", "上光起泡",
]
BACK_CLASSES = [
    "表面脏污", "波浪", "表面损伤", "烤焦起泡", "烤漆色差_青线割线",
    "前工序接头", "上光起泡",
]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    checkpoint: str      # source .pth (dev-only, for export)
    dataset_dir: str     # COCO dataset (dev-only, for export / parity check)
    onnx: str            # plaintext ONNX produced by export_onnx.py
    encrypted: str       # AES-GCM encrypted model shipped with the app
    class_names: list
    resolution: int = 576


def _spec(name, checkpoint, dataset_dir, class_names):
    model_dir = os.path.join(MODELS_DIR, name)
    return ModelSpec(
        name=name, checkpoint=checkpoint, dataset_dir=dataset_dir,
        onnx=os.path.join(model_dir, "model.onnx"),
        encrypted=os.path.join(model_dir, "model.enc"),
        class_names=class_names,
    )


MODELS = {
    "front": _spec("front", "outputs/rfdetr_medium_front/checkpoint_best_total.pth",
                   "data/internal_train/rfdetr_coco_front", FRONT_CLASSES),
    "back": _spec("back", "outputs/rfdetr_medium_back_v3/checkpoint_best_total.pth",
                  "data/internal_train/rfdetr_coco_back", BACK_CLASSES),
}

# AES key file (raw 32 bytes). Kept out of git; see app/README.md for the
# key-management trade-off when packaging into an .exe.
KEY_PATH = os.path.join(MODELS_DIR, "model.key")


# Derived artifact paths (kept here so scripts and runtime agree).
def fp16_onnx_path(spec) -> str:      # plaintext FP16 ONNX (dev only)
    return str(Path(spec.onnx).with_name("model_fp16.onnx"))


def fp16_enc_path(spec) -> str:       # encrypted FP16 ONNX (shipped; input to TRT build)
    return str(Path(spec.onnx).with_name("model_fp16.enc"))


def trt_cache_path(spec) -> str:      # AES-encrypted TensorRT engine, built + cached on-device
    return os.path.join(CACHE_DIR, spec.name, "model.trt")


def demo_dir(spec) -> str:
    """Folder of images the Demo button plays. Prefers a bundled `demo/<side>/`
    (beside the exe when frozen, under `app/` from source; populated by
    make_demo.py), and falls back to the dataset test split for dev where the
    symlinks resolve (the container)."""
    base = (os.path.dirname(sys.executable) if getattr(sys, "frozen", False)
            else os.path.dirname(os.path.dirname(__file__)))   # app/
    bundled = os.path.join(base, "demo", spec.name)
    if os.path.isdir(bundled):
        return bundled
    return os.path.join(spec.dataset_dir, "test")
