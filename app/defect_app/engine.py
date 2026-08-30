"""ONNX inference wrapper for the RF-DETR defect detectors.

Loads a (optionally AES-encrypted) ONNX model into onnxruntime, replicates
RF-DETR's preprocessing and LWDETR postprocessing, and returns detections in
original-image pixel coordinates. Preprocessing uses OpenCV so it runs on raw
camera frames without a torch dependency on the deployed side.
"""

from typing import NamedTuple, Optional, Sequence

import cv2
import numpy as np
import onnxruntime as ort

from . import crypto
from .config import MEANS, STDS


class Detection(NamedTuple):
    xyxy: tuple      # (x1, y1, x2, y2) in source-image pixels
    score: float
    class_id: int
    class_name: str


_MEAN = np.asarray(MEANS, dtype=np.float32).reshape(3, 1, 1)
_STD = np.asarray(STDS, dtype=np.float32).reshape(3, 1, 1)


def preprocess(image_bgr: np.ndarray, resolution: int) -> np.ndarray:
    """RF-DETR preprocessing: BGR -> RGB, resize square, /255, ImageNet norm, NCHW."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
    chw = resized.astype(np.float32).transpose(2, 0, 1) / 255.0
    chw = (chw - _MEAN) / _STD
    return chw[None]  # (1, 3, R, R)


def postprocess(boxes_cxcywh, logits, img_w, img_h, threshold, class_names, num_select=300):
    """LWDETR postprocessing: sigmoid, top-`num_select` query/class pairs, cxcywh
    (normalized) -> pixel xyxy. `boxes_cxcywh` (Q,4) and `logits` (Q,C) are one image."""
    prob = 1.0 / (1.0 + np.exp(-logits))
    num_classes = len(class_names)
    flat = prob.reshape(-1)
    k = min(num_select, flat.shape[0])
    top = np.argpartition(flat, -k)[-k:]
    num_cls_cols = prob.shape[1]
    query_idx = top // num_cls_cols
    class_idx = top % num_cls_cols
    scores = flat[top]

    results = []
    for q, c, s in zip(query_idx, class_idx, scores):
        if c >= num_classes or s < threshold:   # drop no-object slot + low scores
            continue
        cx, cy, bw, bh = boxes_cxcywh[q]
        x1 = float(np.clip((cx - bw / 2) * img_w, 0, img_w))
        y1 = float(np.clip((cy - bh / 2) * img_h, 0, img_h))
        x2 = float(np.clip((cx + bw / 2) * img_w, 0, img_w))
        y2 = float(np.clip((cy + bh / 2) * img_h, 0, img_h))
        results.append(Detection((x1, y1, x2, y2), float(s), int(c), class_names[int(c)]))
    results.sort(key=lambda d: d.score, reverse=True)
    return results


class RFDetrOnnx:
    def __init__(
        self,
        model_bytes: bytes,
        class_names: Sequence[str],
        resolution: int = 576,
        num_select: int = 300,
        device_id: int = 0,
        providers: Optional[Sequence] = None,
    ):
        # Deployment has two RTX 3070s: pin each stream to its own GPU via
        # device_id (front -> 0, back -> 1) so the cameras don't share a device.
        if providers is None:
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers = [("CUDAExecutionProvider", {"device_id": device_id}),
                             "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_bytes, providers=list(providers))
        self.input_name = self.session.get_inputs()[0].name
        self.class_names = list(class_names)
        self.resolution = resolution
        self.num_select = num_select

    @classmethod
    def from_onnx(cls, path: str, **kwargs) -> "RFDetrOnnx":
        with open(path, "rb") as f:
            return cls(f.read(), **kwargs)

    @classmethod
    def from_encrypted(cls, path: str, key: bytes, **kwargs) -> "RFDetrOnnx":
        with open(path, "rb") as f:
            return cls(crypto.decrypt(f.read(), key), **kwargs)

    def predict(self, image_bgr: np.ndarray, threshold: float = 0.5):
        """Run detection on a single BGR image (as read by cv2)."""
        h, w = image_bgr.shape[:2]
        inp = preprocess(image_bgr, self.resolution)
        dets, logits = self.session.run(None, {self.input_name: inp})
        return postprocess(dets[0], logits[0], w, h, threshold, self.class_names, self.num_select)
