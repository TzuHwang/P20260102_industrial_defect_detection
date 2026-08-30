"""Verify the encrypted-ONNX engine matches the original RF-DETR checkpoint.

Runs both the reference `rfdetr.predict` (PyTorch) and the AES-decrypted ONNX
engine on the same test-split images and reports, per image, how many detections
match by class + IoU. Dev-only (needs rfdetr + the plaintext checkpoint).

Usage:
    python -m app.scripts.verify_parity --model front --num-images 20 --threshold 0.5
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from rfdetr import RFDETR

from app.defect_app import crypto
from app.defect_app.config import KEY_PATH, MODELS
from app.defect_app.engine import RFDetrOnnx


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _sample_images(spec, num_images, seed):
    split_dir = Path(spec.dataset_dir) / "test"
    coco = json.load(open(split_dir / "_annotations.coco.json", encoding="utf-8"))
    paths = [str(split_dir / im["file_name"]) for im in coco["images"]]
    random.seed(seed)
    return random.sample(paths, min(num_images, len(paths)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS), default="front")
    parser.add_argument("--num-images", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--iou-match", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    spec = MODELS[args.model]
    key = crypto.load_key(KEY_PATH)
    engine = RFDetrOnnx.from_encrypted(
        spec.encrypted, key, class_names=spec.class_names,
        resolution=spec.resolution, providers=["CPUExecutionProvider"],
    )
    ref = RFDETR.from_checkpoint(spec.checkpoint)

    paths = _sample_images(spec, args.num_images, args.seed)
    n_ref = n_onnx = n_matched = 0
    ious = []
    for p in paths:
        det_ref = ref.predict(p, threshold=args.threshold)
        ref_boxes = [(tuple(xyxy), int(c)) for xyxy, c in zip(det_ref.xyxy, det_ref.class_id)
                     if int(c) < len(spec.class_names)]

        onnx_dets = engine.predict(cv2.imread(p), threshold=args.threshold)
        onnx_boxes = [(d.xyxy, d.class_id) for d in onnx_dets]

        n_ref += len(ref_boxes)
        n_onnx += len(onnx_boxes)
        used = set()
        for rb, rc in ref_boxes:
            best_i, best_iou = -1, 0.0
            for i, (ob, oc) in enumerate(onnx_boxes):
                if i in used or oc != rc:
                    continue
                iou = _iou(rb, ob)
                if iou > best_iou:
                    best_i, best_iou = i, iou
            if best_iou >= args.iou_match:
                used.add(best_i)
                n_matched += 1
                ious.append(best_iou)

    print(f"model={args.model} images={len(paths)} threshold={args.threshold}")
    print(f"  reference detections : {n_ref}")
    print(f"  onnx detections      : {n_onnx}")
    print(f"  matched (IoU>={args.iou_match}, same class): {n_matched}")
    if ious:
        print(f"  mean IoU of matches  : {np.mean(ious):.4f}")
    recall = n_matched / n_ref if n_ref else 1.0
    print(f"  ref-recall by onnx   : {recall:.4f}")


if __name__ == "__main__":
    main()
