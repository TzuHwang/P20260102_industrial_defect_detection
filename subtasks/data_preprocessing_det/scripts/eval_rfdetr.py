"""Evaluate a fine-tuned RF-DETR checkpoint against the project's detection targets:
macro AUROC >= 0.99 and mAP >= 0.8.

mAP reuses the project's existing MeanAveragePrecision metric
(project_src/analyst/metrics/map.py), fed with the same
[x1, y1, x2, y2, score, class_id] / [x1, y1, x2, y2, class_id] arrays that
LitModel._accumulate_det builds for the Lightning detection pipeline.

Macro AUROC is an image-level multi-label metric defined as: for each image
and class, take the highest detection confidence for that class (0 if no
detection), and whether the image actually contains that defect class as the
ground-truth label; stack into (N_images, num_classes) matrices and score with
sklearn.metrics.roc_auc_score(average='macro') — mirroring how the original
classification task's macro AUC is computed.

Usage:
    python -m subtasks.data_preprocessing_det.scripts.eval_rfdetr \
        --checkpoint outputs/rfdetr_medium_front/checkpoint_best_total.pth \
        --dataset-dir data/internal_train/rfdetr_coco_front \
        --split test --batch-size 16 --score-threshold 0.001
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.metrics import roc_auc_score

from rfdetr import RFDETR
from project_src.analyst.metrics.map import MeanAveragePrecision


def _load_split(dataset_dir, split):
    """Return (image_paths, gt_boxes_per_image[xyxy+class], class_names)."""
    split_dir = Path(dataset_dir) / split
    coco = json.load(open(split_dir / '_annotations.coco.json', encoding='utf-8'))
    class_names = [c['name'] for c in sorted(coco['categories'], key=lambda c: c['id'])]

    anns_by_image = {}
    for ann in coco['annotations']:
        x, y, w, h = ann['bbox']
        box = [x, y, x + w, y + h, ann['category_id'] - 1]   # COCO category_id is 1-based
        anns_by_image.setdefault(ann['image_id'], []).append(box)

    image_paths, gt_per_image = [], []
    for img in sorted(coco['images'], key=lambda im: im['id']):
        image_paths.append(str(split_dir / img['file_name']))
        gt_per_image.append(np.asarray(anns_by_image.get(img['id'], []), dtype=np.float32).reshape(-1, 5))

    return image_paths, gt_per_image, class_names


def _batched(seq, batch_size):
    for i in range(0, len(seq), batch_size):
        yield seq[i:i + batch_size]


def evaluate(checkpoint, dataset_dir, split, batch_size, score_threshold, iou_thresholds):
    image_paths, gt_per_image, class_names = _load_split(dataset_dir, split)
    num_classes = len(class_names)
    model = RFDETR.from_checkpoint(checkpoint)

    preds_list, score_matrix, gt_matrix = [], [], []
    for batch_paths in _batched(image_paths, batch_size):
        batch_dets = model.predict(batch_paths, threshold=score_threshold)
        if not isinstance(batch_dets, list):
            batch_dets = [batch_dets]
        for det in batch_dets:
            keep = det.class_id < num_classes  # drop the "no-object" background slot
            xyxy, confidence, class_id = det.xyxy[keep], det.confidence[keep], det.class_id[keep]

            if len(xyxy):
                pred_arr = np.column_stack([xyxy, confidence, class_id]).astype(np.float32)
            else:
                pred_arr = np.zeros((0, 6), dtype=np.float32)
            preds_list.append(pred_arr)

            class_score = np.zeros(num_classes, dtype=np.float32)
            for cls_id, conf in zip(class_id, confidence):
                class_score[int(cls_id)] = max(class_score[int(cls_id)], float(conf))
            score_matrix.append(class_score)

    for gt in gt_per_image:
        presence = np.zeros(num_classes, dtype=np.int32)
        if len(gt):
            presence[gt[:, 4].astype(int)] = 1
        gt_matrix.append(presence)

    map_metric = MeanAveragePrecision(SimpleNamespace(num_classes=num_classes, iou_thresholds=iou_thresholds))
    map_value = map_metric(preds_list, gt_per_image)

    score_matrix = np.stack(score_matrix)
    gt_matrix = np.stack(gt_matrix)

    # AUROC is undefined for classes with only one ground-truth label value
    # (e.g. a defect that never/always appears in this split) - skip those.
    valid = [c for c in range(num_classes) if len(np.unique(gt_matrix[:, c])) > 1]
    skipped = [class_names[c] for c in range(num_classes) if c not in valid]
    auroc_value = roc_auc_score(gt_matrix[:, valid], score_matrix[:, valid], average='macro')

    print(f'split={split}  images={len(image_paths)}')
    print(f'mAP            = {map_value:.4f}  (target >= 0.80)')
    print(f'macro AUROC    = {auroc_value:.4f}  (target >= 0.99)  [{len(valid)}/{num_classes} classes]')
    if skipped:
        print(f'  skipped (single-class ground truth): {skipped}')
    print('per-class AUROC:')
    for c in valid:
        n_pos = int(gt_matrix[:, c].sum())
        cls_auroc = roc_auc_score(gt_matrix[:, c], score_matrix[:, c])
        print(f'  {class_names[c]:20s} AUROC={cls_auroc:.4f}  (positives={n_pos}/{len(image_paths)})')
    return map_value, auroc_value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset-dir', required=True)
    parser.add_argument('--split', default='test', choices=['train', 'valid', 'test'])
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--score-threshold', type=float, default=0.001)
    parser.add_argument('--iou-thresholds', type=float, nargs='+', default=None,
                        help='e.g. --iou-thresholds 0.5  for Pascal-VOC style mAP@50; default: COCO [.5:.95]')
    args = parser.parse_args()
    evaluate(args.checkpoint, args.dataset_dir, args.split, args.batch_size,
             args.score_threshold, args.iou_thresholds)
