import numpy as np

__all__ = ['MeanAveragePrecision']


def _box_iou(boxes_a, boxes_b):
    """(N,4) × (M,4) → (N,M) IoU matrix."""
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    inter_x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    inter_y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    inter_x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    inter_y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(inter_y2 - inter_y1, 0)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def _compute_ap(recalls, precisions):
    """Area under PR curve via all-points interpolation (COCO style)."""
    r = np.concatenate([[0.0], recalls, [1.0]])
    p = np.concatenate([[0.0], precisions, [0.0]])
    for i in range(len(p) - 2, -1, -1):
        p[i] = max(p[i], p[i + 1])
    idx = np.where(r[1:] != r[:-1])[0]
    return float(np.sum((r[idx + 1] - r[idx]) * p[idx + 1]))


def _ap_at_iou(preds, gts, iou_threshold, num_classes):
    """Per-class AP at a single IoU threshold, averaged over classes."""
    aps = []
    for cls in range(num_classes):
        # Collect per-image GT boxes and all predictions for this class
        cls_gts = []
        cls_det = []   # (score, img_idx, det_row_idx)
        n_gt = 0

        for img_idx, (pred, gt) in enumerate(zip(preds, gts)):
            gt_mask = gt[:, 4].astype(int) == cls if len(gt) else np.array([], dtype=bool)
            cls_gts.append(gt[gt_mask, :4])
            n_gt += gt_mask.sum()

            if len(pred):
                det_mask = pred[:, 5].astype(int) == cls
                for i in np.where(det_mask)[0]:
                    cls_det.append((pred[i, 4], img_idx, i))

        if n_gt == 0:
            continue

        cls_det.sort(key=lambda x: -x[0])   # sort by score descending
        matched = [np.zeros(len(g), dtype=bool) for g in cls_gts]
        tp = np.zeros(len(cls_det))
        fp = np.zeros(len(cls_det))

        for rank, (_, img_idx, row_i) in enumerate(cls_det):
            gt_boxes = cls_gts[img_idx]
            if len(gt_boxes) == 0:
                fp[rank] = 1
                continue

            iou = _box_iou(preds[img_idx][row_i, :4][None], gt_boxes)[0]
            best = iou.argmax()
            if iou[best] >= iou_threshold and not matched[img_idx][best]:
                tp[rank] = 1
                matched[img_idx][best] = True
            else:
                fp[rank] = 1

        cum_tp = tp.cumsum()
        cum_fp = fp.cumsum()
        recalls    = cum_tp / n_gt
        precisions = cum_tp / (cum_tp + cum_fp + 1e-6)
        aps.append(_compute_ap(recalls, precisions))

    return np.mean(aps) if aps else 0.0


class MeanAveragePrecision:
    """mAP for object detection.

    Args (via args namespace)
    -------------------------
    num_classes    : int
    iou_thresholds : list[float] | None
        None → COCO-style [0.50, 0.55, …, 0.95]
        [0.5] → Pascal-VOC style mAP@50

    Call signature
    --------------
    outputs : list[np.ndarray]  shape (N_det, 6)  [x1, y1, x2, y2, score, class_id]
    targets : list[np.ndarray]  shape (N_gt,  5)  [x1, y1, x2, y2, class_id]

    Returns
    -------
    float : mAP averaged over IoU thresholds and classes, rounded to 4 dp.
    """

    def __init__(self, args):
        self.num_classes = args.num_classes
        thresholds = getattr(args, 'iou_thresholds', None)
        if thresholds is None:
            thresholds = np.round(np.arange(0.5, 1.0, 0.05), 2).tolist()
        self.iou_thresholds = thresholds

    def __call__(self, outputs, targets):
        preds = [np.asarray(p) if len(p) else np.zeros((0, 6)) for p in outputs]
        gts = [np.asarray(g) if len(g) else np.zeros((0, 5)) for g in targets]

        per_threshold = [
            _ap_at_iou(preds, gts, t, self.num_classes)
            for t in self.iou_thresholds
        ]
        return np.round(np.mean(per_threshold), 4)
