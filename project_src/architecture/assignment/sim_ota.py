import torch
import torch.nn.functional as F

_INF = float('inf')


def _decode_ltrb(ltrb: torch.Tensor, anchor_points: torch.Tensor) -> torch.Tensor:
    """Decode ltrb distance predictions to xyxy boxes.

    Args:
        ltrb:          (P, 4) left/top/right/bottom distances from anchor center.
        anchor_points: (P, 2) xy center coordinates of each prediction.

    Returns:
        (P, 4) boxes in xyxy format.
    """
    ltrb = ltrb.clamp(min=0)
    x1 = anchor_points[:, 0] - ltrb[:, 0]
    y1 = anchor_points[:, 1] - ltrb[:, 1]
    x2 = anchor_points[:, 0] + ltrb[:, 2]
    y2 = anchor_points[:, 1] + ltrb[:, 3]
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _pairwise_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU between M and N boxes.

    Args:
        boxes_a: (M, 4) xyxy
        boxes_b: (N, 4) xyxy

    Returns:
        (M, N) IoU matrix.
    """
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    inter_x1 = torch.max(boxes_a[:, None, 0], boxes_b[None, :, 0])
    inter_y1 = torch.max(boxes_a[:, None, 1], boxes_b[None, :, 1])
    inter_x2 = torch.min(boxes_a[:, None, 2], boxes_b[None, :, 2])
    inter_y2 = torch.min(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-6)


class BaseAssigner:
    """Abstract base class for label assignment strategies."""

    def assign(
        self,
        cls_logits: torch.Tensor,
        bbox_ltrb: torch.Tensor,
        anchor_points: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_boxes: torch.Tensor,
    ) -> dict:
        """Assign predictions to ground-truth objects for a single image.

        Args:
            cls_logits:    (P, C) raw classification logits.
            bbox_ltrb:     (P, 4) ltrb distance predictions.
            anchor_points: (P, 2) xy center of each prediction location.
            gt_labels:     (G,)  ground-truth class indices.
            gt_boxes:      (G, 4) ground-truth boxes in xyxy format.

        Returns:
            dict with keys (tensors contain only positive/matched samples):
                'cls_logits': (num_pos, C)
                'bbox_preds': (num_pos, 4) decoded xyxy
                'gt_labels':  (num_pos,)
                'gt_boxes':   (num_pos, 4)
        """
        raise NotImplementedError


class SimOTA(BaseAssigner):
    """Simplified Optimal Transport Assignment (SimOTA).

    For each GT box, a dynamic number of predictions (k) are selected as
    positive samples based on a combined classification + regression cost.
    Prediction-GT conflicts are resolved by lowest cost.

    Reference: YOLOX, "Exceeding Yolo Series Detectors", arXiv:2107.08430.

    Args:
        topk:       Candidate pool size used to estimate dynamic k per GT.
        iou_weight: Weight of the IoU term in the cost matrix.
        cls_weight: Weight of the classification BCE term in the cost matrix.
    """

    def __init__(self, args):
        self.topk = getattr(args, 'topk', 10)
        self.iou_weight = getattr(args, 'iou_weight', 3.0)
        self.cls_weight = getattr(args, 'cls_weight', 1.0)

    def assign(
        self,
        cls_logits: torch.Tensor,
        bbox_ltrb: torch.Tensor,
        anchor_points: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_boxes: torch.Tensor,
    ) -> dict:
        num_preds, num_cls = cls_logits.shape
        num_gt = gt_labels.shape[0]
        device = cls_logits.device

        def _empty():
            return dict(
                cls_logits=cls_logits.new_zeros(0, num_cls),
                bbox_preds=cls_logits.new_zeros(0, 4),
                gt_labels=gt_labels.new_zeros(0),
                gt_boxes=gt_boxes.new_zeros(0, 4),
            )

        if num_gt == 0:
            return _empty()

        decoded = _decode_ltrb(bbox_ltrb, anchor_points)  # (P, 4)

        # ── Candidate mask: anchor center must lie inside the GT box ─────────
        ax, ay = anchor_points[:, 0], anchor_points[:, 1]
        candidate_mask = (
            (ax[:, None] > gt_boxes[None, :, 0]) &
            (ay[:, None] > gt_boxes[None, :, 1]) &
            (ax[:, None] < gt_boxes[None, :, 2]) &
            (ay[:, None] < gt_boxes[None, :, 3])
        )  # (P, G)

        if not candidate_mask.any():
            return _empty()

        # ── Cost matrix ──────────────────────────────────────────────────────
        iou_matrix = _pairwise_iou(decoded, gt_boxes)  # (P, G)

        with torch.no_grad():
            probs = cls_logits.sigmoid()  # (P, C)
            gt_onehot = F.one_hot(gt_labels.long(), num_cls).float()  # (G, C)
            cls_cost = -(
                gt_onehot[None] * (probs[:, None] + 1e-8).log() +
                (1 - gt_onehot[None]) * (1 - probs[:, None] + 1e-8).log()
            ).sum(dim=-1)  # (P, G)

        cost_matrix = self.cls_weight * cls_cost + self.iou_weight * (1.0 - iou_matrix)
        cost_matrix[~candidate_mask] = _INF

        # ── Dynamic k: sum of top-topk IoU values per GT ─────────────────────
        topk_k = min(self.topk, num_preds)
        topk_iou, _ = torch.topk(iou_matrix, topk_k, dim=0)  # (topk_k, G)
        dynamic_k = topk_iou.sum(dim=0).int().clamp(min=1)   # (G,)

        # ── OTA selection with conflict resolution ───────────────────────────
        matching = torch.zeros(num_preds, num_gt, device=device)
        for g in range(num_gt):
            k = min(int(dynamic_k[g]), int((cost_matrix[:, g] < _INF).sum()))
            if k == 0:
                continue
            _, top_idx = torch.topk(cost_matrix[:, g], k=k, largest=False)
            matching[top_idx, g] = 1.0

        # Predictions claimed by more than one GT → assign to lowest-cost GT
        conflict = matching.sum(dim=1) > 1
        if conflict.any():
            _, best_gt = cost_matrix[conflict].min(dim=1)
            matching[conflict] = 0.0
            matching[conflict, best_gt] = 1.0

        # ── Gather positives ─────────────────────────────────────────────────
        pos_mask = matching.sum(dim=1) > 0       # (P,)
        gt_idx = matching[pos_mask].argmax(dim=1) # (num_pos,)

        return dict(
            cls_logits=cls_logits[pos_mask],      # (num_pos, C)
            bbox_preds=decoded[pos_mask],          # (num_pos, 4) xyxy
            gt_labels=gt_labels[gt_idx],           # (num_pos,)
            gt_boxes=gt_boxes[gt_idx],             # (num_pos, 4)
        )
