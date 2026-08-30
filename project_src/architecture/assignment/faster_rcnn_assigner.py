import torch

from .sim_ota import _pairwise_iou
from ..models.head.faster_rcnn_head import _encode_boxes


def _sample_indices(pos_idx, neg_idx, num_samples: int, pos_fraction: float, device):
    """Subsample positive and negative indices to a fixed total size."""
    num_pos = min(int(num_samples * pos_fraction), len(pos_idx))
    num_neg = num_samples - num_pos

    perm_pos = torch.randperm(len(pos_idx), device=device)[:num_pos]
    perm_neg = torch.randperm(len(neg_idx), device=device)[:num_neg]
    return pos_idx[perm_pos], neg_idx[perm_neg]


class FasterRCNNAssigner:
    """Two-stage label assignment for Faster R-CNN.

    Matches anchors to GT boxes for RPN training and proposals to GT boxes
    for RoI-head training.

    Returns a list of four (predictions, targets) pairs consumed by
    LossFactory in order: [RPN-cls, RPN-reg, RoI-cls, RoI-reg].

    Args:
        rpn_pos_iou:   IoU threshold to call an anchor positive for RPN.
        rpn_neg_iou:   IoU threshold to call an anchor negative for RPN.
        rpn_samples:   Total anchors sampled per image for RPN loss.
        rpn_pos_frac:  Fraction of sampled anchors that are positive.
        roi_pos_iou:   IoU threshold for a proposal to be a positive RoI.
        roi_neg_iou_hi: Upper IoU for negative RoI (proposal is negative below this).
        roi_neg_iou_lo: Lower IoU for negative RoI (ignore proposals below this).
        roi_samples:   Total RoIs sampled per image for RoI loss.
        roi_pos_frac:  Fraction of sampled RoIs that are positive.
    """

    def __init__(self, args):
        self.rpn_pos_iou = getattr(args, 'rpn_pos_iou', 0.7)
        self.rpn_neg_iou = getattr(args, 'rpn_neg_iou', 0.3)
        self.rpn_samples = getattr(args, 'rpn_samples', 256)
        self.rpn_pos_frac = getattr(args, 'rpn_pos_frac', 0.5)

        self.roi_pos_iou = getattr(args, 'roi_pos_iou', 0.5)
        self.roi_neg_iou_hi = getattr(args, 'roi_neg_iou_hi', 0.5)
        self.roi_neg_iou_lo = getattr(args, 'roi_neg_iou_lo', 0.0)
        self.roi_samples = getattr(args, 'roi_samples', 512)
        self.roi_pos_frac = getattr(args, 'roi_pos_frac', 0.25)

    # ------------------------------------------------------------------
    def assign_batch(self, head_output: dict, targets: dict):
        """
        Args:
            head_output: dict from FasterRCNNHead.forward (see head docstring).
            targets: batched TapeMeasureDetection targets:
                'labels':    (N, max_boxes)  int64, -1 = padding
                'boxes':     (N, max_boxes, 4) float32, xyxy
                'num_boxes': (N,)

        Returns:
            List of 4 (predictions, targets_tensor) pairs:
                [0] RPN cls:  (sampled_anchors, 1)  logits, (sampled_anchors,) binary labels
                [1] RPN reg:  (pos_anchors, 4)       deltas, (pos_anchors, 4)  delta targets
                [2] RoI cls:  (sampled_rois, C+1)    logits, (sampled_rois,)   class labels
                [3] RoI reg:  (pos_rois, 4)           deltas, (pos_rois, 4)    delta targets
        """
        anchors = head_output['anchors_flat']        # (A, 4)
        rpn_cls = head_output['rpn_cls_flat']        # (N, A)
        rpn_reg = head_output['rpn_reg_flat']        # (N, A, 4)
        proposals = head_output['proposals_cat']     # (total_P, 4)
        roi_img_ids = head_output['roi_img_ids']     # (total_P,)
        roi_cls_scores = head_output['roi_cls_scores']  # (total_P, C+1)
        roi_bbox_preds = head_output['roi_bbox_preds']  # (total_P, 4)

        N = rpn_cls.shape[0]
        device = anchors.device
        num_boxes = targets['num_boxes']

        rpn_cls_preds, rpn_cls_tgts = [], []
        rpn_reg_preds, rpn_reg_tgts = [], []
        roi_cls_preds, roi_cls_tgts = [], []
        roi_reg_preds, roi_reg_tgts = [], []

        for i in range(N):
            n = int(num_boxes[i]) if isinstance(num_boxes, torch.Tensor) else int(num_boxes)
            gt_boxes_i = targets['boxes'][i, :n]      # (n, 4)
            gt_labels_i = targets['labels'][i, :n]    # (n,)

            # ── RPN assignment ──────────────────────────────────────
            rpn_c, rpn_r = self._assign_rpn(
                anchors, rpn_cls[i], rpn_reg[i], gt_boxes_i, device)
            rpn_cls_preds.append(rpn_c[0])
            rpn_cls_tgts.append(rpn_c[1])
            rpn_reg_preds.append(rpn_r[0])
            rpn_reg_tgts.append(rpn_r[1])

            # ── RoI assignment ──────────────────────────────────────
            mask_i = roi_img_ids == i
            props_i = proposals[mask_i]                # (P_i, 4)
            cls_i = roi_cls_scores[mask_i]             # (P_i, C+1)
            reg_i = roi_bbox_preds[mask_i]             # (P_i, 4)

            roi_c, roi_r = self._assign_roi(
                props_i, cls_i, reg_i, gt_boxes_i, gt_labels_i, device)
            roi_cls_preds.append(roi_c[0])
            roi_cls_tgts.append(roi_c[1])
            roi_reg_preds.append(roi_r[0])
            roi_reg_tgts.append(roi_r[1])

        def _cat(lst):
            return torch.cat(lst, dim=0) if lst else torch.zeros(0, device=device)

        return [
            (_cat(rpn_cls_preds).unsqueeze(-1), _cat(rpn_cls_tgts).float().unsqueeze(-1)),
            (_cat(rpn_reg_preds), _cat(rpn_reg_tgts)),
            (_cat(roi_cls_preds), _cat(roi_cls_tgts)),
            (_cat(roi_reg_preds), _cat(roi_reg_tgts)),
        ]

    # ------------------------------------------------------------------
    def _assign_rpn(self, anchors, cls_logits, reg_deltas, gt_boxes, device):
        """Match anchors to GT boxes for RPN.

        Returns:
            cls_pair: (sampled_cls_logits (K,), sampled_binary_labels (K,))
            reg_pair: (positive_reg_deltas (pos, 4), delta_targets (pos, 4))
        """
        A = anchors.shape[0]

        if gt_boxes.shape[0] == 0:
            # No GT: all anchors are negative
            neg_idx = torch.arange(A, device=device)
            k = min(self.rpn_samples, A)
            perm = torch.randperm(A, device=device)[:k]
            cls_logits_s = cls_logits[perm]
            cls_tgts_s = torch.zeros(k, device=device)
            return ((cls_logits_s, cls_tgts_s),
                    (cls_logits.new_zeros(0, 4), cls_logits.new_zeros(0, 4)))

        iou = _pairwise_iou(anchors, gt_boxes)          # (A, G)
        best_iou_per_anchor, best_gt_per_anchor = iou.max(dim=1)

        # Every GT's highest-IoU anchor is always positive
        _, best_anchor_per_gt = iou.max(dim=0)

        labels = torch.full((A,), -1, dtype=torch.long, device=device)
        labels[best_iou_per_anchor >= self.rpn_pos_iou] = 1
        labels[best_iou_per_anchor < self.rpn_neg_iou] = 0
        labels[best_anchor_per_gt] = 1                  # guarantee each GT has ≥1 positive

        pos_idx = (labels == 1).nonzero(as_tuple=True)[0]
        neg_idx = (labels == 0).nonzero(as_tuple=True)[0]
        pos_s, neg_s = _sample_indices(pos_idx, neg_idx, self.rpn_samples,
                                       self.rpn_pos_frac, device)
        sampled = torch.cat([pos_s, neg_s])

        cls_logits_s = cls_logits[sampled]
        cls_tgts_s = labels[sampled].float().clamp(min=0)   # -1 removed, only 0/1

        # Regression targets for positive anchors only
        delta_tgts = _encode_boxes(anchors[pos_s], gt_boxes[best_gt_per_anchor[pos_s]])

        return ((cls_logits_s, cls_tgts_s),
                (reg_deltas[pos_s], delta_tgts))

    # ------------------------------------------------------------------
    def _assign_roi(self, proposals, cls_scores, bbox_preds, gt_boxes, gt_labels, device):
        """Match proposals to GT boxes for RoI head.

        Returns:
            cls_pair: (sampled_cls_scores (K, C+1), sampled_class_labels (K,))
            reg_pair: (positive_bbox_preds (pos, 4), delta_targets (pos, 4))
        """
        P = proposals.shape[0]

        if P == 0 or gt_boxes.shape[0] == 0:
            return ((cls_scores.new_zeros(0, cls_scores.shape[-1]),
                     gt_labels.new_zeros(0)),
                    (bbox_preds.new_zeros(0, 4),
                     bbox_preds.new_zeros(0, 4)))

        iou = _pairwise_iou(proposals, gt_boxes)            # (P, G)
        best_iou_per_prop, best_gt_per_prop = iou.max(dim=1)

        labels = torch.full((P,), -1, dtype=torch.long, device=device)
        labels[best_iou_per_prop >= self.roi_pos_iou] = 1
        labels[(best_iou_per_prop >= self.roi_neg_iou_lo) &
               (best_iou_per_prop < self.roi_neg_iou_hi)] = 0

        pos_idx = (labels == 1).nonzero(as_tuple=True)[0]
        neg_idx = (labels == 0).nonzero(as_tuple=True)[0]

        if len(pos_idx) == 0 and len(neg_idx) == 0:
            return ((cls_scores.new_zeros(0, cls_scores.shape[-1]),
                     gt_labels.new_zeros(0)),
                    (bbox_preds.new_zeros(0, 4), bbox_preds.new_zeros(0, 4)))

        pos_s, neg_s = _sample_indices(pos_idx, neg_idx, self.roi_samples,
                                       self.roi_pos_frac, device)
        sampled = torch.cat([pos_s, neg_s])

        cls_scores_s = cls_scores[sampled]

        # Class targets: positive → GT class, negative → 0 (background)
        cls_tgts = torch.zeros(len(sampled), dtype=torch.long, device=device)
        if len(pos_s) > 0:
            cls_tgts[:len(pos_s)] = gt_labels[best_gt_per_prop[pos_s]].long()

        # Regression targets for positive RoIs only
        if len(pos_s) > 0:
            delta_tgts = _encode_boxes(proposals[pos_s], gt_boxes[best_gt_per_prop[pos_s]])
        else:
            delta_tgts = bbox_preds.new_zeros(0, 4)

        return ((cls_scores_s, cls_tgts),
                (bbox_preds[pos_s], delta_tgts))
