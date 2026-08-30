import torch

from .sim_ota import SimOTA
from .faster_rcnn_assigner import FasterRCNNAssigner


class AssignmentFactory:
    """Wraps an assigner and handles batch-level assignment.

    assign_batch() always returns a list of (predictions, targets) pairs —
    one pair per loss function in the same order as LossFactory.losses.

    SimOTA  → [(cls_logits, gt_labels), (bbox_preds, gt_boxes)]
    FasterRCNN → [(rpn_cls, rpn_cls_tgt), (rpn_reg, rpn_reg_tgt),
                  (roi_cls, roi_cls_tgt), (roi_reg, roi_reg_tgt)]

    Args:
        args: Config namespace with:
              - assigner (str): 'SimOTA' | 'FasterRCNN'
              - strides  (list[int]): FPN strides (SimOTA only).
              - (assigner-specific params forwarded to the assigner)
    """

    _registry = {'SimOTA': SimOTA, 'FasterRCNN': FasterRCNNAssigner}

    def __init__(self, args):
        name = getattr(args, 'assigner', 'SimOTA')
        if name not in self._registry:
            raise ValueError(f"Unknown assigner: '{name}'. Available: {list(self._registry)}")
        self.assigner = self._registry[name](args)
        self.strides = getattr(args, 'strides', [])
        self._is_faster_rcnn = (name == 'FasterRCNN')

    def assign_batch(self, head_output: dict, targets: dict) -> list:
        """Return list of (pred, tgt) pairs, one per loss function."""
        if self._is_faster_rcnn:
            return self.assigner.assign_batch(head_output, targets)
        return self._simota_assign_batch(head_output, targets)

    # ------------------------------------------------------------------
    # SimOTA path
    # ------------------------------------------------------------------
    def _simota_assign_batch(self, head_output: dict, targets: dict) -> list:
        cls_per_level = head_output['cls_scores']
        box_per_level = head_output['bbox_preds']
        N = cls_per_level[0].shape[0]
        device = cls_per_level[0].device

        flat_cls, flat_box, anchor_pts = self._flatten(cls_per_level, box_per_level, device)
        num_boxes = targets['num_boxes']
        all_cls_logits, all_cls_targets = [], []
        all_box, all_gt = [], []

        for i in range(N):
            n = int(num_boxes[i]) if isinstance(num_boxes, torch.Tensor) else int(num_boxes)
            result = self.assigner.assign(
                flat_cls[i], flat_box[i], anchor_pts,
                targets['labels'][i, :n], targets['boxes'][i, :n],
            )
            # cls_logits contains only positive predictions (one-hot sigmoid targets)
            all_cls_logits.append(result['cls_logits'])
            all_cls_targets.append(result['cls_targets'])
            if result['bbox_preds'].shape[0] > 0:
                all_box.append(result['bbox_preds'])
                all_gt.append(result['gt_boxes'])

        cls_logits_cat = torch.cat(all_cls_logits, dim=0)   # (N*P, C)
        cls_targets_cat = torch.cat(all_cls_targets, dim=0)  # (N*P, C)

        if all_box:
            bbox_preds_cat = torch.cat(all_box, dim=0)
            gt_boxes_cat = torch.cat(all_gt, dim=0)
        else:
            bbox_preds_cat = flat_cls.new_zeros(0, 4)
            gt_boxes_cat = flat_cls.new_zeros(0, 4)

        return [
            (cls_logits_cat, cls_targets_cat),   # (num_pos, C) one-hot float → sigmoid binary FocalLoss
            (bbox_preds_cat, gt_boxes_cat),       # (num_pos, 4) → GIoULoss
        ]

    def _flatten(self, cls_per_level, box_per_level, device):
        cls_parts, box_parts, anchor_parts = [], [], []
        for level, (c, b) in enumerate(zip(cls_per_level, box_per_level)):
            N, C, H, W = c.shape
            stride = self.strides[level]
            cls_parts.append(c.permute(0, 2, 3, 1).reshape(N, -1, C))
            box_parts.append(b.permute(0, 2, 3, 1).reshape(N, -1, 4))
            y = (torch.arange(H, device=device, dtype=torch.float32) + 0.5) * stride
            x = (torch.arange(W, device=device, dtype=torch.float32) + 0.5) * stride
            gy, gx = torch.meshgrid(y, x, indexing='ij')
            anchor_parts.append(torch.stack([gx, gy], dim=-1).reshape(-1, 2))
        return (torch.cat(cls_parts, dim=1),
                torch.cat(box_parts, dim=1),
                torch.cat(anchor_parts, dim=0))
