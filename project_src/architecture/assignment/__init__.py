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
        all_cls, all_box, all_lbl, all_gt = [], [], [], []

        for i in range(N):
            n = int(num_boxes[i]) if isinstance(num_boxes, torch.Tensor) else int(num_boxes)
            if n == 0:
                continue
            result = self.assigner.assign(
                flat_cls[i], flat_box[i], anchor_pts,
                targets['labels'][i, :n], targets['boxes'][i, :n],
            )
            if result['cls_logits'].shape[0] == 0:
                continue
            all_cls.append(result['cls_logits'])
            all_box.append(result['bbox_preds'])
            all_lbl.append(result['gt_labels'])
            all_gt.append(result['gt_boxes'])

        if not all_cls:
            C = flat_cls.shape[-1]
            return [
                (flat_cls.new_zeros(0, C), targets['labels'].new_zeros(0)),
                (flat_box.new_zeros(0, 4), targets['boxes'].new_zeros(0, 4)),
            ]

        return [
            (torch.cat(all_cls, dim=0), torch.cat(all_lbl, dim=0)),
            (torch.cat(all_box, dim=0), torch.cat(all_gt, dim=0)),
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
