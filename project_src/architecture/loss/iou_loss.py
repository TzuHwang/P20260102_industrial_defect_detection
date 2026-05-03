import torch
import torch.nn as nn


def _box_iou_elementwise(boxes_a: torch.Tensor, boxes_b: torch.Tensor):
    """Elementwise IoU and union for matched pairs of boxes.

    Args:
        boxes_a: (N, 4) in [x1, y1, x2, y2] format.
        boxes_b: (N, 4) in [x1, y1, x2, y2] format.

    Returns:
        iou   : (N,) IoU values.
        union : (N,) union areas (needed by GIoU).
    """
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    inter_x1 = torch.maximum(boxes_a[:, 0], boxes_b[:, 0])
    inter_y1 = torch.maximum(boxes_a[:, 1], boxes_b[:, 1])
    inter_x2 = torch.minimum(boxes_a[:, 2], boxes_b[:, 2])
    inter_y2 = torch.minimum(boxes_a[:, 3], boxes_b[:, 3])

    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    union = area_a + area_b - inter
    return inter / union.clamp(min=1e-6), union


class IoULoss(nn.Module):
    """Box regression loss: 1 - IoU(pred, target).

    Both tensors must be in [x1, y1, x2, y2] format and matched pairwise (N, 4).

    Args:
        reduction: 'mean' | 'sum' | 'none'.
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        iou, _ = _box_iou_elementwise(predictions, targets)
        loss = 1.0 - iou
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


class GIoULoss(nn.Module):
    """Generalized IoU loss: 1 - GIoU(pred, target).

    GIoU subtracts the gap between the enclosing box and the union from IoU,
    providing a gradient signal even for completely non-overlapping boxes.

    Both tensors must be in [x1, y1, x2, y2] format and matched pairwise (N, 4).

    Reference: Rezatofighi et al., "Generalized Intersection over Union", CVPR 2019.

    Args:
        reduction: 'mean' | 'sum' | 'none'.
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        iou, union = _box_iou_elementwise(predictions, targets)

        enc_x1 = torch.minimum(predictions[:, 0], targets[:, 0])
        enc_y1 = torch.minimum(predictions[:, 1], targets[:, 1])
        enc_x2 = torch.maximum(predictions[:, 2], targets[:, 2])
        enc_y2 = torch.maximum(predictions[:, 3], targets[:, 3])
        enc_area = (enc_x2 - enc_x1).clamp(min=0) * (enc_y2 - enc_y1).clamp(min=0)

        giou = iou - (enc_area - union) / enc_area.clamp(min=1e-6)
        loss = 1.0 - giou
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss
