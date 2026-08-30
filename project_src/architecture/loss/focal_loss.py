import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in classification tasks.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = None, reduction: str = 'mean'):
        """
        Args:
            gamma: Focusing parameter. Higher values down-weight easy examples more.
            alpha: Weighting factor for the rare class. If None, no class weighting.
            reduction: 'mean', 'sum', or 'none'.
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: Raw logits.
                - (N, C) with C > 1 and targets float (N, C): sigmoid binary per class
                  (detection style — one-hot for positives, zeros for negatives).
                - (N, C) with C > 1 and targets long (N,): softmax multi-class
                  (classification style).
                - (N,) or (N, 1): single binary logit.

        Returns:
            Focal loss scalar (or per-sample tensor if reduction='none').
        """
        if predictions.dim() == 2 and predictions.shape[1] > 1 and targets.dim() == 2:
            # Detection binary-sigmoid path: targets are (N, C) float one-hot / zero.
            # Each class is treated as an independent binary classifier.
            bce = F.binary_cross_entropy_with_logits(predictions, targets.float(),
                                                     reduction='none')  # (N, C)
            p_t = torch.exp(-bce)
            focal_weight = (1 - p_t) ** self.gamma
            if self.alpha is not None:
                alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
                focal_weight = focal_weight * alpha_t
            loss = (focal_weight * bce).sum(dim=-1)  # sum over classes, keep per-sample
        elif predictions.dim() == 1 or predictions.shape[1] == 1:
            # Scalar binary case
            predictions = predictions.view(-1)
            targets = targets.float().view(-1)
            bce = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
            p_t = torch.exp(-bce)
            focal_weight = (1 - p_t) ** self.gamma
            if self.alpha is not None:
                focal_weight = focal_weight * (self.alpha * targets + (1 - self.alpha) * (1 - targets))
            loss = focal_weight * bce
        else:
            # Multi-class softmax case (classification tasks)
            log_p = F.log_softmax(predictions, dim=1)
            ce = F.nll_loss(log_p, targets.long(), reduction='none')
            p_t = torch.exp(-ce)
            focal_weight = (1 - p_t) ** self.gamma
            if self.alpha is not None:
                focal_weight = focal_weight * self.alpha
            loss = focal_weight * ce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
