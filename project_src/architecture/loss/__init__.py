import torch
import torch.nn as nn

from .focal_loss import FocalLoss
from .iou_loss import GIoULoss, IoULoss


class LossFactory:
    # Loss functions with built-in activation — expect raw logits
    LOGIT_LOSSES = {'CrossEntropyLoss', 'BCEWithLogitsLoss', 'FocalLoss'}

    # When head output is a dict, which key each loss reads
    _INPUT_KEY = {
        'FocalLoss': 'cls_scores',
        'CrossEntropyLoss': 'cls_scores',
        'BCEWithLogitsLoss': 'cls_scores',
        'BCELoss': 'cls_scores',
        'IoULoss': 'bbox_preds',
        'GIoULoss': 'bbox_preds',
    }
    # When target is a dict, which key each loss reads
    _TARGET_KEY = {
        'FocalLoss': 'labels',
        'CrossEntropyLoss': 'labels',
        'BCEWithLogitsLoss': 'labels',
        'BCELoss': 'labels',
        'IoULoss': 'boxes',
        'GIoULoss': 'boxes',
    }

    def __init__(self, args):
        self.losses = args.losses
        self.channel_weights = args.channel_weights
        self.weights = args.loss_weights
        self.loss_fcns = {}
        self.loss_values = {}
        self.in_channel_loss_values = {}
        self.assigner = None  # set via set_assigner() for detection tasks

        # Initialize loss functions
        self._init_loss_functions()

    def _init_loss_functions(self):
        """Initialize the loss functions based on the specified losses."""
        loss_mapping = {
            'CrossEntropyLoss': nn.CrossEntropyLoss,
            'BCELoss': nn.BCELoss,
            'BCEWithLogitsLoss': nn.BCEWithLogitsLoss,
            'MSELoss': nn.MSELoss,
            'L1Loss': nn.L1Loss,
            'SmoothL1Loss': nn.SmoothL1Loss,
            'FocalLoss': FocalLoss,
            'IoULoss': IoULoss,
            'GIoULoss': GIoULoss,
        }

        for loss_name in self.losses:
            if loss_name in loss_mapping:
                self.loss_fcns[loss_name] = loss_mapping[loss_name]()
            else:
                raise ValueError(f"Unknown loss function: {loss_name}")

    def get_loss_fcns(self):
        return self.loss_fcns

    def set_assigner(self, assigner) -> None:
        """Attach an AssignmentFactory for detection training."""
        self.assigner = assigner

    @staticmethod
    def _flatten_multilevel(tensor_list: list) -> torch.Tensor:
        """Flatten a list of (N, C, H, W) tensors → (N*sum(H*W), C)."""
        parts = []
        for t in tensor_list:
            N, C, H, W = t.shape
            parts.append(t.permute(0, 2, 3, 1).reshape(-1, C))
        return torch.cat(parts, dim=0)

    @staticmethod
    def _filter_valid_labels(labels: torch.Tensor) -> torch.Tensor:
        """Remove -1 padding; labels: (N, max_boxes) → (num_valid,)."""
        return labels.reshape(-1)[labels.reshape(-1) >= 0]

    @staticmethod
    def _filter_valid_boxes(labels: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """Remove -1 padding; boxes: (N, max_boxes, 4) → (num_valid, 4)."""
        mask = labels.reshape(-1) >= 0
        return boxes.reshape(-1, 4)[mask]

    def _unpack_inputs(self, loss_name: str, inputs):
        """Extract and flatten the correct tensor from a detection head dict."""
        if not isinstance(inputs, dict):
            return inputs
        key = self._INPUT_KEY.get(loss_name)
        return self._flatten_multilevel(inputs[key])

    def _unpack_target(self, loss_name: str, target):
        """Extract the correct sub-tensor from a detection target dict."""
        if not isinstance(target, dict):
            return target
        t_key = self._TARGET_KEY.get(loss_name)
        if t_key == 'labels':
            return self._filter_valid_labels(target['labels'])
        if t_key == 'boxes':
            return self._filter_valid_boxes(target['labels'], target['boxes'])
        return target

    def compute_loss_value(self, probs, logits, target):
        """
        Compute the weighted sum of loss values.

        Args:
            probs: Model probabilities (or dict for detection heads)
            logits: Model logits (or dict for detection heads)
            target: Ground truth targets (tensor or dict with 'labels'/'boxes')

        Returns:
            Total loss value
        """
        self.loss_values = {}
        total_loss = 0.0

        # Detection path: use assigner to produce list of (pred, tgt) pairs
        if self.assigner is not None and isinstance(logits, dict):
            matched_pairs = self.assigner.assign_batch(logits, target)

            # Guard: all pairs empty → return zero loss with grad
            all_empty = all(pair[0].shape[0] == 0 for pair in matched_pairs)
            if all_empty:
                ref = (logits.get('cls_scores') or logits.get('rpn_cls_flat')
                       or next(iter(logits.values())))
                anchor = ref[0] if isinstance(ref, (list, tuple)) else ref
                return anchor.sum() * 0.0

            # num_pos: number of positive (bbox) samples — used to normalise
            # FocalLoss so that each positive contributes the same gradient
            # regardless of how many negatives are sampled (RTMDet-style).
            num_pos = max(1, matched_pairs[-1][0].shape[0])

            for i, (loss_name, (inp, tgt)) in enumerate(zip(self.losses, matched_pairs)):
                if inp.shape[0] == 0:
                    continue
                loss_fcn = self.loss_fcns[loss_name]
                loss_value = loss_fcn(inp, tgt)

                # For binary-sigmoid FocalLoss over balanced (pos+neg) samples:
                # convert mean → sum/num_pos so every positive has equal weight.
                if (loss_name in self.LOGIT_LOSSES
                        and tgt.dim() == 2
                        and tgt.dtype == torch.float32):
                    loss_value = loss_value * inp.shape[0] / num_pos

                if self.weights is not None:
                    loss_value = loss_value * self.weights[i]
                self.loss_values[loss_name] = loss_value
                total_loss += loss_value
            return total_loss

        # Classification / no-assigner fallback path
        for i, loss_name in enumerate(self.losses):
            loss_fcn = self.loss_fcns[loss_name]
            raw_inputs = logits if loss_name in self.LOGIT_LOSSES else probs
            inputs = self._unpack_inputs(loss_name, raw_inputs)
            target_t = self._unpack_target(loss_name, target)
            loss_value = loss_fcn(inputs, target_t)

            # Apply loss weight if specified
            if self.weights is not None:
                loss_value = loss_value * self.weights[i]

            self.loss_values[loss_name] = loss_value
            total_loss += loss_value

        return total_loss
