import torch.nn as nn

from .focal_loss import FocalLoss


class LossFactory:
    # Loss functions with built-in activation — expect raw logits
    LOGIT_LOSSES = {'CrossEntropyLoss', 'BCEWithLogitsLoss', 'FocalLoss'}

    def __init__(self, args):
        self.losses = args.losses
        self.channel_weights = args.channel_weights
        self.weights = args.loss_weights
        self.loss_fcns = {}
        self.loss_values = {}
        self.in_channel_loss_values = {}

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
        }

        for loss_name in self.losses:
            if loss_name in loss_mapping:
                self.loss_fcns[loss_name] = loss_mapping[loss_name]()
            else:
                raise ValueError(f"Unknown loss function: {loss_name}")

    def get_loss_fcns(self):
        return self.loss_fcns

    def compute_loss_value(self, probs, logits, target):
        """
        Compute the weighted sum of loss values.

        Args:
            probs: Model probabilities
            logits: Model logits
            target: Ground truth targets

        Returns:
            Total loss value
        """
        self.loss_values = {}
        total_loss = 0.0

        for i, loss_name in enumerate(self.losses):
            loss_fcn = self.loss_fcns[loss_name]
            inputs = logits if loss_name in self.LOGIT_LOSSES else probs
            loss_value = loss_fcn(inputs, target)

            # Apply loss weight if specified
            if self.weights is not None:
                loss_value = loss_value * self.weights[i]

            self.loss_values[loss_name] = loss_value
            total_loss += loss_value

        return total_loss
