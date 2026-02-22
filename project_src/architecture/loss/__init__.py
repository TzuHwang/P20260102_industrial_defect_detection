import torch.nn as nn


class LossFuncs:
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
        }

        for loss_name in self.losses:
            if loss_name in loss_mapping:
                self.loss_fcns[loss_name] = loss_mapping[loss_name]()
            else:
                raise ValueError(f"Unknown loss function: {loss_name}")

    def get_loss_fcns(self):
        return self.loss_fcns

    def get_loss_value(self, pred, target):
        """
        Compute the weighted sum of loss values.

        Args:
            pred: Model predictions
            target: Ground truth targets

        Returns:
            Total loss value
        """
        total_loss = 0.0

        for i, loss_name in enumerate(self.losses):
            loss_fcn = self.loss_fcns[loss_name]
            loss_value = loss_fcn(pred, target)

            # Apply loss weight if specified
            if self.weights is not None:
                loss_value = loss_value * self.weights[i]

            self.loss_values[loss_name] = loss_value
            total_loss += loss_value

        return total_loss
