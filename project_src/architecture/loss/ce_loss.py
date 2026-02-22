from typing import Union

import torch
from torch.nn import CrossEntropyLoss

from .__template__ import LossFuncTemplate


class CELoss(LossFuncTemplate):
    """
    Cross Entropy Loss implementation.

    This class implements the Cross Entropy Loss function using PyTorch's built-in
    loss functions. It supports both multi-class classification (using CrossEntropyLoss)
    and binary classification (using BCEWithLogitsLoss or BCELoss).
    """

    def __init__(self, args, **kwargs):
        """
        Initialize the Cross Entropy Loss module.

        Args:
            args: Arguments for the loss function, including 'binary' to specify
                  if the task is binary classification.
            **kwargs: Additional arguments for the loss function
        """
        super().__init__(args, **kwargs)
        weight = args.get('weight', None)
        self.loss_func = CrossEntropyLoss(weight=weight, **kwargs)

    def forward(
            self, predictions: Union[torch.Tensor, dict],
            targets: Union[torch.Tensor, dict],
            **kwargs
    ) -> torch.Tensor:
        """
        Compute the Cross Entropy Loss value.

        Args:
            predictions: Model predictions (output from the model)
            targets: Ground truth labels/targets
            **kwargs: Additional arguments specific to the loss function

        Returns:
            torch.Tensor: Computed loss value (scalar tensor)
        """
        if isinstance(predictions, dict):
            predictions = predictions['classification']
            targets = targets['classification']

        return self.loss_func(predictions, targets)
