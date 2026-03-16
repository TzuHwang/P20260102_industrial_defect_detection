import torch
import torch.nn as nn


class LossFuncTemplate(nn.Module):
    """
    Base class that combines PyTorch's nn.Module with our AbstractLoss interface.

    This allows for compatibility with PyTorch's module system while maintaining
    the abstract interface for consistent loss implementations.
    """

    def __init__(self, args, **kwargs):
        """
        Initialize the base loss module.

        Args:
            args: Arguments for the loss function
            **kwargs: Additional arguments for the loss function
        """
        super().__init__()
        self.args = args
        self.loss_func = None  # Placeholder for the actual loss function implementation

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute the loss value.

        Args:
            predictions: Model predictions (output from the model)
            targets: Ground truth labels/targets
            **kwargs: Additional arguments specific to the loss function

        Returns:
            torch.Tensor: Computed loss value (scalar tensor)
        """
        if self.loss_func is None:
            raise NotImplementedError("Loss function not implemented.")

        return self.loss_func(predictions, targets, **kwargs)

    def get_name(self) -> str:
        """
        Get the name of the loss function.

        Returns:
            str: Name of the loss function
        """
        return self.__class__.__name__
