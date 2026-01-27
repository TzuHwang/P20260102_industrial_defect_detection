import torch.nn as nn

from . import backbone, head, neck


class ModelFactory(nn.Module):
    """Factory class to create different types of models."""

    def __init__(self, args):
        super(ModelFactory, self).__init__()
        self.args = args
        self.backbone = backbone.__dict__.get(args.backbone)(getattr(args, args.backbone))
        self.neck = neck.__dict__.get(args.neck)(getattr(args, args.neck))
        self.head = head.__dict__.get(args.head)(getattr(args, args.head))

    def forward(self, x):
        """Forward pass through the model."""
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x
