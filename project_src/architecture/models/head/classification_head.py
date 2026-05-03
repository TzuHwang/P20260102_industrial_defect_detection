import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Classification head for image classification tasks."""

    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 dropout_rate: float = 0.0):
        super(ClassificationHead, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Dropout layer (optional)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)

        # Classification layer
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        """Forward pass through the classification head."""
        # Handle both spatial (4D) and non-spatial (3D/2D) inputs
        if x.dim() == 4:
            # Global average pooling for spatial features (N, C, H, W)
            x = self.global_pool(x)
            x = x.view(x.size(0), -1)  # Flatten to (N, C)
        elif x.dim() == 3:
            # For sequence features (N, C, L) or (N, L, C), pool over the sequence dimension
            x = x.mean(dim=-1)  # (N, C)
        # If x.dim() == 2, it's already (N, C), no processing needed

        # Apply dropout if enabled
        if hasattr(self, 'dropout'):
            x = self.dropout(x)

        # Classification
        x = self.classifier(x)

        return x


class LinearClassifier(ClassificationHead):
    """Linear classifier head without dropout."""

    def __init__(self, args):
        super().__init__(
            in_channels=args.in_channels,
            num_classes=args.num_classes,
            dropout_rate=args.dropout_rate
        )
