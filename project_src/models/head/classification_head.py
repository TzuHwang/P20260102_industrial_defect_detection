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
        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten

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


class DropoutClassifier(ClassificationHead):
    """Classifier head with dropout for regularization."""

    def __init__(self, args):
        super().__init__(
            in_channels=args.in_channels,
            num_classes=args.num_classes,
            dropout_rate=args.dropout_rate
        )
