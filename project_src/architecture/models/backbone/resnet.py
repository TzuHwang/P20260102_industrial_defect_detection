import torch.nn as nn
from torchvision.models import (resnet18, resnet34, resnet50, resnet101,
                                resnet152)
from torchvision.models.resnet import (ResNet18_Weights, ResNet34_Weights,
                                       ResNet50_Weights, ResNet101_Weights,
                                       ResNet152_Weights)


class ResNetBackbone(nn.Module):
    """ResNet backbone with optional ImageNet pretrained weights"""

    def __init__(self,
                 depth: int = 50,
                 pretrained: bool = True,
                 frozen_stages: int = -1,
                 norm_eval: bool = False):
        super().__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval

        # Select the appropriate ResNet model and weights
        if depth == 18:
            self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        elif depth == 34:
            self.model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        elif depth == 50:
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        elif depth == 101:
            self.model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1 if pretrained else None)
        elif depth == 152:
            self.model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            raise ValueError(f'Unsupported ResNet depth: {depth}. Supported depths: 18, 34, 50, 101, 152')

        # Remove the final classification layer to use as backbone
        self.feature_dim = self.model.fc.in_features
        self.model.fc = nn.Identity()

        self._freeze_stages()

    def _freeze_stages(self):
        """Freeze stages of the ResNet model"""
        if self.frozen_stages >= 0:
            # Freeze the stem layer
            self.model.conv1.eval()
            self.model.bn1.eval()
            for param in self.model.conv1.parameters():
                param.requires_grad = False
            for param in self.model.bn1.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self.model, f'layer{i}')
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        """Override the train method to handle norm evaluation"""
        super().train(mode)
        if self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    def forward(self, x):
        """Forward pass through the ResNet backbone"""
        return self.model(x)


class ResNet18(ResNetBackbone):
    def __init__(self, args):
        super().__init__(
            depth=18,
            pretrained=args.pretrained,
            frozen_stages=args.frozen_stages,
            norm_eval=args.norm_eval,
        )


class ResNet34(ResNetBackbone):
    def __init__(self, args):
        super().__init__(
            depth=34,
            pretrained=args.pretrained,
            frozen_stages=args.frozen_stages,
            norm_eval=args.norm_eval,
        )


class ResNet50(ResNetBackbone):
    def __init__(self, args):
        super().__init__(
            depth=50,
            pretrained=args.pretrained,
            frozen_stages=args.frozen_stages,
            norm_eval=args.norm_eval,
        )


class ResNet101(ResNetBackbone):
    def __init__(self, args):
        super().__init__(
            depth=101,
            pretrained=args.pretrained,
            frozen_stages=args.frozen_stages,
            norm_eval=args.norm_eval,
        )


class ResNet152(ResNetBackbone):
    def __init__(self, args):
        super().__init__(
            depth=152,
            pretrained=args.pretrained,
            frozen_stages=args.frozen_stages,
            norm_eval=args.norm_eval,
        )
