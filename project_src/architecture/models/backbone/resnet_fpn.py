import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor


class ResNet50FPN(nn.Module):
    """ResNet-50 backbone that returns (C3, C4, C5) feature maps for FPN.

    C3: stride 8,  512 channels  (after layer2)
    C4: stride 16, 1024 channels (after layer3)
    C5: stride 32, 2048 channels (after layer4)

    Args:
        pretrained:     Load ImageNet weights when True.
        frozen_stages:  Number of early stages to freeze (0 = freeze stem+stage1,
                        1 = also freeze stage2, -1 = freeze nothing).
    """

    out_channels = (512, 1024, 2048)

    def __init__(self, args):
        super().__init__()
        pretrained = getattr(args, 'pretrained', True)
        frozen_stages = getattr(args, 'frozen_stages', 0)

        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        base = resnet50(weights=weights)

        self.body = create_feature_extractor(base, {
            'layer2': 'C3',
            'layer3': 'C4',
            'layer4': 'C5',
        })

        freeze_prefixes = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3'][:frozen_stages + 2]
        for name, param in base.named_parameters():
            if any(name.startswith(p) for p in freeze_prefixes):
                param.requires_grad_(False)

    def forward(self, x):
        feats = self.body(x)
        return feats['C3'], feats['C4'], feats['C5']
