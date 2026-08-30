import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    """Feature Pyramid Network.

    Takes (C3, C4, C5) from backbone and produces (P3, P4, P5) with uniform
    channel depth via lateral projections and top-down upsampling.

    Args:
        in_channels:  (C3_ch, C4_ch, C5_ch) from backbone.
        out_channels: Uniform output channels for all levels.
    """

    def __init__(self, in_channels=(512, 1024, 2048), out_channels=256):
        super().__init__()
        C3, C4, C5 = in_channels
        out_ch = out_channels

        self.lat_p5 = nn.Conv2d(C5, out_ch, 1)
        self.lat_p4 = nn.Conv2d(C4, out_ch, 1)
        self.lat_p3 = nn.Conv2d(C3, out_ch, 1)

        self.out_p5 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.out_p4 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.out_p3 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.out_channels = (out_ch, out_ch, out_ch)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features):
        """Args:
            features: (C3, C4, C5) tuple from ResNet50FPN.

        Returns:
            (P3, P4, P5) with uniform out_channels at strides 8, 16, 32.
        """
        c3, c4, c5 = features

        p5 = self.lat_p5(c5)
        p4 = self.lat_p4(c4) + F.interpolate(p5, scale_factor=2, mode='nearest')
        p3 = self.lat_p3(c3) + F.interpolate(p4, scale_factor=2, mode='nearest')

        return self.out_p3(p3), self.out_p4(p4), self.out_p5(p5)


class ResNet50FPNNeck(FPN):
    def __init__(self, args):
        super().__init__(
            in_channels=tuple(args.in_channels),
            out_channels=args.out_channels,
        )
