import torch
import torch.nn as nn


class ConvBnSiLU(nn.Module):
    """Conv2d + BatchNorm2d + SiLU building block."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, groups=1):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class CSPBottleneck(nn.Module):
    """Bottleneck block with two 3x3 convs and optional residual shortcut."""

    def __init__(self, channels, shortcut=True):
        super().__init__()
        mid = channels // 2
        self.conv1 = ConvBnSiLU(channels, mid, 3)
        self.conv2 = ConvBnSiLU(mid, channels, 3)
        self.shortcut = shortcut

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        return x + out if self.shortcut else out


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        N, C, _, _ = x.shape
        attn = self.fc(self.gap(x).view(N, C)).view(N, C, 1, 1)
        return x * attn


class CSPLayer(nn.Module):
    """Cross Stage Partial layer with N bottleneck blocks and optional channel attention."""

    def __init__(self, in_channels, out_channels, num_blocks=1, shortcut=True, use_attn=False):
        super().__init__()
        mid = out_channels // 2
        self.conv1 = ConvBnSiLU(in_channels, mid, 1)
        self.conv2 = ConvBnSiLU(in_channels, mid, 1)
        self.blocks = nn.Sequential(*[
            CSPBottleneck(mid, shortcut=shortcut) for _ in range(num_blocks)
        ])
        self.conv3 = ConvBnSiLU(mid * 2, out_channels, 1)
        self.attn = ChannelAttention(out_channels) if use_attn else nn.Identity()

    def forward(self, x):
        x1 = self.blocks(self.conv1(x))
        x2 = self.conv2(x)
        return self.attn(self.conv3(torch.cat([x1, x2], dim=1)))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling Fast via cascaded 5x5 max-pool operations."""

    def __init__(self, in_channels, out_channels, pool_size=5):
        super().__init__()
        mid = in_channels // 2
        self.conv1 = ConvBnSiLU(in_channels, mid, 1)
        self.pool = nn.MaxPool2d(pool_size, stride=1, padding=pool_size // 2)
        self.conv2 = ConvBnSiLU(mid * 4, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        p1 = self.pool(x)
        p2 = self.pool(p1)
        p3 = self.pool(p2)
        return self.conv2(torch.cat([x, p1, p2, p3], dim=1))


class CSPNeXtBackbone(nn.Module):
    """CSPNeXt backbone for RTMDet.

    Outputs three feature maps at strides 8, 16, and 32 (P3, P4, P5).
    """

    def __init__(self,
                 base_channels: int = 64,
                 num_blocks: tuple = (3, 6, 6, 3),
                 use_attn: bool = True):
        super().__init__()

        C = base_channels
        # Stem: 3 convs with total stride 2 (1/2 resolution)
        self.stem = nn.Sequential(
            ConvBnSiLU(3, C // 2, 3, stride=2),
            ConvBnSiLU(C // 2, C // 2, 3),
            ConvBnSiLU(C // 2, C, 3),
        )
        self.stage1 = nn.Sequential(             # 1/4
            ConvBnSiLU(C, C * 2, 3, stride=2),
            CSPLayer(C * 2, C * 2, num_blocks=num_blocks[0]),
        )
        self.stage2 = nn.Sequential(             # 1/8 → P3
            ConvBnSiLU(C * 2, C * 4, 3, stride=2),
            CSPLayer(C * 4, C * 4, num_blocks=num_blocks[1]),
        )
        self.stage3 = nn.Sequential(             # 1/16 → P4
            ConvBnSiLU(C * 4, C * 8, 3, stride=2),
            CSPLayer(C * 8, C * 8, num_blocks=num_blocks[2]),
        )
        self.stage4 = nn.Sequential(             # 1/32 → P5
            ConvBnSiLU(C * 8, C * 8, 3, stride=2),
            SPPF(C * 8, C * 8),
            CSPLayer(C * 8, C * 8, num_blocks=num_blocks[3], use_attn=use_attn),
        )
        self.out_channels = (C * 4, C * 8, C * 8)

    def forward(self, x):
        """Forward pass, returns (P3, P4, P5) feature maps."""
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        return p3, p4, p5


class CSPNeXtTiny(CSPNeXtBackbone):
    """CSPNeXt-Tiny: base_channels=32, num_blocks=(1, 2, 2, 1)."""

    def __init__(self, args):
        super().__init__(
            base_channels=args.base_channels,
            num_blocks=tuple(args.num_blocks),
            use_attn=args.use_attn,
        )


class CSPNeXtSmall(CSPNeXtBackbone):
    """CSPNeXt-Small: base_channels=64, num_blocks=(3, 6, 6, 3)."""

    def __init__(self, args):
        super().__init__(
            base_channels=args.base_channels,
            num_blocks=tuple(args.num_blocks),
            use_attn=args.use_attn,
        )
