import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone.cspnext import ConvBnSiLU, CSPLayer


class CSPNeXtPAFPN(nn.Module):
    """Feature Pyramid Network with Path Aggregation built from CSP layers.

    Runs a top-down FPN pass followed by a bottom-up PAN pass, each using
    CSP layers for feature fusion.  Accepts (P3, P4, P5) from CSPNeXtBackbone
    and returns aligned multi-scale features at the same three resolutions.
    """

    def __init__(self,
                 in_channels: tuple = (256, 512, 512),
                 out_channels: int = 128,
                 num_blocks: int = 3):
        super().__init__()

        C3, C4, C5 = in_channels
        O = out_channels

        # Lateral projections
        self.reduce_p5 = ConvBnSiLU(C5, O, 1)
        self.reduce_p4 = ConvBnSiLU(C4, O, 1)
        self.reduce_p3 = ConvBnSiLU(C3, O, 1)

        # Top-down fusion
        self.csp_td_p4 = CSPLayer(O * 2, O, num_blocks=num_blocks, shortcut=False)
        self.csp_td_p3 = CSPLayer(O * 2, O, num_blocks=num_blocks, shortcut=False)

        # Bottom-up downsampling and fusion
        self.down_p3 = ConvBnSiLU(O, O, 3, stride=2)
        self.csp_bu_p4 = CSPLayer(O * 2, O, num_blocks=num_blocks, shortcut=False)
        self.down_p4 = ConvBnSiLU(O, O, 3, stride=2)
        self.csp_bu_p5 = CSPLayer(O * 2, O, num_blocks=num_blocks, shortcut=False)

        self.out_channels = (O, O, O)

    def forward(self, features):
        """Forward pass.

        Args:
            features: (P3, P4, P5) tuple from CSPNeXtBackbone

        Returns:
            (fp3, fp4, fp5) – fused features at strides 8, 16, 32
        """
        p3, p4, p5 = features

        # Top-down pathway
        p5_td = self.reduce_p5(p5)
        p4_td = self.csp_td_p4(torch.cat([
            self.reduce_p4(p4),
            F.interpolate(p5_td, scale_factor=2, mode='nearest'),
        ], dim=1))
        p3_out = self.csp_td_p3(torch.cat([
            self.reduce_p3(p3),
            F.interpolate(p4_td, scale_factor=2, mode='nearest'),
        ], dim=1))

        # Bottom-up pathway
        p4_out = self.csp_bu_p4(torch.cat([p4_td, self.down_p3(p3_out)], dim=1))
        p5_out = self.csp_bu_p5(torch.cat([p5_td, self.down_p4(p4_out)], dim=1))

        return p3_out, p4_out, p5_out


class RTMDetPAFPN(CSPNeXtPAFPN):
    def __init__(self, args):
        super().__init__(
            in_channels=tuple(args.in_channels),
            out_channels=args.out_channels,
            num_blocks=args.num_blocks,
        )
