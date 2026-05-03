import torch.nn as nn

from ..backbone.cspnext import ConvBnSiLU


class SepBNConv(nn.Module):
    """Conv with shared weights and separate BatchNorm per scale level.

    Implements the Separate-BN structure from RTMDet: a single Conv2d kernel
    is applied across all scale levels, but each level has its own BN
    parameters to handle differing feature statistics.
    """

    def __init__(self, in_channels, out_channels, num_levels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False
        )
        self.bn_list = nn.ModuleList([nn.BatchNorm2d(out_channels) for _ in range(num_levels)])
        self.act = nn.SiLU(inplace=True)

    def forward(self, x, level_idx: int):
        return self.act(self.bn_list[level_idx](self.conv(x)))


class RTMDetHead(nn.Module):
    """RTMDet detection head with shared convs and per-scale Separate-BN.

    Applies stacked_convs shared conv layers with separate BN for each scale,
    then produces per-level classification scores and bounding-box predictions.
    """

    def __init__(self,
                 in_channels: int = 128,
                 feat_channels: int = 128,
                 num_classes: int = 80,
                 num_levels: int = 3,
                 stacked_convs: int = 2):
        super().__init__()

        self.num_levels = num_levels

        # Optional input projection when channels differ across the two branches
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(stacked_convs):
            c_in = in_channels if i == 0 else feat_channels
            self.cls_convs.append(SepBNConv(c_in, feat_channels, num_levels))
            self.reg_convs.append(SepBNConv(c_in, feat_channels, num_levels))

        # Final 1x1 prediction layers (shared across scales)
        self.cls_pred = nn.Conv2d(feat_channels, num_classes, 1)
        self.reg_pred = nn.Conv2d(feat_channels, 4, 1)

    def forward(self, features):
        """Forward pass.

        Args:
            features: (fp3, fp4, fp5) tuple from RTMDetPAFPN

        Returns:
            dict with keys:
              'cls_scores': list of (N, num_classes, Hi, Wi) tensors per level
              'bbox_preds': list of (N, 4, Hi, Wi) tensors per level (ltrb offsets)
        """
        cls_scores, bbox_preds = [], []

        for level_idx, x in enumerate(features):
            cls_feat = x
            reg_feat = x

            for conv in self.cls_convs:
                cls_feat = conv(cls_feat, level_idx)
            for conv in self.reg_convs:
                reg_feat = conv(reg_feat, level_idx)

            cls_scores.append(self.cls_pred(cls_feat))
            bbox_preds.append(self.reg_pred(reg_feat))

        return {'cls_scores': cls_scores, 'bbox_preds': bbox_preds}


class RTMDetSepBNHead(RTMDetHead):
    def __init__(self, args):
        super().__init__(
            in_channels=args.in_channels,
            feat_channels=args.feat_channels,
            num_classes=args.num_classes,
            num_levels=args.num_levels,
            stacked_convs=args.stacked_convs,
        )
