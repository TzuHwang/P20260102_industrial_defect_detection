import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align, nms, clip_boxes_to_image


# ---------------------------------------------------------------------------
# Box utilities
# ---------------------------------------------------------------------------

def _encode_boxes(anchors: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    """Encode GT boxes as deltas relative to anchors.

    Both tensors in xyxy format (N, 4). Returns (N, 4) deltas (dx, dy, dw, dh).
    """
    aw = anchors[:, 2] - anchors[:, 0]
    ah = anchors[:, 3] - anchors[:, 1]
    acx = anchors[:, 0] + aw * 0.5
    acy = anchors[:, 1] + ah * 0.5

    gw = gt_boxes[:, 2] - gt_boxes[:, 0]
    gh = gt_boxes[:, 3] - gt_boxes[:, 1]
    gcx = gt_boxes[:, 0] + gw * 0.5
    gcy = gt_boxes[:, 1] + gh * 0.5

    dx = (gcx - acx) / aw
    dy = (gcy - acy) / ah
    dw = torch.log(gw / aw.clamp(min=1e-6))
    dh = torch.log(gh / ah.clamp(min=1e-6))
    return torch.stack([dx, dy, dw, dh], dim=1)


def _decode_boxes(anchors: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """Decode predicted deltas back to xyxy boxes."""
    aw = anchors[:, 2] - anchors[:, 0]
    ah = anchors[:, 3] - anchors[:, 1]
    acx = anchors[:, 0] + aw * 0.5
    acy = anchors[:, 1] + ah * 0.5

    dx, dy, dw, dh = deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3]

    # Clamp dw/dh to avoid inf
    dw = dw.clamp(max=math.log(1000.0))
    dh = dh.clamp(max=math.log(1000.0))

    px = acx + dx * aw
    py = acy + dy * ah
    pw = aw * dw.exp()
    ph = ah * dh.exp()

    return torch.stack([px - pw * 0.5, py - ph * 0.5,
                        px + pw * 0.5, py + ph * 0.5], dim=1)


def _pairwise_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Compute (M, N) pairwise IoU between M and N xyxy boxes."""
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    ix1 = torch.max(boxes_a[:, None, 0], boxes_b[None, :, 0])
    iy1 = torch.max(boxes_a[:, None, 1], boxes_b[None, :, 1])
    ix2 = torch.min(boxes_a[:, None, 2], boxes_b[None, :, 2])
    iy2 = torch.min(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-6)


# ---------------------------------------------------------------------------
# Anchor generator
# ---------------------------------------------------------------------------

class AnchorGenerator:
    """Generate anchor boxes for each FPN level.

    Args:
        sizes:  Base anchor size (area = size²) per FPN level.
        ratios: Height/width aspect ratios shared across levels.
    """

    def __init__(self, sizes=(32, 64, 128), ratios=(0.5, 1.0, 2.0)):
        self.sizes = sizes
        self.ratios = torch.tensor(ratios, dtype=torch.float32)

    def num_anchors(self):
        return len(self.ratios)

    def _base_anchors(self, size: int, device: torch.device) -> torch.Tensor:
        """(num_ratios, 4) centre-zero anchors for one level."""
        ratios = self.ratios.to(device)
        h = (size * ratios.sqrt())
        w = (size / ratios.sqrt())
        return torch.stack([-w / 2, -h / 2, w / 2, h / 2], dim=1)

    def grid_anchors(self, feat_h: int, feat_w: int, stride: int,
                     size: int, device: torch.device) -> torch.Tensor:
        """(feat_h * feat_w * num_ratios, 4) anchors in original image space."""
        base = self._base_anchors(size, device)              # (R, 4)

        ys = (torch.arange(feat_h, device=device, dtype=torch.float32) + 0.5) * stride
        xs = (torch.arange(feat_w, device=device, dtype=torch.float32) + 0.5) * stride
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        shifts = torch.stack([gx.flatten(), gy.flatten(),
                               gx.flatten(), gy.flatten()], dim=1)  # (H*W, 4)

        anchors = (base[None] + shifts[:, None]).reshape(-1, 4)  # (H*W*R, 4)
        return anchors


# ---------------------------------------------------------------------------
# RPN (one shared conv for all levels)
# ---------------------------------------------------------------------------

class RPN(nn.Module):
    def __init__(self, in_channels: int, num_anchors: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.cls_head = nn.Conv2d(in_channels, num_anchors, 1)
        self.reg_head = nn.Conv2d(in_channels, num_anchors * 4, 1)

        for layer in [self.conv, self.cls_head, self.reg_head]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.zeros_(layer.bias)

    def forward(self, feat: torch.Tensor):
        x = F.relu(self.conv(feat), inplace=True)
        return self.cls_head(x), self.reg_head(x)


# ---------------------------------------------------------------------------
# RoI Head
# ---------------------------------------------------------------------------

class RoIHead(nn.Module):
    """Two-layer FC RoI classifier and box regressor.

    Args:
        in_channels:   FPN out_channels.
        roi_pool_size: Spatial size after RoI Align (7×7 typical).
        fc_dim:        Hidden dimension for both FC layers.
        num_classes:   Number of object classes (background excluded).
    """

    def __init__(self, in_channels: int, roi_pool_size: int, fc_dim: int, num_classes: int):
        super().__init__()
        flat = in_channels * roi_pool_size * roi_pool_size
        self.fc1 = nn.Linear(flat, fc_dim)
        self.fc2 = nn.Linear(fc_dim, fc_dim)
        self.cls_head = nn.Linear(fc_dim, num_classes + 1)   # +1 for background
        self.reg_head = nn.Linear(fc_dim, 4)                 # class-agnostic regression

        for layer in [self.fc1, self.fc2]:
            nn.init.kaiming_uniform_(layer.weight, a=1)
            nn.init.zeros_(layer.bias)
        nn.init.normal_(self.cls_head.weight, std=0.01)
        nn.init.zeros_(self.cls_head.bias)
        nn.init.normal_(self.reg_head.weight, std=0.001)
        nn.init.zeros_(self.reg_head.bias)

    def forward(self, roi_feats: torch.Tensor):
        x = roi_feats.flatten(start_dim=1)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        return self.cls_head(x), self.reg_head(x)


# ---------------------------------------------------------------------------
# FasterRCNNHead
# ---------------------------------------------------------------------------

class FasterRCNNHead(nn.Module):
    """Faster R-CNN two-stage detection head.

    Runs RPN on FPN features to generate proposals, then RoI-aligns and
    classifies/regresses each proposal with the RoI head.

    During training (assigner-driven), targets are NOT passed here;
    the FasterRCNNAssigner handles GT matching externally.

    Forward output dict:
        anchors_flat:    (total_anchors, 4)  – all anchors concatenated
        rpn_cls_flat:    (total_anchors,)    – RPN objectness logits (N*A per batch)
        rpn_reg_flat:    (total_anchors, 4)  – RPN box deltas
        proposals_cat:   (total_proposals, 4) – post-NMS proposals (xyxy)
        roi_img_ids:     (total_proposals,)   – image index per proposal
        roi_cls_scores:  (total_proposals, C+1)
        roi_bbox_preds:  (total_proposals, 4)

    Args:
        in_channels:    FPN out_channels.
        num_classes:    Object classes (background excluded).
        strides:        FPN level strides, e.g. [8, 16, 32].
        anchor_sizes:   Base anchor size per FPN level.
        anchor_ratios:  Aspect ratios shared across levels.
        roi_pool_size:  Spatial size for RoI Align.
        fc_dim:         RoI head hidden dim.
        rpn_pre_nms_topk:   Pre-NMS proposals kept per level.
        rpn_post_nms_topk:  Max proposals after cross-level NMS.
        rpn_nms_thresh:     RPN NMS IoU threshold.
        rpn_score_thresh:   Minimum objectness score for proposals.
    """

    def __init__(self, args):
        super().__init__()
        in_channels = args.in_channels
        num_classes = args.num_classes
        strides = list(args.strides)
        anchor_sizes = list(args.anchor_sizes)
        anchor_ratios = list(getattr(args, 'anchor_ratios', [0.5, 1.0, 2.0]))

        self.strides = strides
        self.anchor_gen = AnchorGenerator(sizes=anchor_sizes, ratios=anchor_ratios)
        num_anchors = self.anchor_gen.num_anchors()

        self.rpn = RPN(in_channels, num_anchors)
        self.roi_head = RoIHead(
            in_channels=in_channels,
            roi_pool_size=getattr(args, 'roi_pool_size', 7),
            fc_dim=getattr(args, 'fc_dim', 1024),
            num_classes=num_classes,
        )

        self.rpn_pre_nms_topk = getattr(args, 'rpn_pre_nms_topk', 2000)
        self.rpn_post_nms_topk = getattr(args, 'rpn_post_nms_topk', 300)
        self.rpn_nms_thresh = getattr(args, 'rpn_nms_thresh', 0.7)
        self.rpn_score_thresh = getattr(args, 'rpn_score_thresh', 0.0)
        self.roi_pool_size = getattr(args, 'roi_pool_size', 7)
        self.num_levels = len(strides)
        self.anchor_sizes = anchor_sizes

    # ------------------------------------------------------------------
    def forward(self, features):
        """
        Args:
            features: tuple of (P3, P4, P5) from FPN.

        Returns:
            dict (see class docstring).
        """
        N = features[0].shape[0]
        device = features[0].device

        # ── RPN ──────────────────────────────────────────────────────
        all_anchors, all_cls_logits, all_reg_deltas = [], [], []
        for lvl, feat in enumerate(features):
            _, _, H, W = feat.shape
            anchors = self.anchor_gen.grid_anchors(
                H, W, self.strides[lvl], self.anchor_sizes[lvl], device)
            all_anchors.append(anchors)                  # (A_l, 4)

            rpn_cls, rpn_reg = self.rpn(feat)            # (N, A, H, W)
            A = self.anchor_gen.num_anchors()
            # (N, A, H, W) → (N, H*W*A)
            all_cls_logits.append(rpn_cls.permute(0, 2, 3, 1).reshape(N, -1))
            # (N, A*4, H, W) → (N, H*W*A, 4)
            all_reg_deltas.append(rpn_reg.permute(0, 2, 3, 1).reshape(N, -1, 4))

        anchors_cat = torch.cat(all_anchors, dim=0)              # (total_A, 4)
        cls_flat = torch.cat(all_cls_logits, dim=1)              # (N, total_A)
        reg_flat = torch.cat(all_reg_deltas, dim=1)              # (N, total_A, 4)

        # ── Generate proposals ────────────────────────────────────────
        with torch.no_grad():
            proposals_per_img = self._generate_proposals(
                N, anchors_cat, cls_flat, reg_flat, features[0])

        # ── RoI Align ─────────────────────────────────────────────────
        roi_feats, roi_img_ids = self._roi_align(features, proposals_per_img, device)
        proposals_cat = torch.cat(proposals_per_img, dim=0)      # (total_P, 4)

        # ── RoI Head ──────────────────────────────────────────────────
        roi_cls_scores, roi_bbox_preds = self.roi_head(roi_feats)

        return {
            # RPN outputs (needed by assigner for RPN loss)
            'anchors_flat':   anchors_cat,       # (total_A, 4)
            'rpn_cls_flat':   cls_flat,           # (N, total_A) — batch dim kept for per-img assignment
            'rpn_reg_flat':   reg_flat,           # (N, total_A, 4)
            # RoI outputs (needed by assigner for RoI loss)
            'proposals_cat':  proposals_cat,      # (total_P, 4)
            'roi_img_ids':    roi_img_ids,         # (total_P,)
            'roi_cls_scores': roi_cls_scores,      # (total_P, C+1)
            'roi_bbox_preds': roi_bbox_preds,      # (total_P, 4)
        }

    # ------------------------------------------------------------------
    def _generate_proposals(self, N, anchors, cls_flat, reg_flat, ref_feat):
        """Decode anchors, NMS → per-image proposal list."""
        img_h = ref_feat.shape[2] * self.strides[0]
        img_w = ref_feat.shape[3] * self.strides[0]
        proposals = []

        for i in range(N):
            scores = cls_flat[i].sigmoid()                        # (total_A,)
            deltas = reg_flat[i]                                  # (total_A, 4)

            boxes = _decode_boxes(anchors, deltas)
            boxes = clip_boxes_to_image(boxes, (img_h, img_w))

            # Pre-NMS top-k
            k = min(self.rpn_pre_nms_topk, scores.shape[0])
            _, topk_idx = scores.topk(k)
            boxes_k = boxes[topk_idx]
            scores_k = scores[topk_idx]

            # Remove tiny boxes (area < 1)
            wh = (boxes_k[:, 2:] - boxes_k[:, :2]).clamp(min=0)
            valid = (wh[:, 0] * wh[:, 1]) > 1
            boxes_k = boxes_k[valid]
            scores_k = scores_k[valid]

            # NMS
            keep = nms(boxes_k, scores_k, self.rpn_nms_thresh)
            keep = keep[:self.rpn_post_nms_topk]
            proposals.append(boxes_k[keep])                      # (n_i, 4)

        return proposals

    # ------------------------------------------------------------------
    def _assign_level(self, boxes: torch.Tensor) -> torch.Tensor:
        """Assign each box to an FPN level index [0, num_levels-1]."""
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        lvls = torch.floor(torch.log2((areas.clamp(min=1e-6).sqrt() / 56.0))).long()
        return lvls.clamp(0, self.num_levels - 1)

    # ------------------------------------------------------------------
    def _roi_align(self, features, proposals_per_img, device):
        """RoI Align across FPN levels, returning concatenated features."""
        # Build (total_P, 5) boxes with batch index
        total_rois = sum(p.shape[0] for p in proposals_per_img)
        if total_rois == 0:
            C = features[0].shape[1]
            S = self.roi_pool_size
            return (features[0].new_zeros(0, C, S, S),
                    torch.zeros(0, dtype=torch.long, device=device))

        img_ids = torch.cat([
            torch.full((p.shape[0],), i, dtype=torch.long, device=device)
            for i, p in enumerate(proposals_per_img)
        ])
        boxes_cat = torch.cat(proposals_per_img, dim=0)          # (total_P, 4)

        lvl_assign = self._assign_level(boxes_cat)               # (total_P,)

        output = boxes_cat.new_zeros(total_rois, features[0].shape[1],
                                     self.roi_pool_size, self.roi_pool_size)

        for lvl, feat in enumerate(features):
            mask = (lvl_assign == lvl)
            if not mask.any():
                continue
            spatial_scale = 1.0 / self.strides[lvl]
            rois = torch.cat([img_ids[mask, None].float(), boxes_cat[mask]], dim=1)
            output[mask] = roi_align(feat, rois, self.roi_pool_size,
                                     spatial_scale=spatial_scale, aligned=True)

        return output, img_ids
