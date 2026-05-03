import torch
from torchvision.ops import batched_nms


class NMSWrapper:
    """
    Post-process RTMDet head outputs into final detections.

    Decodes per-level ltrb predictions into x1y1x2y2 boxes, applies a score
    threshold, then runs batched NMS. All tensor ops stay on the model's device
    (GPU when available).

    Args:
        score_threshold: Minimum per-class sigmoid score to keep a candidate.
        iou_threshold:   IoU threshold for NMS suppression.
        strides:         Feature-map strides matching the head's output levels.
        max_detections:  Hard cap on returned detections per image.
    """

    def __init__(
        self,
        score_threshold: float = 0.05,
        iou_threshold: float = 0.5,
        strides: list = None,
        max_detections: int = 300,
    ):
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.strides = strides or [8, 16, 32]
        self.max_detections = max_detections

    def _decode_level(self, cls_score: torch.Tensor, bbox_pred: torch.Tensor, stride: int):
        """Decode one feature-map level into image-space boxes and scores.

        Args:
            cls_score: (N, C, H, W)
            bbox_pred: (N, 4, H, W)  ltrb offsets in feature-map units
            stride:    int
        Returns:
            boxes:  (N, H*W, 4)  x1y1x2y2 in image pixels
            scores: (N, H*W, C)  sigmoid class probabilities
        """
        N, C, H, W = cls_score.shape
        device = cls_score.device

        ys = (torch.arange(H, device=device) + 0.5) * stride
        xs = (torch.arange(W, device=device) + 0.5) * stride
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # (H, W)
        centers = torch.stack([grid_x, grid_y], dim=-1).view(1, H * W, 2)  # (1, HW, 2)

        ltrb = bbox_pred.permute(0, 2, 3, 1).reshape(N, H * W, 4) * stride
        x1y1 = centers - ltrb[..., :2]   # cx - l, cy - t
        x2y2 = centers + ltrb[..., 2:]   # cx + r, cy + b
        boxes = torch.cat([x1y1, x2y2], dim=-1)  # (N, HW, 4)

        scores = cls_score.permute(0, 2, 3, 1).reshape(N, H * W, C).sigmoid()

        return boxes, scores

    def __call__(self, head_output: dict) -> list:
        """Apply decode + NMS to a batch of RTMDet head outputs.

        Args:
            head_output: dict from RTMDetHead with keys:
                'cls_scores': list of (N, C, Hi, Wi) tensors per level
                'bbox_preds': list of (N, 4, Hi, Wi) tensors per level
        Returns:
            List of dicts, one per image in the batch:
                'boxes':   (K, 4) float tensor  x1y1x2y2
                'scores':  (K,)   float tensor  confidence
                'classes': (K,)   long tensor   class indices
        """
        cls_scores = head_output['cls_scores']
        bbox_preds = head_output['bbox_preds']
        N = cls_scores[0].shape[0]

        level_boxes, level_scores = [], []
        for cls_s, bbox_p, stride in zip(cls_scores, bbox_preds, self.strides):
            boxes, scores = self._decode_level(cls_s, bbox_p, stride)
            level_boxes.append(boxes)
            level_scores.append(scores)

        all_boxes = torch.cat(level_boxes, dim=1)    # (N, total_preds, 4)
        all_scores = torch.cat(level_scores, dim=1)  # (N, total_preds, C)

        max_scores, class_ids = all_scores.max(dim=-1)  # (N, total_preds)

        results = []
        for i in range(N):
            keep = max_scores[i] > self.score_threshold
            boxes_i = all_boxes[i][keep]
            scores_i = max_scores[i][keep]
            classes_i = class_ids[i][keep]

            if boxes_i.numel() == 0:
                results.append({'boxes': boxes_i, 'scores': scores_i, 'classes': classes_i})
                continue

            # torchvision batched_nms runs on GPU natively when tensors are on CUDA
            nms_keep = batched_nms(boxes_i, scores_i, classes_i, self.iou_threshold)
            nms_keep = nms_keep[:self.max_detections]

            results.append({
                'boxes': boxes_i[nms_keep],
                'scores': scores_i[nms_keep],
                'classes': classes_i[nms_keep],
            })

        return results
