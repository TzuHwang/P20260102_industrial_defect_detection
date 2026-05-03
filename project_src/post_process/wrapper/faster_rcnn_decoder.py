import torch
from torchvision.ops import batched_nms

from project_src.architecture.models.head.faster_rcnn_head import _decode_boxes


class FasterRCNNDecoder:
    """Post-process FasterRCNNHead outputs into final per-image detections.

    Decodes RoI box deltas from proposals, applies softmax, filters by score,
    then runs per-class NMS.

    Args:
        score_threshold: Minimum class score to keep a detection.
        nms_threshold:   IoU threshold for NMS suppression.
        max_detections:  Hard cap on returned detections per image.
    """

    def __init__(self, score_threshold=0.3, nms_threshold=0.5, max_detections=100):
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections

    def __call__(self, head_output: dict) -> list:
        """
        Args:
            head_output: dict from FasterRCNNHead.forward with keys:
                'proposals_cat':  (total_P, 4) xyxy proposals
                'roi_img_ids':    (total_P,) image index per RoI
                'roi_cls_scores': (total_P, C+1) class logits (incl. background at 0)
                'roi_bbox_preds': (total_P, 4) box regression deltas

        Returns:
            List of dicts, one per image:
                'boxes':   (K, 4) float tensor  x1y1x2y2
                'scores':  (K,)   float tensor
                'classes': (K,)   long tensor   (0-indexed, background excluded)
        """
        proposals = head_output['proposals_cat']       # (total_P, 4)
        img_ids = head_output['roi_img_ids']           # (total_P,)
        cls_logits = head_output['roi_cls_scores']     # (total_P, C+1)
        reg_deltas = head_output['roi_bbox_preds']     # (total_P, 4)

        N = int(img_ids.max().item()) + 1 if len(img_ids) > 0 else 0
        results = []

        cls_probs = cls_logits.softmax(dim=-1)          # (total_P, C+1)
        decoded = _decode_boxes(proposals, reg_deltas)   # (total_P, 4) xyxy

        for i in range(N):
            mask = img_ids == i
            if not mask.any():
                results.append({
                    'boxes': proposals.new_zeros(0, 4),
                    'scores': proposals.new_zeros(0),
                    'classes': img_ids.new_zeros(0),
                })
                continue

            probs_i = cls_probs[mask]      # (P_i, C+1)
            boxes_i = decoded[mask]        # (P_i, 4)

            # Exclude background class (index 0)
            fg_probs = probs_i[:, 1:]      # (P_i, C)
            scores_i, cls_i = fg_probs.max(dim=1)

            keep = scores_i > self.score_threshold
            if not keep.any():
                results.append({
                    'boxes': boxes_i.new_zeros(0, 4),
                    'scores': boxes_i.new_zeros(0),
                    'classes': cls_i.new_zeros(0),
                })
                continue

            boxes_k = boxes_i[keep]
            scores_k = scores_i[keep]
            cls_k = cls_i[keep]

            nms_keep = batched_nms(boxes_k, scores_k, cls_k, self.nms_threshold)
            nms_keep = nms_keep[:self.max_detections]

            results.append({
                'boxes': boxes_k[nms_keep],
                'scores': scores_k[nms_keep],
                'classes': cls_k[nms_keep],
            })

        return results
