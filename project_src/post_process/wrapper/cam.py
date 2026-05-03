import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad

cam_methods = {
    'GradCAM': GradCAM,
    'HiResCAM': HiResCAM,
    'ScoreCAM': ScoreCAM,
    'GradCAMPlusPlus': GradCAMPlusPlus,
    'AblationCAM': AblationCAM,
    'XGradCAM': XGradCAM,
    'EigenCAM': EigenCAM,
    'FullGrad': FullGrad,
}


class CAMWrapper:
    def __init__(self, args, model, target_layer):
        if args.method not in cam_methods:
            raise ValueError(
                f'Unsupported CAM method: {args.method}. \n',
                'Supported methods are: {list(cam_methods.keys())}',
            )
        self.cam_method = cam_methods[args.method]
        self.model = self.cam_method(model=model, target_layers=[target_layer], use_cuda=torch.cuda.is_available())

    def __call__(self, input_tensor, target_category=None):
        """
        Args:
            input_tensor:    (B, C, H, W)
            target_category: None          → compute CAMs for all classes
                             list[int]     → compute CAMs for the given class indices only
        Returns:
            predictions:    (B, num_classes) raw logits tensor
            grayscale_cams: (B, len(targets), H, W) numpy array;
                            len(targets) == num_classes when target_category is None
        """
        # First pass: seeds predictions; reuses its CAM rather than discarding it.
        first = target_category[0] if target_category is not None else 0
        all_cams = [self.model(input_tensor=input_tensor, target_category=first)]
        predictions = self.model.outputs    # (B, num_classes) tensor

        remaining = target_category[1:] if target_category is not None else range(1, predictions.shape[1])
        for cls_idx in remaining:
            all_cams.append(self.model(input_tensor=input_tensor, target_category=cls_idx))

        grayscale_cams = np.stack(all_cams, axis=1)     # (B, len(targets), H, W)
        return predictions, grayscale_cams


class CAM2BoxWrapper(CAMWrapper):
    """
    Transforms CAM heatmaps into bounding boxes:

    1. Identify classes with softmax probability above cls_threshold.
    2. For each active class, generate its CAM and threshold at cam_threshold to produce a binary mask.
    3. Extract connected components from the mask and filter by minimum pixel area (box_threshold).
    """
    def __init__(self, args, model, target_layer):
        super().__init__(args, model, target_layer)
        self.cls_threshold = args.cls_threshold  # probability threshold
        self.cam_threshold = args.cam_threshold  # CAM → mask threshold
        self.box_threshold = args.box_thresh     # minimum connected-component area (px)
        self.softmax = nn.Softmax(dim=1)

    def _connected_components(self, mask: torch.BoolTensor) -> torch.LongTensor:
        """
        8-connectivity connected components via iterative min-label propagation.

        Each foreground pixel starts with a unique positive ID; background = 0.
        Each step spreads the minimum neighbour label through a 3x3 window.
        Converges in O(component diameter) steps; early-exits when stable.

        Args:
            mask: (H, W) bool tensor on device
        Returns:
            (H, W) long tensor — unique label per component, 0 for background
        """
        H, W = mask.shape
        device = mask.device
        INF = H * W + 1

        labels = (torch.arange(H * W, device=device, dtype=torch.long) + 1).view(H, W)
        labels = labels * mask.long()  # background = 0

        for _ in range(H + W):  # upper bound on component diameter
            prev = labels

            # Replace background (0) with INF so it never wins a min contest
            eff = labels.masked_fill(~mask, INF).float()

            # Pad borders with INF, then slide a 3x3 min window
            padded = F.pad(eff.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), value=float(INF))
            neighbor_min = (
                padded.unfold(2, 3, 1).unfold(3, 3, 1)   # (1, 1, H, W, 3, 3)
                .reshape(1, 1, H, W, 9)
                .min(dim=-1).values
                .squeeze()
                .long()
            )

            # Only update foreground pixels; take element-wise min with current label
            labels = torch.where(mask, torch.minimum(labels, neighbor_min), labels)

            if torch.equal(labels, prev):
                break

        return labels

    def cam2box(self, grayscale_cam: np.ndarray, class_idx: int) -> list:
        """
        Convert a single CAM heatmap to bounding boxes, fully on-device.

        Args:
            grayscale_cam: (H, W) float32 numpy array in [0, 1] from pytorch_grad_cam
            class_idx: class this CAM was computed for
        Returns:
            list of {'bbox': [x1, y1, x2, y2], 'class': int} for each kept component
        """
        device = next(self.model.model.parameters()).device

        # One-time CPU→GPU transfer; everything after stays on device
        cam_t = torch.as_tensor(grayscale_cam, device=device)
        mask = cam_t > self.cam_threshold

        if not mask.any():
            return []

        labels = self._connected_components(mask)   # (H, W) on device

        boxes = []
        for label_id in labels.unique():
            if label_id == 0:                       # skip background
                continue
            component = labels == label_id
            area = component.sum().item()
            if area < self.box_threshold:
                continue
            ys, xs = component.nonzero(as_tuple=True)
            boxes.append({
                'bbox': [xs.min().item(), ys.min().item(),
                         xs.max().item() + 1, ys.max().item() + 1],  # x2/y2 exclusive
                'class': class_idx,
            })
        return boxes

    def __call__(self, input_tensor):
        # Parent returns (B, num_classes, H, W) — one CAM per class
        predictions, all_grayscale_cams = super().__call__(input_tensor)
        probs = self.softmax(predictions)   # (B, num_classes), stays on device

        batch_size = input_tensor.shape[0]
        all_boxes = [[] for _ in range(batch_size)]

        for cls_idx in range(probs.shape[1]):
            active = probs[:, cls_idx] > self.cls_threshold
            if not active.any():
                continue
            for b in range(batch_size):
                if not active[b]:
                    continue
                boxes = self.cam2box(all_grayscale_cams[b, cls_idx], cls_idx)
                all_boxes[b].extend(boxes)

        return probs, all_boxes
