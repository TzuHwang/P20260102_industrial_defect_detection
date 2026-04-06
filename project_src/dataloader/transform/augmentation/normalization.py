import numpy as np
import torch

from .__template__ import AugmentationTemplate


class ImageNetNorm(AugmentationTemplate):
    _MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    _STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __init__(self, args=None):
        super().__init__(p=1.0)

    def apply(self, subject: dict) -> dict:
        image = subject['inputs']  # (C, H, W) float32 in [0, 1]
        mean = self._MEAN.to(image.device)
        std = self._STD.to(image.device)
        subject['inputs'] = (image - mean) / std
        return subject


class ToTensor(AugmentationTemplate):
    def __init__(self, args=None):
        super().__init__(p=1.0)

    def apply(self, subject: dict) -> dict:
        image = subject['inputs']
        # (H, W, C) uint8/float -> (C, H, W) float32 in [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        subject['inputs'] = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        return subject
