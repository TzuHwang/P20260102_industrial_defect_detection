import random
from typing import Any, Dict, Mapping

import albumentations as A
import torch.nn as nn


class AugmentationTemplate(nn.Module):
    """
    Base class for all dict-based augmentations.

    Input / Output contract:
        subject: Dict[str, Any]
        must contain at least:
            - 'inputs'
            - 'targets'

    The augmentation may add / modify other keys (meta, masks, etc.)
    """

    REQUIRED_KEYS = ('inputs', 'targets')

    def __init__(self, p: float = 1.0):
        super().__init__()
        self.p = float(p)

    # ----------------------------
    # Public API
    # ----------------------------
    def forward(self, subject: Dict[str, Any]) -> Dict[str, Any]:
        self._validate_subject(subject)

        if not self._should_apply():
            return subject

        return self.apply(subject)

    # ----------------------------
    # To be overridden by subclasses
    # ----------------------------
    def apply(self, subject: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform augmentation.

        MUST return a dict with the same contract.
        """
        _ = subject
        raise NotImplementedError

    # ----------------------------
    # Internal helpers
    # ----------------------------
    def _should_apply(self) -> bool:
        return self.p >= 1.0 or random.random() < self.p

    @classmethod
    def _validate_subject(cls, subject: Mapping[str, Any]):
        if not isinstance(subject, Mapping):
            raise TypeError(f'subject must be a mapping, got {type(subject)}')

        missing = [k for k in cls.REQUIRED_KEYS if k not in subject]
        if missing:
            raise KeyError(f'subject missing required keys: {missing}')


class AlbumentationsAPI(AugmentationTemplate):
    def __init__(self, p=1.):
        super().__init__(p)
        self.transform = None

    def apply(self, subject: dict) -> dict:
        """
        subject: {
            'inputs': image (H, W, C),
            'targets': {
                'mask': mask (H, W),
                'boxes': boxes (Nx4),
                ...
            }
        }
        """
        image = subject['inputs']
        targets = subject.get('targets', {})

        mask = targets.get('mask', None)
        boxes = targets.get('boxes', None)

        # Albumentations expects a dict with 'image' and optional 'mask' or 'bboxes'
        aug_input = {'image': image}
        if mask is not None:
            aug_input['mask'] = mask
        if boxes is not None:
            # Albumentations expects boxes as list of [x_min, y_min, x_max, y_max]
            aug_input['bboxes'] = boxes
            aug_input['bbox_params'] = A.BboxParams(format='pascal_voc', label_fields=['labels'])
            # Ensure 'labels' exists
            aug_input['labels'] = targets.get('labels', [0] * len(boxes))

        result = self.transform(**aug_input)

        # Update subject
        subject['inputs'] = result['image']
        if mask is not None:
            subject['targets']['mask'] = result['mask']
        if boxes is not None:
            subject['targets']['boxes'] = result['bboxes']
            subject['targets']['labels'] = result['labels']

        return subject
