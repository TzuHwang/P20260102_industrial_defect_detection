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
            'inputs': image (H, W, C)  np.ndarray uint8
            'targets': {
                'mask':   (H, W) np.ndarray          optional
                'boxes':  list of [x1, y1, x2, y2]   optional  pascal_voc / xyxy
                'labels': list of int                 optional  parallel to boxes
            }
        }
        When boxes are present, self.transform is wrapped in A.Compose with
        bbox_params so that coordinates are rescaled / flipped / rotated
        consistently with the image.
        """
        image = subject['inputs']
        targets = subject.get('targets', {})

        mask = targets.get('mask', None)
        boxes = targets.get('boxes', None)
        has_boxes = boxes is not None and len(boxes) > 0

        aug_input = {'image': image}
        if mask is not None:
            aug_input['mask'] = mask

        if has_boxes:
            aug_input['bboxes'] = [list(b) for b in boxes]
            aug_input['labels'] = list(targets.get('labels', [0] * len(boxes)))
            transform = A.Compose(
                [self.transform],
                bbox_params=A.BboxParams(
                    format='pascal_voc',
                    label_fields=['labels'],
                    min_visibility=0.1,
                    clip=True,
                ),
            )
        else:
            transform = self.transform

        result = transform(**aug_input)

        subject['inputs'] = result['image']
        if mask is not None:
            subject['targets']['mask'] = result['mask']
        if has_boxes:
            subject['targets']['boxes'] = [list(b) for b in result['bboxes']]
            subject['targets']['labels'] = list(result['labels'])

        return subject
