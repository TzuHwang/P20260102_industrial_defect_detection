import os

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

from project_src.utils.file_dealer import json_loader


class TapeMeasureInspection(data.Dataset):
    def __init__(self, args, split, augmenter):
        super().__init__()

        self.side = getattr(args, 'side', 'front')
        self.data = json_loader(args.data_split)[split]
        self.augmenter = augmenter

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        record = self.data[index]
        img_path, target = list(record.items())[0]

        img = Image.open(img_path).convert('RGB')
        input = np.array(img)

        aug_data = self.augmenter({
            'inputs': input,
            'targets': {
                'labels': target
            }
        })
        return aug_data


class TapeMeasureDetection(data.Dataset):
    """TapeMeasure inspection dataset for object detection.

    Loads a detection split JSON where each entry maps one image path to its
    annotation list: [{"class": int, "bbox": [x, y, w, h]}, ...].
    Negative images (no defects) have an empty list.

    Boxes are zero-padded to max_boxes; valid count is returned in num_boxes.
    bbox format is [x, y, w, h] matching the XML source.

    NOTE: Only image-level augmentations (BrightnessContrast, Resize, ImageNetNorm,
    ToTensor) are safe here. Geometric augmenters (Flip, Rotate) are not bbox-aware
    and will corrupt box coordinates.

    Args:
        args: Dataset config namespace. Requires:
              - data_split (str): path to detection split JSON.
              - max_boxes (int, optional): pad target to this length. Default 50.
        split: 'train', 'val', or 'test'.
        augmenter: Augmenter instance applying image-only transforms.
    """

    def __init__(self, args, split, augmenter):
        super().__init__()
        self.max_boxes = getattr(args, 'max_boxes', 50)
        raw = json_loader(args.data_split)[split]
        self.records = [(list(e.keys())[0], list(e.values())[0]) for e in raw]
        self.augmenter = augmenter

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index: int):
        img_path, annotations = self.records[index]

        img = Image.open(img_path).convert('RGB')

        valid = annotations[:self.max_boxes]
        # Convert xywh → xyxy for Albumentations (pascal_voc format).
        raw_boxes = [[a['bbox'][0], a['bbox'][1],
                      a['bbox'][0] + a['bbox'][2],
                      a['bbox'][1] + a['bbox'][3]] for a in valid]
        raw_labels = [a['class'] for a in valid]

        aug_data = self.augmenter({
            'inputs': np.array(img),
            'targets': {'boxes': raw_boxes, 'labels': raw_labels},
        })

        # Augmenter may drop boxes clipped below min_visibility; use returned lists.
        aug_boxes = aug_data['targets'].get('boxes', [])
        aug_labels = aug_data['targets'].get('labels', [])
        n = len(aug_boxes)

        boxes = np.zeros((self.max_boxes, 4), dtype=np.float32)
        labels = np.full(self.max_boxes, -1, dtype=np.int64)
        for i, (box, lbl) in enumerate(zip(aug_boxes, aug_labels)):
            boxes[i] = box   # xyxy, already rescaled by Resize
            labels[i] = lbl

        aug_data['targets'] = {
            'boxes': torch.from_numpy(boxes),    # (max_boxes, 4) xyxy
            'labels': torch.from_numpy(labels),  # (max_boxes,)  -1 = padding
            'num_boxes': n,
        }
        return aug_data
