import os

import numpy as np
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
                'label': target
            }
        })
        print(aug_data['inputs'].max(), aug_data['inputs'].min(), aug_data['inputs'].dtype)
        return aug_data
