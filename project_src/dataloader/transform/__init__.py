from types import SimpleNamespace

import torch.nn as nn
import cv2
from torchvision.transforms import Compose

from . import augmentation


def compose_util_transform(args):
    util_transform = [
        augmentation.__dict__.get(norm)(
            getattr(args, norm)) for norm in args.normalizer if norm in augmentation.__all__]
    util_transform.append(
        augmentation.Resize(
            SimpleNamespace(
                input_size=args.input_size,
                interpolation='LINEAR',
                mask_interpolation='NEAREST',
                p=1.
            )
        )
    )
    return Compose(util_transform)


class Augmenter:
    def __init__(self, args, split):
        self.split = split

        if args.augmenters is None:
            self.train_transform = nn.Identity()
        else:
            self.train_transform = Compose([
                augmentation.__dict__.get(aug)(
                    getattr(args, aug)) for aug in args.augmenters if aug in augmentation.__all__])

        self.util_transform = compose_util_transform(args)

    def __call__(self, subject: dict):
        if self.split in ['train', 'final_train']:
            subject = self.train_transform(subject)
        return self.util_transform(subject)
