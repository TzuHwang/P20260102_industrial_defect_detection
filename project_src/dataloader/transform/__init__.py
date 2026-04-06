import torch.nn as nn
from torchvision.transforms import Compose, ToTensor

from . import augmentation


def compose_transform(args, name='augmenters'):
    transform = []
    if getattr(args, name, None) is None:
        transform.append(nn.Identity())
    else:
        for key, value in getattr(args, name).__dict__.items():
            if key in augmentation.__all__:
                transform.append(augmentation.__dict__.get(key)(value))
    return Compose(transform)


class Augmenter:
    def __init__(self, args, split):
        self.split = split
        self.train_transform = compose_transform(args, name='augmenters')
        self.util_transform = compose_transform(args, name='normalizers')

    def __call__(self, subject: dict):
        if self.split in ['train', 'final_train']:
            subject = self.train_transform(subject)
        return self.util_transform(subject)
