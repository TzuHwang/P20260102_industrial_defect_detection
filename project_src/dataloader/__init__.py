import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler

from . import datasets
from .transform import Augmenter


class DataLoaderFactory:
    def __init__(self, args, split):
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        augmenter = Augmenter(args.transform, split)

        if args.dataset.name in datasets.__all__:
            self.data = datasets.__dict__.get(args.dataset.name)(args.dataset, split, augmenter)

    def get_loader(self):
        indices = np.arange(len(self.data))
        sampler = SubsetRandomSampler(indices)

        return DataLoader(
            self.data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=sampler,
            drop_last=True
        )
