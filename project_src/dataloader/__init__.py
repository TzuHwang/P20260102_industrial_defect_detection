import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler

from . import datasets


class Data_Loader:
    def __init__(self, args, split):
        if args.dataset in datasets.__all__:
            self.data = datasets.__dict__.get(args.dataset)(args, split)

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