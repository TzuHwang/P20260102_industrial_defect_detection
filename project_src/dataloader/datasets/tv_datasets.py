import numpy as np
import torchvision.datasets as D  # noqa
import torch.utils.data as data


class FashionMNIST(data.Dataset):
    def __init__(self, args, split, augmenter):
        super().__init__()
        train = True if split in ['train', 'final_train'] else False

        self.data = D.FashionMNIST(root=args.data_root, train=train, download=True)
        self.augmenter = augmenter

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        # get input and target
        input, target = self.data.__getitem__(index)

        # The input is a PIL image, target is an integer label
        # Since we are using albumentations for augmentation,
        # we need to convert the PIL image to a numpy array
        input = np.array(input)

        # data augmentation
        # It is a classification task aug target is unnecessary
        aug_data = self.augmenter({
            'inputs': input,
            'targets': {
                'label': target
            }
        })

        return aug_data
