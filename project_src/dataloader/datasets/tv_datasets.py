import numpy as np
import torch.utils.data as data
import torchvision.datasets as D  # noqa


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
        # FashionMNIST is grayscale, convert to 3-channel for compatibility
        input = np.array(input)
        if input.ndim == 2:
            input = np.repeat(np.expand_dims(input, axis=2), 3, axis=2)
        else:
            input = np.expand_dims(input, axis=2)

        # data augmentation
        # It is a classification task aug target is unnecessary
        aug_data = self.augmenter({
            'inputs': input,
            'targets': {
                'labels': target
            }
        })

        # change to channel-first format for PyTorch
        aug_data['inputs'] = np.transpose(aug_data['inputs'], (2, 0, 1))

        return aug_data
