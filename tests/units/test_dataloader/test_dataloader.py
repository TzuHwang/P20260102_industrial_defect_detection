import pytest
from types import SimpleNamespace

import numpy as np
import torch

from project_src.dataloader import DataLoaderFactory


"""
The Data_Loader object is required to return a dictionary-based batch structure that
strictly follows the interface defined below. This contract is critical to ensure compatibility
with downstream training, evaluation, and inference pipelines.

Each iteration of Data_Loader must return a dictionary (dict[str, Tensor | ndarray | Any]).
The dictionary must contain at least the following mandatory keys:

    1. inputs: The model input tensor(s).

    2. targets: The corresponding ground-truth annotations (class or mask).

Additional auxiliary keys (e.g., meta, spacing, path, etc.) are permitted but optional.

Furthermore, all tensor-like values in the returned dictionary must be batched, meaning that the
first dimension always represents the batch dimension: (batch, …).
"""


@pytest.fixture(params=[1, 2], ids=['batch=1', 'batch>1'])
def testcase(request):
    return request.param


def validate_data(data, batch_size):
    for k, v in data.items():
        if isinstance(v, dict):
            validate_data(v, batch_size)
        else:
            # 4. All values must be tensor-like and batched
            assert isinstance(v, (torch.Tensor, np.ndarray)), f'{k} must be Tensor/ndarray'
            assert v.ndim >= 1, f'{k} lost batch dimension'

            # 5. All fields must align on batch dim
            assert v.shape[0] == batch_size, \
                f'{k} misaligned: {v.shape[0]} vs {batch_size}'

            # 6. Per-sample shape consistency
            if batch_size > 1:
                assert v[0].shape == v[1].shape, \
                    f'{k} has inconsistent per-sample shapes'


@pytest.fixture
def args(testcase):
    return SimpleNamespace(
        dataset='FashionMNIST',
        augmenters=['BrightnessContrast'],
        batch_size=testcase,
        num_workers=0,
        transform=SimpleNamespace(
            augmenters=['BrightnessContrast'],
            normalizer=[],
            input_size=224,
            BrightnessContrast=SimpleNamespace(
                brightness=0.2,
                contrast=0.2,
                p=1.0,
            ),
        ),
        FashionMNIST=SimpleNamespace(
            data_root='data/test/integration/FashionMNIST'
        ),
    )


def test_dataloader(args, testcase):
    dataloader = DataLoaderFactory(args, 'train').get_loader()
    batch = next(iter(dataloader))

    # 1. Must be dict
    assert isinstance(batch, dict)

    # 2. Mandatory keys
    for key in ('inputs', 'targets'):
        assert key in batch, f'Missing mandatory key: {key}'

    batch_size = batch['inputs'].shape[0]

    # 3. Batch dim must be preserved (never squeezed)
    assert batch_size == testcase, \
        f'Batch dimension collapsed: expected {testcase}, got {batch_size}'

    validate_data(batch, batch_size)
