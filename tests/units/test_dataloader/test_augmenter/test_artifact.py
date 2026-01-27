import pytest
from copy import deepcopy
from types import SimpleNamespace

import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim

from project_src.dataloader.transform.augmentation.artifact import BrightnessContrast


@pytest.fixture(
    params=['BrightnessContrast']
)
def testcase(request):
    return request.param


@pytest.fixture
def args(testcase):
    if testcase in ['BrightnessContrast']:
        return SimpleNamespace(
            brightness=0.2,
            contrast=0.2,
            p=1.0,
        )


def to_image(tensorlike_np: torch.Tensor) -> np.ndarray:
    """
    (C, H, W) torch tensor -> (H, W, C) numpy array
    """
    return tensorlike_np.squeeze().transpose(1, 2, 0)


@pytest.fixture
def subject():
    return {
        'inputs': np.random.randint(0, 255, (1, 3, 224, 224), dtype=np.uint8),
        'targets': {'labels': [1]}
    }


def test_BrightnessContrast(args, subject):
    aug_func = BrightnessContrast(args)
    aug_subject = aug_func(deepcopy(subject))

    x_in = subject["inputs"]
    x_out = aug_subject["inputs"]

    # 1. Shape must be preserved
    assert x_out.shape == x_in.shape
    assert subject['targets']['labels'] == aug_subject['targets']['labels']

    # 2. Must NOT be identical
    assert not np.allclose(x_out, x_in), \
        "BrightnessContrast should modify pixel values"

    # 3. SSIM should be high (feature-preserving)
    ssim_value = ssim(
        to_image(x_in),
        to_image(x_out),
        channel_axis=-1,
        data_range=1.0,
    )

    assert ssim_value > 0.9, \
        f"SSIM too low ({ssim_value:.4f}), transformation damages structure"
