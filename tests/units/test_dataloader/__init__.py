import pytest
from types import SimpleNamespace

import numpy as np
import torch

from project_src.dataloader import Data_Loader


"""
The Data_Loader object is required to return a dictionary-based batch structure that
strictly follows the interface defined below. This contract is critical to ensure compatibility
with downstream training, evaluation, and inference pipelines.

Each iteration of Data_Loader must return a dictionary (dict[str, Tensor | ndarray | Any]).
The dictionary must contain at least the following mandatory keys:

    1. inputs: The model input tensor(s).

    2. labels: The corresponding ground-truth annotations (class or mask).

Additional auxiliary keys (e.g., meta, spacing, path, etc.) are permitted but optional.

Furthermore, all tensor-like values in the returned dictionary must be batched, meaning that the
first dimension always represents the batch dimension: (batch, …).
"""


@pytest.fixture(params=[1, 2], ids=["batch=1", "batch>1"])
def testcase(request):
    return request.param


@pytest.fixture
def args(testcase):
    return SimpleNamespace(
        batch_size=testcase,
    )



def test_dataloader(args):
    dataloader=Data_Loader(args)
    batch = next(iter(dataloader))

    # 1. Must be dict
    assert isinstance(batch, dict)

    # 2. Mandatory keys
    for key in ("inputs", "labels"):
        assert key in batch, f"Missing mandatory key: {key}"

    batch_size = batch["inputs"].shape[0]

    # 3. Batch dim must be preserved (never squeezed)
    assert batch_size == testcase, \
        f"Batch dimension collapsed: expected {testcase}, got {batch_size}"

    # 4. All values must be tensor-like and batched
    for k, v in batch.items():
        assert isinstance(v, (torch.Tensor, np.ndarray)), f"{k} must be Tensor/ndarray"
        assert v.ndim >= 1, f"{k} lost batch dimension"

    # 5. All fields must align on batch dim
    for k, v in batch.items():
        assert v.shape[0] == batch_size, \
            f"{k} misaligned: {v.shape[0]} vs {batch_size}"

    # 6. Per-sample shape consistency
    if batch_size > 1:
        for k, v in batch.items():
            assert v[0].shape == v[1].shape, \
                f"{k} has inconsistent per-sample shapes"