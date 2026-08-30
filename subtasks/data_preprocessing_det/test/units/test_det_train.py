"""Sanity tests for the detection training pipeline.

Creates a mock dataset (synthetic images + split JSON) in a temp directory,
then verifies that the full training pipeline — LitModel creation, training
step, validation step, and a 2-epoch Lightning Trainer run — works end-to-end.
"""

import json

import numpy as np
import pytest
import torch
from PIL import Image

from project_src.train import LitModel
from project_src.utils.file_dealer import load_yaml_as_ns

_SANITY_YAML = 'configs/yamls/internal_train/det/tape_measure_det_sanity.yaml'
_IMG_SIZE = 128  # must match Resize.target_size in the YAML


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def mock_det_split(tmp_path_factory):
    """Create synthetic images and a detection split JSON in a temp dir."""
    root = tmp_path_factory.mktemp('mock_det')

    def _save_image(name):
        arr = np.random.randint(0, 256, (_IMG_SIZE, _IMG_SIZE, 3), dtype=np.uint8)
        p = root / name
        Image.fromarray(arr).save(str(p))
        return str(p)

    paths = [_save_image(f'img_{i:02d}.jpg') for i in range(8)]

    # xywh boxes — all within 128×128
    split = {
        'train': [
            {paths[0]: [{'class': 0, 'bbox': [10, 10, 40, 40]},
                        {'class': 1, 'bbox': [70, 70, 40, 40]}]},
            {paths[1]: [{'class': 0, 'bbox': [5, 5, 30, 50]}]},
            {paths[2]: [{'class': 1, 'bbox': [60, 10, 50, 60]}]},
            {paths[3]: []},                                          # negative
        ],
        'val': [
            {paths[4]: [{'class': 0, 'bbox': [20, 20, 60, 60]}]},
            {paths[5]: [{'class': 1, 'bbox': [10, 50, 40, 60]}]},
        ],
        'test': [
            {paths[6]: [{'class': 0, 'bbox': [30, 30, 50, 50]}]},
            {paths[7]: [{'class': 1, 'bbox': [10, 10, 70, 70]}]},
        ],
    }

    split_path = root / 'mock_det_split.json'
    with open(split_path, 'w') as f:
        json.dump(split, f)

    return str(split_path)


@pytest.fixture(scope='module')
def det_args(mock_det_split):
    """Load the sanity YAML and patch the data_split path."""
    args = load_yaml_as_ns(_SANITY_YAML)
    args.dataloader.dataset.data_split = mock_det_split
    return args


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_litmodel_creation(det_args):
    """LitModel builds correctly for detection: has model, criterion, decoder."""
    model = LitModel(det_args)
    assert model is not None
    assert hasattr(model, 'model')
    assert hasattr(model, 'criterion')
    assert model.det_decoder is not None, 'Detection decoder must be configured'


def test_training_step(det_args):
    """Training step produces a positive, differentiable loss."""
    from project_src.dataloader import DataLoaderFactory

    model = LitModel(det_args)
    model.train()
    loader = DataLoaderFactory(det_args.dataloader, 'train').get_loader()
    batch = next(iter(loader))

    loss = model.training_step(batch, batch_idx=0)

    assert isinstance(loss, torch.Tensor), 'Loss must be a tensor'
    assert loss.requires_grad, 'Loss must require grad for backprop'
    assert loss.item() > 0, 'Loss must be positive'


def test_validation_step_accumulates_preds(det_args):
    """Validation step decodes predictions and accumulates them for mAP."""
    from project_src.dataloader import DataLoaderFactory

    model = LitModel(det_args)
    model.eval()
    loader = DataLoaderFactory(det_args.dataloader, 'val').get_loader()
    batch = next(iter(loader))

    with torch.no_grad():
        loss = model.validation_step(batch, batch_idx=0)

    assert isinstance(loss, torch.Tensor)
    assert len(model._val_det_preds) > 0, 'Det preds must be accumulated after val step'
    assert len(model._val_det_targets) > 0, 'Det targets must be accumulated after val step'

    # Each pred array is (K, 6): [x1, y1, x2, y2, score, class]
    for pred_arr in model._val_det_preds:
        assert pred_arr.ndim == 2
        assert pred_arr.shape[1] == 6

    # Each target array is (N, 5): [x1, y1, x2, y2, class]
    for tgt_arr in model._val_det_targets:
        assert tgt_arr.ndim == 2
        assert tgt_arr.shape[1] == 5


def test_full_trainer_run(det_args, tmp_path):
    """2-epoch Lightning Trainer run completes and logs val_loss + mAP."""
    import lightning as L

    from project_src.dataloader import DataLoaderFactory

    L.seed_everything(42)
    model = LitModel(det_args)
    train_loader = DataLoaderFactory(det_args.dataloader, 'train').get_loader()
    val_loader = DataLoaderFactory(det_args.dataloader, 'val').get_loader()

    trainer = L.Trainer(
        max_epochs=2,
        accelerator='cpu',
        devices=1,
        default_root_dir=str(tmp_path / 'outputs'),
        logger=False,
        enable_progress_bar=False,
    )
    trainer.fit(model, train_loader, val_loader)

    metrics = trainer.callback_metrics
    assert 'val_loss' in metrics, f'val_loss not found; got: {list(metrics.keys())}'
    assert metrics['val_loss'].item() > 0
    assert 'val_MeanAveragePrecision' in metrics, (
        f'val_MeanAveragePrecision not found; got: {list(metrics.keys())}'
    )
