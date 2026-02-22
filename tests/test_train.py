"""Tests for the training module."""

import pytest
import torch
from types import SimpleNamespace

from project_src.train import LitModel, get_callbacks


@pytest.fixture
def sample_args():
    """Create sample arguments for testing."""
    args = SimpleNamespace(
        dataset='FashionMNIST',
        FashionMNIST=SimpleNamespace(data_root='./data'),
        batch_size=8,
        num_workers=0,
        transform=SimpleNamespace(
            augmenters=None,
            normalizer=['Normalize'],
            input_size=[32, 32]
        ),
        Normalize=SimpleNamespace(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        Resize=SimpleNamespace(input_size=[32, 32]),
        backbone=SimpleNamespace(
            name='ResNet18',
            pretrained=False,
            frozen_stages=-1,
            norm_eval=False
        ),
        neck=SimpleNamespace(name='Identity'),
        head=SimpleNamespace(
            name='LinearClassifier',
            in_channels=512,
            num_classes=10,
            dropout_rate=0.1
        ),
        losses=['ce'],
        channel_weights=None,
        loss_weights=None,
        optimizer=SimpleNamespace(
            name='adam',
            lr=0.001,
            weight_decay=0.01
        ),
        scheduler=SimpleNamespace(
            scheduler=SimpleNamespace(
                name='exponential',
                gamma=0.99
            ),
            warmup_iters=0
        ),
        epochs=1,
        seed=42,
        monitor_metric='val_loss',
        metric_mode='min',
        checkpoint_dir='./test_checkpoints',
        save_top_k=1,
        early_stopping=False,
    )
    return args


def test_lit_model_creation(sample_args):
    """Test that LitModel can be created."""
    model = LitModel(sample_args)
    assert model is not None
    assert hasattr(model, 'model')
    assert hasattr(model, 'criterion')


def test_lit_model_forward(sample_args):
    """Test forward pass through LitModel."""
    model = LitModel(sample_args)
    model.eval()

    # Create dummy input (batch_size=2, channels=3, height=32, width=32)
    x = torch.randn(2, 3, 32, 32)
    output = model(x)

    # Output should be (batch_size, num_classes)
    assert output.shape == (2, 10)


def test_lit_model_training_step(sample_args):
    """Test training step."""
    model = LitModel(sample_args)
    model.train()

    # Create dummy batch
    inputs = torch.randn(2, 3, 32, 32)
    targets = torch.randint(0, 10, (2,)).long()
    batch = (inputs, targets)

    loss = model.training_step(batch, batch_idx=0)
    assert loss is not None
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad


def test_lit_model_validation_step(sample_args):
    """Test validation step."""
    model = LitModel(sample_args)
    model.eval()

    # Create dummy batch
    inputs = torch.randn(2, 3, 32, 32)
    targets = torch.randint(0, 10, (2,)).long()
    batch = (inputs, targets)

    loss = model.validation_step(batch, batch_idx=0)
    assert loss is not None
    assert isinstance(loss, torch.Tensor)


def test_configure_optimizers(sample_args):
    """Test optimizer configuration."""
    model = LitModel(sample_args)
    opt_config = model.configure_optimizers()

    assert 'optimizer' in opt_config
    assert 'lr_scheduler' in opt_config


def test_get_callbacks(sample_args):
    """Test callback creation."""
    callbacks = get_callbacks(sample_args)
    assert len(callbacks) > 0
