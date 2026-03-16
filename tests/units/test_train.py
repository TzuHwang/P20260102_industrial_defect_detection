"""Tests for the training module."""

from types import SimpleNamespace

import pytest
import torch

from project_src.train import LitModel
from project_src.utils.file_dealer import load_yaml_as_ns


@pytest.fixture
def args():
    """Create sample arguments for testing."""
    template_yml = 'configs/yamls/template_config.yaml'
    args = load_yaml_as_ns(template_yml)
    args.task = 'train'
    return args


def test_lit_model_creation(args):
    """Test that LitModel can be created."""
    model = LitModel(args)
    assert model is not None
    assert hasattr(model, 'model')
    assert hasattr(model, 'criterion')


def test_lit_model_forward(args):
    """Test forward pass through LitModel."""
    model = LitModel(args)
    model.eval()

    # Create dummy input (batch_size=2, channels=3, height=32, width=32)
    x = torch.randn(2, 3, 32, 32)
    output = model(x)

    # Output should be (batch_size, num_classes)
    assert output.shape == (2, 10)


def test_lit_model_training_step(args):
    """Test training step."""
    model = LitModel(args)
    model.train()

    # Create dummy batch
    inputs = torch.randn(2, 3, 32, 32)
    targets = torch.randint(0, 10, (2,)).long()
    batch = (inputs, targets)

    loss = model.training_step(batch, batch_idx=0)
    assert loss is not None
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad


def test_lit_model_validation_step(args):
    """Test validation step."""
    model = LitModel(args)
    model.eval()

    # Create dummy batch
    inputs = torch.randn(2, 3, 32, 32)
    targets = torch.randint(0, 10, (2,)).long()
    batch = (inputs, targets)

    loss = model.validation_step(batch, batch_idx=0)
    assert loss is not None
    assert isinstance(loss, torch.Tensor)


def test_configure_optimizers(args):
    """Test optimizer configuration."""
    model = LitModel(args)
    opt_config = model.configure_optimizers()

    assert 'optimizer' in opt_config
    assert 'lr_scheduler' in opt_config
