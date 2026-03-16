from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from project_src.architecture.loss import LossFactory


@pytest.fixture()
def args():
    return SimpleNamespace(
        losses=['CrossEntropyLoss', 'MSELoss'],
        channel_weights=None,
        loss_weights=[1.0, 0.5],
    )


def test_factory_initialization_computation(args):
    """Test that the LossFactory initializes the correct loss functions."""
    loss_factory = LossFactory(args)
    loss_fcns = loss_factory.get_loss_fcns()

    assert 'CrossEntropyLoss' in loss_fcns, "CrossEntropyLoss should be initialized"
    assert 'MSELoss' in loss_fcns, "MSELoss should be initialized"
    assert isinstance(loss_fcns['CrossEntropyLoss'], nn.CrossEntropyLoss)
    assert isinstance(loss_fcns['MSELoss'], nn.MSELoss)

    preds = torch.randn(10, 5)
    targets = torch.randint(0, 5, (10, 5)).float()
    loss_values = loss_factory.compute_loss_value(preds, targets)
    assert 'CrossEntropyLoss' in loss_factory.loss_values, "CrossEntropyLoss value should be computed"
    assert 'MSELoss' in loss_factory.loss_values, "MSELoss value should be computed"
    np.testing.assert_allclose(
        loss_values,
        loss_factory.loss_values['CrossEntropyLoss'] + loss_factory.loss_values['MSELoss']
    )
