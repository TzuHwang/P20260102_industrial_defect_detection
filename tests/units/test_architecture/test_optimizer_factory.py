from types import SimpleNamespace

import pytest
import torch.nn as nn
import torch.optim as optim

from project_src.architecture.optimizer import OptimizerFactory


@pytest.fixture
def model():
    """A simple model for testing."""
    return nn.Linear(10, 1)


def test_optimizer_factory_with_explicit_name_and_args(model):
    """OptimizerFactory should build requested optimizer and pass args."""

    args = SimpleNamespace(name='sgd', lr=0.01, momentum=0.9)

    opt = OptimizerFactory(model.parameters(), args).get_optimizer()
    assert isinstance(opt, optim.SGD)
    assert opt.defaults['lr'] == 0.01
    assert pytest.approx(opt.defaults.get('momentum', 0.0)) == 0.9


def test_optimizer_factory_default_and_unknown_name(model):
    """Default optimizer is Adam; unknown names fall back to Adam."""
    args_default = SimpleNamespace()
    opt_default = OptimizerFactory(model.parameters(), args_default).get_optimizer()
    assert isinstance(opt_default, optim.Adam)

    args_unknown = SimpleNamespace(name='nope', lr=0.005)
    opt_unknown = OptimizerFactory(model.parameters(), args_unknown).get_optimizer()
    assert isinstance(opt_unknown, optim.Adam)
    assert opt_unknown.defaults['lr'] == 0.005
