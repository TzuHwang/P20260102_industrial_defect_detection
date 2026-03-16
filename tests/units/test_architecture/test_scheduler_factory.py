from types import SimpleNamespace

import pytest
import torch.nn as nn
import torch.optim as optim

from project_src.architecture.scheduler import SchedulerFactory


@pytest.fixture
def model():
    """A simple model for testing."""
    return nn.Linear(10, 1)


@pytest.fixture
def optimizer(model):
    """Create an optimizer for testing."""
    return optim.Adam(model.parameters(), lr=0.001)


def test_scheduler_factory_with_explicit_name_and_args(optimizer):
    """SchedulerFactory should build requested scheduler and pass args."""

    args = SimpleNamespace(name='StepLR', step_size=10, gamma=0.5)

    scheduler = SchedulerFactory(optimizer, args).get_scheduler()
    assert scheduler.__class__.__name__ == 'StepLR'
    assert scheduler.step_size == 10
    assert scheduler.gamma == 0.5


def test_scheduler_factory_default_and_unknown_name(optimizer):
    """Default scheduler is ExponentialLR; unknown names fall back to ExponentialLR."""
    args_default = SimpleNamespace(gamma=0.95)
    scheduler_default = SchedulerFactory(optimizer, args_default).get_scheduler()
    assert scheduler_default.__class__.__name__ == 'ExponentialLR'
    assert scheduler_default.gamma == 0.95

    args_unknown = SimpleNamespace(name='nope', gamma=0.9)
    scheduler_unknown = SchedulerFactory(optimizer, args_unknown).get_scheduler()
    assert scheduler_unknown.__class__.__name__ == 'ExponentialLR'
    assert scheduler_unknown.gamma == 0.9


def test_scheduler_factory_with_warmup(optimizer):
    """SchedulerFactory should wrap scheduler with GradualWarmupScheduler when warmup_iters is set."""

    args = SimpleNamespace(name='ExponentialLR', gamma=0.9, warmup_iters=5)

    scheduler = SchedulerFactory(optimizer, args).get_scheduler()
    assert scheduler.__class__.__name__ == 'GradualWarmupScheduler'
    assert scheduler.total_iters == 5


def test_scheduler_factory_warmup_with_different_base_scheduler(optimizer):
    """GradualWarmupScheduler should work with various base schedulers."""

    args = SimpleNamespace(name='CosineAnnealingLR', T_max=50, warmup_iters=10)

    scheduler = SchedulerFactory(optimizer, args).get_scheduler()
    assert scheduler.__class__.__name__ == 'GradualWarmupScheduler'
    assert scheduler.total_iters == 10
