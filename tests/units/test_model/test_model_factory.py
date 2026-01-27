"""
Test module for model architectures.
Tests that our implementations match torchvision's structure and functionality.
"""
from types import SimpleNamespace

import torch
import pytest
from torchvision.models import resnet50

from project_src.models import ModelFactory


@pytest.fixture(params=['resnet50'])
def test_model_architecture(request):
    """Fixture to parametrize model architecture tests."""
    return request.param


@pytest.fixture()
def args(test_model_architecture):
    if test_model_architecture == 'resnet50':
        return SimpleNamespace(
            backbone='ResNet50',
            neck='Identity',
            head='LinearClassifier',
            ResNet50=SimpleNamespace(
                pretrained=True,
                frozen_stages=-1,
                norm_eval=False,
            ),
            Identity=SimpleNamespace(),
            LinearClassifier=SimpleNamespace(
                in_channels=2048,
                num_classes=1000,
                dropout_rate=0.0,
            ),
        )


@pytest.fixture()
def expected_model_structure(test_model_architecture):
    """Fixture to provide expected model structure based on architecture."""
    if test_model_architecture == 'resnet50':
        return resnet50(pretrained=False)
    else:
        raise ValueError(f"Unknown model architecture: {test_model_architecture}")


def test_model_build(args, expected_model_structure):
    """Test that a ResNet50 model can be built and has the same structure as torchvision's implementation."""
    # Create our model (assuming we have a similar interface)
    # For this test, we'll compare with torchvision's implementation
    model = ModelFactory(args)
    torchvision_model = expected_model_structure

    # Compare model structures
    assert len(list(torchvision_model.parameters())) == len(list(model.parameters())), \
        "Number of parameters should match torchvision implementation"

    for param1, param2 in zip(torchvision_model.parameters(), model.parameters()):
        assert param1.shape == param2.shape, "Parameter shapes should match torchvision implementation"
