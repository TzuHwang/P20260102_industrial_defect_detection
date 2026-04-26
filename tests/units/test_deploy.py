"""Tests for the deploy module — verifies PT and ONNX outputs are numerically close."""

import os
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from project_src.architecture import ArchitectureBuilder
from project_src.deploy import export_onnx
from project_src.utils.file_dealer import load_yaml_as_ns


TEMPLATE_YML = 'configs/yamls/template_config.yaml'


@pytest.fixture
def args(tmp_path):
    args = load_yaml_as_ns(TEMPLATE_YML)
    args.task = 'deploy'
    args.deploy = SimpleNamespace(
        checkpoint_path=None,
        output_path=str(tmp_path / 'model.onnx'),
        input_shape=[1, 3, 28, 28],
        opset_version=17,
        dynamic_axes=None,
    )
    return args


@pytest.fixture
def pt_model(args):
    arch = ArchitectureBuilder(args.architecture)
    model = arch.get_model()
    model.eval()
    return model


@pytest.fixture
def onnx_path(args, pt_model, tmp_path):
    ckpt_path = str(tmp_path / 'dummy.ckpt')
    torch.save({'state_dict': {f'model.{k}': v for k, v in pt_model.state_dict().items()}}, ckpt_path)
    args.deploy.checkpoint_path = ckpt_path
    return export_onnx(args)


def test_onnx_file_created(onnx_path):
    assert os.path.exists(onnx_path)
    assert os.path.getsize(onnx_path) > 0


def test_pt_onnx_output_close(args, pt_model, onnx_path):
    try:
        import onnxruntime as ort
    except ImportError:
        pytest.skip('onnxruntime not installed')

    dummy = torch.randn(args.deploy.input_shape)

    with torch.no_grad():
        pt_out = pt_model(dummy).numpy()

    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    onnx_out = session.run(None, {input_name: dummy.numpy()})[0]

    np.testing.assert_allclose(pt_out, onnx_out, rtol=1e-4, atol=1e-5)


def test_onnx_output_shape(args, pt_model, onnx_path):
    try:
        import onnxruntime as ort
    except ImportError:
        pytest.skip('onnxruntime not installed')

    dummy = torch.randn(args.deploy.input_shape)
    num_classes = args.architecture.model.head.num_classes

    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    onnx_out = session.run(None, {input_name: dummy.numpy()})[0]

    assert onnx_out.shape == (args.deploy.input_shape[0], num_classes)
