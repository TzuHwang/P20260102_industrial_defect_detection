"""
Deployment module for exporting trained models to ONNX format.
"""

import torch

from project_src.architecture import ArchitectureBuilder


def export_onnx(args):
    """
    Export trained model to ONNX format.

    Args:
        args: Namespace with deploy configuration loaded from YAML
    """
    deploy_cfg = args.deploy

    checkpoint_path = getattr(deploy_cfg, 'checkpoint_path', None)
    if checkpoint_path is None:
        raise ValueError("deploy.checkpoint_path must be provided in config for ONNX export")

    output_path = getattr(deploy_cfg, 'output_path', 'outputs/model.onnx')
    input_shape = getattr(deploy_cfg, 'input_shape', [1, 3, 224, 224])
    opset_version = getattr(deploy_cfg, 'opset_version', 17)
    dynamic_axes = getattr(deploy_cfg, 'dynamic_axes', None)

    # Build model and load weights
    arch = ArchitectureBuilder(args.architecture)
    arch.load_pretrained()
    model = arch.get_model()

    # Load checkpoint weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    # Strip Lightning prefix if present
    state_dict = {k.removeprefix('model.'): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Build dummy input
    dummy_input = torch.zeros(input_shape)

    # Resolve dynamic axes: accepts dict or list of axis-name strings
    if isinstance(dynamic_axes, list):
        dynamic_axes = {'input': {0: 'batch'}, 'output': {0: 'batch'}}

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )

    print(f"Model exported to ONNX: {output_path}")
    return output_path
