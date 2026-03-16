"""
Inference module using PyTorch Lightning.

This module provides inference functionality using the trained Lightning model.
"""

import numpy as np
import torch

from project_src.architecture import ArchitectureBuilder
from project_src.dataloader import DataLoaderFactory


@torch.no_grad()
def inference(args):
    """
    Run inference on test data.

    Args:
        args: Namespace with inference configuration loaded from YAML
    """
    # Load model from checkpoint
    checkpoint_path = args.get('checkpoint_path', None)
    if checkpoint_path is None:
        raise ValueError("checkpoint_path must be provided in args for inference")

    arch = ArchitectureBuilder(args.architecture)
    arch.load_pretrained()
    model = arch.get_model()
    model.eval()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Create test dataloader
    test_loader = DataLoaderFactory(args, 'test').get_loader()

    # Run inference
    all_predictions = []
    all_targets = []

    for batch in test_loader:
        inputs, targets = _process_batch(batch, model.criterion)
        inputs = inputs.to(device)

        # Forward pass
        outputs = model(inputs)

        # Get predictions
        _, predicted = torch.max(outputs, dim=1)

        all_predictions.extend(predicted.cpu().numpy())
        all_targets.extend(targets.numpy())

    # Compute final accuracy
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    accuracy = (predictions == targets).mean()

    print(f"Test Accuracy: {accuracy:.4f}")

    return predictions, targets, accuracy


def _process_batch(batch, criterion):
    """Process batch from dataloader."""
    if isinstance(batch, dict):
        inputs = batch['inputs']
        if isinstance(batch['targets'], dict):
            targets = batch['targets']['label']
        else:
            targets = batch['targets']
    elif isinstance(batch, (tuple, list)):
        inputs, targets = batch[0], batch[1]
    else:
        raise ValueError(f"Unexpected batch type: {type(batch)}")

    # Ensure targets are LongTensor for classification
    if targets.dtype == torch.float32 and criterion.__class__.__name__ == 'CrossEntropyLoss':
        targets = targets.long()

    return inputs, targets


def predict_single(model, input_tensor, device=None):
    """
    Run inference on a single input tensor.

    Args:
        model: Trained Lightning model
        input_tensor: Input tensor of shape (C, H, W) or (B, C, H, W)
        device: Device to run inference on

    Returns:
        Predicted class indices and probabilities
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    model.to(device)

    # Ensure input is 4D
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, dim=1)

    return predicted.cpu().numpy(), probabilities.cpu().numpy()
