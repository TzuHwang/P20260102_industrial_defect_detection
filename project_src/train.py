"""
Training module using PyTorch Lightning.

This module provides a LightningModule wrapper for the project's model architecture
and handles the training loop, validation, and logging.
"""

import lightning as L
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from project_src.analyst import StatisticMetrics
from project_src.architecture import ArchitectureBuilder
from project_src.dataloader import DataLoaderFactory
from project_src.utils.pipeline import get_attr


class LitModel(L.LightningModule):
    """
    PyTorch Lightning module for training the model.

    Wraps the ModelFactory and handles optimization, learning rate scheduling,
    and metric logging during training and validation.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        # Setup architecture
        self.arch = ArchitectureBuilder(args.architecture)

        self.arch.load_pretrained()
        self.model = self.arch.get_model()
        self.activation = self.arch.get_activation()
        self.optimizer = self.arch.get_optimizer()
        self.scheduler = self.arch.get_scheduler()
        self.criterion = self.arch.get_loss_funcs()

        # Setup metrics
        self.metrics = StatisticMetrics(args.analyst)

        # Track best validation metric
        metric_mode = get_attr(args, 'metric_mode', 'min')
        self.best_val_metric = float('inf') if metric_mode == 'min' else float('-inf')

        # Accumulate outputs and targets for epoch-end metric computation
        self._train_outputs, self._train_targets = [], []
        self._val_outputs, self._val_targets = [], []
        self._test_outputs, self._test_targets = [], []

    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Configure scheduler for Lightning
        scheduler_config = {
            'scheduler': self.scheduler,
            'monitor': get_attr(self.args.pipeline, 'monitor_metric', 'val_loss'),
            'interval': get_attr(self.args.pipeline, 'scheduler_interval', 'epoch'),
            'frequency': 1,
        }

        # Handle ReduceLROnPlateau specially
        if isinstance(self.scheduler, ReduceLROnPlateau):
            scheduler_config['reduce_on_plateau'] = True

        return {
            'optimizer': self.optimizer,
            'lr_scheduler': scheduler_config,
        }

    def training_step(self, batch, batch_idx):
        """
        Training step.

        Args:
            batch: Tuple of (inputs, targets) from dataloader
            batch_idx: Index of current batch

        Returns:
            Training loss for backpropagation
        """
        loss, outputs, targets = self._run(batch, session='train')
        self._train_outputs.append(outputs)
        self._train_targets.append(targets)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step.

        Args:
            batch: Tuple of (inputs, targets) from dataloader
            batch_idx: Index of current batch

        Returns:
            Validation loss
        """
        loss, outputs, targets = self._run(batch, session='val')
        self._val_outputs.append(outputs)
        self._val_targets.append(targets)
        return loss

    def test_step(self, batch, batch_idx):
        """
        Test step.

        Args:
            batch: Tuple of (inputs, targets) from dataloader
            batch_idx: Index of current batch

        Returns:
            Test loss and accuracy
        """
        loss, outputs, targets = self._run(batch, session='test')
        self._test_outputs.append(outputs)
        self._test_targets.append(targets)
        return loss

    def _run(self, batch, session: str):
        """Shared process logic for training, validation, and testing."""
        inputs, targets = self._process_batch(batch)

        # Forward pass
        outputs = self.activation(self(inputs.float()))

        # Compute loss
        loss = self.criterion.compute_loss_value(outputs, targets)

        return loss, outputs, targets

    def _process_batch(self, batch):
        """
        Process batch from dataloader.

        The dataloader returns augmented data with 'inputs' and 'targets' keys.
        """
        if isinstance(batch, dict):
            inputs = batch['inputs']
            # For classification, get label from targets
            if isinstance(batch['targets'], dict):
                targets = batch['targets']['label']
            else:
                targets = batch['targets']
        elif isinstance(batch, (tuple, list)):
            inputs, targets = batch[0], batch[1]
        else:
            raise ValueError(f"Unexpected batch type: {type(batch)}")

        # Ensure targets are LongTensor for classification
        if targets.dtype == torch.float32 and self.criterion.__class__.__name__ == 'CrossEntropyLoss':
            targets = targets.long()

        return inputs, targets

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        val_loss = self.trainer.callback_metrics.get('val_loss')
        if val_loss is not None:
            # Track best validation metric
            metric_mode = get_attr(self.args, 'metric_mode', 'min')
            if metric_mode == 'min':
                self.best_val_metric = min(self.best_val_metric, val_loss.item())
            else:
                self.best_val_metric = max(self.best_val_metric, val_loss.item())

            self.log('best_val_metric', self.best_val_metric, prog_bar=True, logger=True)

        # Compute validation metrics at epoch end
        if self._val_outputs:
            all_outputs = torch.cat(self._val_outputs, dim=0).cpu()
            all_targets = torch.cat(self._val_targets, dim=0).cpu()
            metrics = self.metrics.compute_metrics(all_outputs, all_targets)
            for metric_name, metric_value in metrics.items():
                self.log(f'val_{metric_name}', metric_value, prog_bar=True, logger=True)
            self._val_outputs.clear()
            self._val_targets.clear()

    def on_train_epoch_end(self):
        """Called at the end of training epoch."""
        # Compute training metrics at epoch end
        if self._train_outputs:
            all_outputs = torch.cat(self._train_outputs, dim=0).cpu()
            all_targets = torch.cat(self._train_targets, dim=0).cpu()
            metrics = self.metrics.compute_metrics(all_outputs, all_targets)
            for metric_name, metric_value in metrics.items():
                self.log(f'train_{metric_name}', metric_value, prog_bar=True, logger=True, sync_dist=True)
            self._train_outputs.clear()
            self._train_targets.clear()

    def on_test_epoch_end(self):
        """Called at the end of test epoch."""
        # Compute test metrics at epoch end
        if self._test_outputs:
            all_outputs = torch.cat(self._test_outputs, dim=0).cpu()
            all_targets = torch.cat(self._test_targets, dim=0).cpu()
            metrics = self.metrics.compute_metrics(all_outputs, all_targets)
            for metric_name, metric_value in metrics.items():
                self.log(f'test_{metric_name}', metric_value, prog_bar=True, logger=True)
            self._test_outputs.clear()
            self._test_targets.clear()


def train(args):
    """
    Main training function.

    Args:
        args: Namespace with training configuration loaded from YAML

    Returns:
        Trained Lightning trainer and model
    """
    # Set seed for reproducibility
    seed = get_attr(args, 'seed', 42)
    L.seed_everything(seed, workers=True)

    # Create Lightning model
    model = LitModel(args)

    # Create dataloaders
    train_loader = DataLoaderFactory(args.dataloader, 'train').get_loader()
    val_loader = DataLoaderFactory(args.dataloader, 'val').get_loader()

    # Create trainer
    trainer = L.Trainer(
        # Training control
        max_epochs=get_attr(args.pipeline, 'epochs', 100),

        # Accelerator selection (auto-detect GPU/TPU/CPU)
        devices=get_attr(args.pipeline, 'devices', 'auto'),
        num_nodes=get_attr(args.pipeline, 'gpu_num', 1),

        # Fast dev run for debugging
        fast_dev_run=get_attr(args.pipeline, 'fast_dev_run', False),

        # setup output directory
        default_root_dir=get_attr(args.pipeline, 'save_dir', './outputs'),
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Test on test set if available
    if get_attr(args, 'test_split', False):
        test_loader = DataLoaderFactory(args, 'test').get_loader()
        trainer.test(model, test_loader)
