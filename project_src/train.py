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
        self.metrics = StatisticMetrics(args.analyst, self.criterion)

        # Track best validation metric
        self.metric_mode = get_attr(args, 'metric_mode', 'min')
        self.monitor_metric = get_attr(args, 'monitor_metric', 'val_loss')
        self.best_val_metric = float('inf') if self.metric_mode == 'min' else float('-inf')

        # Accumulate outputs and targets for epoch-end metric computation
        self._train_probs, self._train_logits, self._train_targets = [], [], []
        self._val_probs, self._val_logits, self._val_targets = [], [], []
        self._test_probs, self._test_logits, self._test_targets = [], [], []

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

    def training_step(self, batch, *args, **kwargs):
        """
        Training step.

        Args:
            batch: Tuple of (inputs, targets) from dataloader
            batch_idx: Index of current batch

        Returns:
            Training loss for backpropagation
        """

        loss, probs, logits, targets = self._run(batch, session='train')
        self._train_probs.append(probs)
        self._train_logits.append(logits)
        self._train_targets.append(targets)
        return loss

    def validation_step(self, batch, *args, **kwargs):
        """
        Validation step.

        Args:
            batch: Tuple of (inputs, targets) from dataloader
            batch_idx: Index of current batch

        Returns:
            Validation loss
        """

        loss, probs, logits, targets = self._run(batch, session='val')
        self._val_probs.append(probs)
        self._val_logits.append(logits)
        self._val_targets.append(targets)
        return loss

    def test_step(self, batch, *args, **kwargs):
        """
        Test step.

        Args:
            batch: Tuple of (inputs, targets) from dataloader
            batch_idx: Index of current batch

        Returns:
            Test loss and accuracy
        """
        loss, outputs, logits, targets = self._run(batch, session='test')
        self._test_probs.append(outputs)
        self._test_logits.append(logits)
        self._test_targets.append(targets)
        return loss

    def _run(self, batch, *args, **kwargs):
        """Shared process logic for training, validation, and testing."""

        inputs, targets = self._process_batch(batch)

        # Forward pass
        logits = self(inputs.float())
        probs = self.activation(logits)

        # Compute loss
        loss = self.criterion.compute_loss_value(probs, logits, targets)

        return loss, probs, logits, targets

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

        # Compute validation metrics at epoch end
        if self._val_probs:
            self._log_metrics(
                self._val_probs,
                self._val_logits,
                self._val_targets,
                split='val'
            )
            self._val_probs.clear()
            self._val_logits.clear()
            self._val_targets.clear()

        val_metric = self.trainer.callback_metrics.get(self.monitor_metric, None)
        if val_metric is not None:
            # Track best validation metric
            is_best = (
                val_metric < self.best_val_metric
            ) if self.metric_mode == 'min' else (
                val_metric > self.best_val_metric
            )

            if is_best:
                # update best metric if current is better than best
                self.best_val_metric = val_metric
                self.trainer.save_checkpoint(f'{self.trainer.default_root_dir}/best_model.ckpt')

    def on_train_epoch_end(self):
        """Called at the end of training epoch."""
        # Compute training metrics at epoch end
        if self._train_probs:
            self._log_metrics(
                self._train_probs,
                self._train_logits,
                self._train_targets,
                split='train'
            )
            self._train_probs.clear()
            self._train_logits.clear()
            self._train_targets.clear()

    def on_test_epoch_end(self):
        """Called at the end of test epoch."""
        # Compute test metrics at epoch end
        if self._test_probs:
            self._log_metrics(
                self._test_probs,
                self._test_logits,
                self._test_targets,
                split='test'
            )
            self._test_probs.clear()
            self._test_logits.clear()
            self._test_targets.clear()

    def _log_metrics(self, prob_collection, logit_collection, target_collection, split):
        all_probs = torch.cat(prob_collection, dim=0).detach().cpu()
        all_logits = torch.cat(logit_collection, dim=0).detach().cpu()
        all_targets = torch.cat(target_collection, dim=0).detach().cpu()
        metrics = self.metrics.compute_metrics(all_probs, all_logits, all_targets)
        if split == 'test':
            self.metrics.plot_metrics(
                all_probs,
                all_targets,
                out_dir=self.trainer.logger.experiment.log_dir,
            )
        for metric_name, metric_value in metrics.items():
            self.log(
                f'{split}_{metric_name}',
                metric_value,
                prog_bar=True,
                logger=True,
                sync_dist=True
            )
        return metrics


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

    # Train the model (pass ckpt_path to resume from a checkpoint)
    ckpt_path = get_attr(args.pipeline, 'resume_ckpt', None)
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)

    # Test on test set if available
    if get_attr(args.pipeline, 'run_test', False):
        test_loader = DataLoaderFactory(args.dataloader, 'test').get_loader()
        trainer.test(model, test_loader)
