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
        self.metric_mode = getattr(args, 'metric_mode', 'min')
        self.monitor_metric = getattr(args, 'monitor_metric', 'val_loss')
        self.best_val_metric = float('inf') if self.metric_mode == 'min' else float('-inf')

        # Accumulate outputs and targets for epoch-end metric computation
        self._train_probs, self._train_logits, self._train_targets = [], [], []
        self._val_probs, self._val_logits, self._val_targets = [], [], []
        self._test_probs, self._test_logits, self._test_targets = [], [], []

        # Accumulate step-level loss scalars (used by _log_metrics for detection tasks)
        self._train_losses: list = []
        self._val_losses: list = []
        self._test_losses: list = []

        # Accumulate per-image detection predictions / GT for mAP (detection tasks only)
        self._val_det_preds: list = []
        self._val_det_targets: list = []
        self._test_det_preds: list = []
        self._test_det_targets: list = []

        # Optional detection decoder for mAP computation
        decoder_args = getattr(args.analyst, 'decoder', None)
        if decoder_args is not None:
            from project_src.post_process import PostProcessFactory
            self.det_decoder = PostProcessFactory(decoder_args, model=None)
        else:
            self.det_decoder = None

    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Configure scheduler for Lightning
        scheduler_config = {
            'scheduler': self.scheduler,
            'monitor': getattr(self.args.pipeline, 'monitor_metric', 'val_loss'),
            'interval': getattr(self.args.pipeline, 'scheduler_interval', 'epoch'),
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
        self._train_losses.append(loss.detach().item())
        if not isinstance(probs, dict):
            self._train_probs.append(probs.detach().cpu())
            self._train_logits.append(logits.detach().cpu())
            self._train_targets.append(
                targets.detach().cpu() if not isinstance(targets, dict) else targets['labels'].detach().cpu()
            )
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
        self._val_losses.append(loss.detach().item())
        if not isinstance(probs, dict):
            self._val_probs.append(probs.detach().cpu())
            self._val_logits.append(logits.detach().cpu())
            self._val_targets.append(
                targets.detach().cpu() if not isinstance(targets, dict) else targets['labels'].detach().cpu()
            )
        elif self.det_decoder is not None:
            self._accumulate_det(logits, targets, 'val')
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
        self._test_losses.append(loss.detach().item())
        if not isinstance(outputs, dict):
            self._test_probs.append(outputs.detach().cpu())
            self._test_logits.append(logits.detach().cpu())
            self._test_targets.append(
                targets.detach().cpu() if not isinstance(targets, dict) else targets['labels'].detach().cpu()
            )
        elif self.det_decoder is not None:
            self._accumulate_det(logits, targets, 'test')
        return loss

    def _run(self, batch, *args, **kwargs):
        """Shared process logic for training, validation, and testing."""

        inputs, targets = self._process_batch(batch)

        # Forward pass
        logits = self(inputs.float())
        probs = logits if isinstance(logits, dict) else self.activation(logits)

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
                targets = batch['targets']
            else:
                raise DeprecationWarning('The list of int labels is deprecating for detection task')
        elif isinstance(batch, (tuple, list)):
            inputs, targets = batch[0], batch[1]
        else:
            raise ValueError(f"Unexpected batch type: {type(batch)}")

        # Ensure targets are LongTensor for classification
        if isinstance(targets, dict):
            if targets['labels'].dtype == torch.float32 and self.criterion.__class__.__name__ == 'CrossEntropyLoss':
                targets['labels'] = targets['labels'].long()
        elif targets.dtype == torch.float32 and self.criterion.__class__.__name__ == 'CrossEntropyLoss':
            targets = targets.long()

        return inputs, targets

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""

        # Compute validation metrics at epoch end
        probs = self._val_probs or [{}]   # pass sentinel for detection (empty probs)
        if self._val_probs or self._val_losses:
            self._log_metrics(probs, self._val_logits, self._val_targets, split='val')
            self._val_probs.clear()
            self._val_logits.clear()
            self._val_targets.clear()
            self._val_losses.clear()

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
        probs = self._train_probs or [{}]
        if self._train_probs or self._train_losses:
            self._log_metrics(probs, self._train_logits, self._train_targets, split='train')
            self._train_probs.clear()
            self._train_logits.clear()
            self._train_targets.clear()
            self._train_losses.clear()

    def on_test_epoch_end(self):
        """Called at the end of test epoch."""
        # Compute test metrics at epoch end
        probs = self._test_probs or [{}]
        if self._test_probs or self._test_losses:
            self._log_metrics(probs, self._test_logits, self._test_targets, split='test')
            self._test_probs.clear()
            self._test_logits.clear()
            self._test_targets.clear()
            self._test_losses.clear()

    @torch.no_grad()
    def _accumulate_det(self, logits: dict, targets: dict, split: str):
        """Decode head output → per-image preds/GT arrays for mAP accumulation."""
        import numpy as np
        det_results = self.det_decoder(logits)          # list[dict] per image
        num_boxes = targets['num_boxes']

        preds_list = getattr(self, f'_{split}_det_preds')
        tgts_list = getattr(self, f'_{split}_det_targets')

        for i, det in enumerate(det_results):
            boxes = det['boxes'].cpu().numpy()          # (K, 4)
            scores = det['scores'].cpu().numpy()        # (K,)
            classes = det['classes'].cpu().numpy()      # (K,)
            if len(boxes):
                pred_arr = np.column_stack([boxes, scores, classes])  # (K, 6)
            else:
                pred_arr = np.zeros((0, 6))
            preds_list.append(pred_arr)

            n = int(num_boxes[i]) if isinstance(num_boxes, torch.Tensor) else int(num_boxes)
            gt_boxes = targets['boxes'][i, :n].cpu().numpy()    # (n, 4) xyxy
            gt_labels = targets['labels'][i, :n].cpu().numpy()  # (n,)
            if n:
                tgt_arr = np.column_stack([gt_boxes, gt_labels])  # (n, 5)
            else:
                tgt_arr = np.zeros((0, 5))
            tgts_list.append(tgt_arr)

    def _log_metrics(self, prob_collection, logit_collection, target_collection, split):
        loss_list = getattr(self, f'_{split}_losses', [])
        if isinstance(prob_collection[0], dict):
            # Detection: log epoch-mean loss + mAP (if decoder is configured).
            if loss_list:
                mean_loss = sum(loss_list) / len(loss_list)
                self.log(f'{split}_loss', mean_loss, prog_bar=True,
                         logger=True, sync_dist=True)

            preds_list = getattr(self, f'_{split}_det_preds', [])
            tgts_list = getattr(self, f'_{split}_det_targets', [])
            if preds_list and 'MeanAveragePrecision' in self.metrics.metrics:
                map_val = self.metrics.metrics['MeanAveragePrecision'](preds_list, tgts_list)
                self.log(f'{split}_MeanAveragePrecision', float(map_val),
                         prog_bar=True, logger=True, sync_dist=True)
                preds_list.clear()
                tgts_list.clear()
            return {}
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


def _parse_devices(devices):
    """Parse a device string into (accelerator, devices) for Lightning Trainer.

    Accepted forms:
        'cpu'      → ('cpu', 1)
        'cuda:0'   → ('cuda', [0])
        'cuda:1'   → ('cuda', [1])
        'auto'     → ('auto', 'auto')
    """
    if isinstance(devices, str):
        if devices == 'cpu':
            return 'cpu', 1
        if devices.startswith('cuda:'):
            return 'cuda', [int(devices.split(':')[1])]
    return 'auto', 'auto'


def train(args):
    """
    Main training function.

    Args:
        args: Namespace with training configuration loaded from YAML

    Returns:
        Trained Lightning trainer and model
    """
    # Set seed for reproducibility
    seed = getattr(args, 'seed', 42)
    L.seed_everything(seed, workers=True)

    # Create Lightning model
    model = LitModel(args)

    # Create dataloaders
    train_loader = DataLoaderFactory(args.dataloader, 'train').get_loader()
    val_loader = DataLoaderFactory(args.dataloader, 'val').get_loader()

    # Create trainer
    accelerator, devices = _parse_devices(getattr(args.pipeline, 'devices', 'auto'))
    trainer = L.Trainer(
        # Training control
        max_epochs=getattr(args.pipeline, 'epochs', 100),

        # Accelerator / device selection
        accelerator=accelerator,
        devices=devices,
        num_nodes=getattr(args.pipeline, 'gpu_num', 1),

        # Fast dev run for debugging
        fast_dev_run=getattr(args.pipeline, 'fast_dev_run', False),

        # setup output directory
        default_root_dir=getattr(args.pipeline, 'save_dir', './outputs'),
    )

    # Train the model (pass ckpt_path to resume from a checkpoint)
    ckpt_path = getattr(args.pipeline, 'resume_ckpt', None)
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)

    # Test on test set if available
    if getattr(args.pipeline, 'run_test', False):
        test_loader = DataLoaderFactory(args.dataloader, 'test').get_loader()
        trainer.test(model, test_loader)
