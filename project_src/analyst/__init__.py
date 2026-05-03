import numpy as np

from . import metrics, figures


class StatisticMetrics:
    """Factory class to create and manage metrics."""

    def __init__(self, args, criterion=None):
        self.criterion = criterion
        self.metric_names = args.metric_names
        self.plot_names = args.plot_names
        self.args = args
        self.metrics = {}
        self.plotters = {}

        self._init_metrics()
        self._init_plotters()

    def _init_metrics(self):
        """Initialize metrics based on the configuration."""
        for name in self.metric_names:
            if name in metrics.__dict__:
                self.metrics[name] = metrics.__dict__.get(name)(self.args)

    def _init_plotters(self):
        """Initialize plotters for visualizing metrics."""
        for name in self.plot_names:
            if name in figures.__dict__:
                self.plotters[name] = figures.__dict__.get(name)()

    def get_metrics(self):
        """Get all initialized metrics."""
        return self.metrics

    def compute_metrics(self, outputs, logits, targets):
        """Compute all metrics and return results as a dictionary.

        Args:
            outputs: Activated model predictions (probabilities).
            logits: Raw model logits.
            targets: Ground truth labels.

        Returns:
            dict: Dictionary mapping metric names to their computed values.
        """

        results = {}
        for name, metric in self.metrics.items():
            if name not in ['AUC']:
                outputs_ = outputs.argmax(dim=1)
            else:
                outputs_ = outputs
            results[name] = metric(outputs_, targets)
        if self.criterion:
            results['loss'] = np.float64(
                self.criterion.compute_loss_value(outputs, logits, targets).numpy()
            )
        return results

    def get_metric(self, name):
        """Get a specific metric by name.

        Args:
            name: Name of the metric.

        Returns:
            The metric instance, or None if not found.
        """
        return self.metrics.get(name)

    def plot_metrics(self, outputs, targets, out_dir=None, label_names=None):
        outputs = np.array(outputs)
        targets = np.array(targets)

        for plotter in self.plotters.values():
            plotter(outputs, targets, out_dir=out_dir, label_names=label_names)
