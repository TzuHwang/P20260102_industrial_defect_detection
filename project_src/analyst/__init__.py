import torch.nn as nn

from . import metrics


class StatisticMetrics:
    """Factory class to create and manage metrics."""

    def __init__(self, args):
        self.metric_names = args.metric_names
        self.average = args.average
        self.metrics = {}
        self._init_metrics()

    def _init_metrics(self):
        """Initialize metrics based on the configuration."""
        args = type('Args', (), {'average': self.average})()
        for name in self.metric_names:
            if name in metrics.__dict__:
                self.metrics[name] = metrics.__dict__.get(name)(args)

    def get_metrics(self):
        """Get all initialized metrics."""
        return self.metrics

    def compute_metrics(self, outputs, targets):
        """Compute all metrics and return results as a dictionary.

        Args:
            outputs: Model predictions.
            targets: Ground truth labels.

        Returns:
            dict: Dictionary mapping metric names to their computed values.
        """
        outputs = outputs.detach().cpu()
        targets = targets.detach().cpu()
        results = {}
        for name, metric in self.metrics.items():
            if name not in ['AUC']:
                outputs_ = outputs.argmax(dim=1)
            else:
                outputs_ = outputs
            results[name] = metric(outputs_, targets)
        return results

    def get_metric(self, name):
        """Get a specific metric by name.

        Args:
            name: Name of the metric.

        Returns:
            The metric instance, or None if not found.
        """
        return self.metrics.get(name)
