import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

from .iou import IoU, Dice
from .map import MeanAveragePrecision

__all__ = [
    'Accuracy',
    'Precision',
    'Recall',
    'F1Score',
    'AUC',
    'IoU',
    'Dice',
    'MeanAveragePrecision',
]


class Accuracy:
    def __init__(self, args):
        self.average = args.average

    def __call__(self, outputs, targets):
        return np.round(accuracy_score(targets, outputs), 4)


class Precision:
    def __init__(self, args):
        self.average = args.average

    def __call__(self, outputs, targets):
        return np.round(precision_score(targets, outputs, average=self.average, zero_division=0), 4)


class Recall:
    def __init__(self, args):
        self.average = args.average

    def __call__(self, outputs, targets):
        return np.round(recall_score(targets, outputs, average=self.average, zero_division=0), 4)


class F1Score:
    def __init__(self, args):
        self.average = args.average

    def __call__(self, outputs, targets):
        return np.round(f1_score(targets, outputs, average=self.average, zero_division=0), 4)


class AUC:
    def __init__(self, args):
        self.average = args.average

    def __call__(self, outputs, targets):
        labels = list(range(outputs.shape[1]))
        return np.round(roc_auc_score(targets, outputs, average=self.average, multi_class='ovr', labels=labels), 4)
