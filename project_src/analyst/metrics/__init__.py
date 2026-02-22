import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


__all__ = [
    'Accuracy',
    'Precision',
    'Recall',
    'F1Score',
    'AUC',
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
        return np.round(roc_auc_score(targets, outputs, average=self.average), 4)
