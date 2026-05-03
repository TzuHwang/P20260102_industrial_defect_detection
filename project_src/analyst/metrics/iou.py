import numpy as np

__all__ = [
    'IoU',
    'Dice',
]


def _per_class_stats(outputs, targets, num_classes):
    outputs = np.asarray(outputs).flatten()
    targets = np.asarray(targets).flatten()
    intersections = np.zeros(num_classes)
    unions = np.zeros(num_classes)
    sums = np.zeros(num_classes)
    for cls in range(num_classes):
        pred = outputs == cls
        gt = targets == cls
        intersections[cls] = np.logical_and(pred, gt).sum()
        unions[cls] = np.logical_or(pred, gt).sum()
        sums[cls] = pred.sum() + gt.sum()
    return intersections, unions, sums


class IoU:
    def __init__(self, args):
        self.average = args.average      # 'binary' | 'micro' | 'macro'
        self.num_classes = args.num_classes

    def __call__(self, outputs, targets):
        if self.average == 'binary':
            outputs = np.asarray(outputs).flatten().astype(bool)
            targets = np.asarray(targets).flatten().astype(bool)
            intersection = np.logical_and(outputs, targets).sum()
            union = np.logical_or(outputs, targets).sum()
            return np.round(intersection / union if union > 0 else 0.0, 4)

        intersections, unions, _ = _per_class_stats(outputs, targets, self.num_classes)
        valid = unions > 0
        if self.average == 'micro':
            score = intersections[valid].sum() / unions[valid].sum() if valid.any() else 0.0
        else:  # macro / mIoU
            score = (intersections[valid] / unions[valid]).mean() if valid.any() else 0.0
        return np.round(score, 4)


class Dice:
    def __init__(self, args):
        self.average = args.average      # 'binary' | 'micro' | 'macro'
        self.num_classes = args.num_classes

    def __call__(self, outputs, targets):
        if self.average == 'binary':
            outputs = np.asarray(outputs).flatten().astype(bool)
            targets = np.asarray(targets).flatten().astype(bool)
            intersection = np.logical_and(outputs, targets).sum()
            denom = outputs.sum() + targets.sum()
            return np.round(2 * intersection / denom if denom > 0 else 0.0, 4)

        intersections, _, sums = _per_class_stats(outputs, targets, self.num_classes)
        valid = sums > 0
        if self.average == 'micro':
            score = 2 * intersections[valid].sum() / sums[valid].sum() if valid.any() else 0.0
        else:  # macro
            score = (2 * intersections[valid] / sums[valid]).mean() if valid.any() else 0.0
        return np.round(score, 4)
