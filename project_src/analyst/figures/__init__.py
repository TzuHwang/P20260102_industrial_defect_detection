import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, roc_curve


__all__ = [
    'ROCCurve',
    'ConfusionMatrix',
    'LossCurve',
]


class ROCCurve:
    def __init__(self):
        pass

    def __call__(self, outputs, targets, out_dir=None, label_names=None, *args, **kwargs):
        n_classes = outputs.shape[1]
        tick_labels = label_names if label_names else [str(i) for i in range(n_classes)]

        fig, ax = plt.subplots(figsize=(6, 5))
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve((targets == i).astype(int), outputs[:, i])
            auc_score = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{tick_labels[i]} (AUC={auc_score:.4f})")
        ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title("ROC Curve")
        ax.legend(loc='lower right', fontsize=8)
        fig.tight_layout()

        if out_dir is not None:
            fig.savefig(os.path.join(out_dir, 'roc_curve.png'), dpi=150, bbox_inches='tight')
        plt.show()
        plt.close(fig)


class ConfusionMatrix:
    def __init__(self):
        pass

    def __call__(self, outputs, targets, out_dir=None, label_names=None, *args, **kwargs):
        n_classes = outputs.shape[1]
        tick_labels = label_names if label_names else [str(i) for i in range(n_classes)]
        preds = outputs.argmax(axis=1)
        cm = confusion_matrix(targets, preds)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')
        ax.set_yticklabels(tick_labels)
        threshold = cm.max() / 2
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                ax.text(c, r, str(cm[r, c]), ha='center', va='center',
                        color='white' if cm[r, c] > threshold else 'black')
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        fig.tight_layout()

        if out_dir is not None:
            fig.savefig(os.path.join(out_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
        plt.show()
        plt.close(fig)


class LossCurve:
    def __init__(self):
        pass

    def __call__(self, csv_path=None, out_dir=None, *args, **kwargs):
        if csv_path is None:
            return
        df = pd.read_csv(csv_path)

        fig, ax = plt.subplots(figsize=(6, 5))
        for col, label in [('train_loss', 'Train'), ('val_loss', 'Val')]:
            if col in df.columns:
                sub = df[['epoch', col]].dropna()
                ax.plot(sub['epoch'], sub[col], label=label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss Curves")
        ax.legend()
        fig.tight_layout()

        if out_dir is not None:
            fig.savefig(os.path.join(out_dir, 'loss_curve.png'), dpi=150, bbox_inches='tight')
        plt.show()
        plt.close(fig)
