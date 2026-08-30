"""Draw up to N sample images per defect class with bounding boxes.

Target-class boxes are drawn in red; other-class boxes in green.
One PNG grid (2 rows × 5 cols) is saved per class.
"""

import random
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


CLASS_NAMES = {
    1: '表面脏污',
    2: '波浪',
    3: '残缺',
    4: '表面损伤',
    5: '烤焦起泡',
    6: '烤漆色差_青线割线',
    7: '空白接头',
    8: '黑块',
    9: '前工序接头',
    10: '印刷重',
    11: '上光起泡',
}

# BGR colors for cv2.rectangle
_TARGET_BGR = (64, 64, 255)   # red
_OTHER_BGR = (64, 255, 64)    # green
_BOX_THICK = 3

# Normalized RGB for matplotlib legend patches
_TARGET_RGB = tuple(v / 255 for v in reversed(_TARGET_BGR))
_OTHER_RGB = tuple(v / 255 for v in reversed(_OTHER_BGR))


def _draw_boxes(img_bgr, annotations, target_class):
    img = img_bgr.copy()
    for ann in annotations:
        x, y, w, h = (int(v) for v in ann['bbox'])
        color = _TARGET_BGR if ann['class'] == target_class else _OTHER_BGR
        cv2.rectangle(img, (x, y), (x + w, y + h), color, _BOX_THICK)
    return img


def _sample_by_class(records, n, seed):
    """Group records by class; sample up to n per class."""
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for rec in records:
        seen = set()
        for ann in rec['annotations']:
            cls = ann['class']
            if cls not in seen:
                seen.add(cls)
                by_class[cls].append(rec)
    return {cls: rng.sample(recs, min(n, len(recs))) for cls, recs in by_class.items()}


def _plot_class_grid(samples, class_id, output_path, n):
    ncols = min(5, len(samples))
    nrows = (len(samples) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = np.array(axes).reshape(-1)

    class_name = CLASS_NAMES.get(class_id, str(class_id))
    fig.suptitle(f'Class {class_id} — {class_name}  ({len(samples)} specimens)', fontsize=13)

    for i, rec in enumerate(samples):
        img_path = Path(rec['file_dir']) / rec['filename']
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            axes[i].set_visible(False)
            continue
        drawn = _draw_boxes(img_bgr, rec['annotations'], class_id)
        axes[i].imshow(cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB))
        axes[i].set_title(rec['filename'], fontsize=7)
        axes[i].axis('off')

    for j in range(len(samples), len(axes)):
        axes[j].set_visible(False)

    fig.legend(
        handles=[
            mpatches.Patch(color=_TARGET_RGB, label='target class'),
            mpatches.Patch(color=_OTHER_RGB, label='other class'),
        ],
        loc='lower right', fontsize=9,
    )
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f'  Saved: {output_path}')


def draw_all_specimens(records, output_dir, n=10, seed=42):
    """Save one specimen-grid PNG per defect class found in records."""
    specimens = _sample_by_class(records, n=n, seed=seed)
    out = Path(output_dir)
    for class_id in sorted(specimens):
        class_name = CLASS_NAMES.get(class_id, str(class_id))
        fname = f'specimens_cls{class_id:02d}_{class_name}.png'
        _plot_class_grid(specimens[class_id], class_id, out / fname, n)
    print(f'Specimen grids saved to: {out}')
