"""
Generate train/val/test split JSON from the data list CSV.

Split is performed at the directory level (file_dir) to prevent data leakage
between images from the same source folder.
"""

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from project_src.arguments import AccessArgs

SIDE_CHOICES = ('front', 'back', 'both')


def force_label_continuity(labels):
    """Remap labels to ensure they are continuous integers starting from 0."""
    unique_labels = sorted(set(labels))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    return [label_map[label] for label in labels], label_map


def split_by_dir(csv_path, output_path, val_ratio=0.1,
                 seed=42, side='both'):
    if side not in SIDE_CHOICES:
        raise ValueError(f"side must be one of {SIDE_CHOICES}, got '{side}'")

    with open(csv_path, 'r', encoding='utf-8') as f:
        records = list(csv.DictReader(f))

    src_pths, labels = [], []
    for r in records:
        if side != 'both' and r['side'] != side:
            continue
        dir, name = r['file_dir'], r['filename']
        src_pths.append(f'{dir}/{name}')
        labels.append(int(r['class']))

    labels, label_map = force_label_continuity(labels)

    train_pths, test_pths, train_labels, test_labels = train_test_split(
        src_pths, labels, test_size=val_ratio, random_state=seed
    )
    train_pths, val_pths, train_labels, val_labels = train_test_split(
        train_pths, train_labels,
        test_size=val_ratio,
        random_state=seed,
    )

    split = {
        'train': [{k: v} for k, v in zip(train_pths, train_labels)],
        'val': [{k: v} for k, v in zip(val_pths, val_labels)],
        'test': [{k: v} for k, v in zip(test_pths, test_labels)],
    }

    out_path = Path(output_path)

    # save split JSON
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(split, f, ensure_ascii=False, indent=2)

    # save label mapping for reference
    label_map_path = out_path.parent / f'{out_path.stem}_label_map.json'
    with open(label_map_path, 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    total = sum(len(v) for v in split.values())
    print(f"Side filter: {side}")
    for name, data in split.items():
        print(f"  {name}: {len(data)} images ({len(data) / total * 100:.1f}%)")
    print(f"Saved to: {out_path}")


def plot_split_hist(split_path, output_path=None):
    with open(split_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    split_names = ['train', 'val', 'test']
    all_classes = sorted({cls for split in split_names for d in data[split] for cls in d.values()})

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle('Class distribution per split', fontsize=14)

    for ax, name in zip(axes, split_names):
        counts = Counter(cls for d in data[name] for cls in d.values())
        values = [counts.get(c, 0) for c in all_classes]
        ax.bar(all_classes, values)
        ax.set_title(f'{name} (n={len(data[name])})')
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_xticks(all_classes)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f'Saved to: {output_path}')
    else:
        plt.show()


def main(args):
    csv_path = getattr(args, 'data_list', 'data/internal_train/data_list/data_list.csv')
    output = getattr(args, 'split_output', 'data/internal_train/data_split.json')
    val_ratio = getattr(args, 'val_ratio', 0.2)
    seed = getattr(args, 'seed', 42)
    side = getattr(args, 'side', 'both')

    print(f"Splitting: {csv_path}")
    split_by_dir(csv_path, output, val_ratio, seed, side)
    plot_split_hist(output, output_path=output.replace('.json', '_hist.png'))


if __name__ == '__main__':
    args = AccessArgs().get_args()
    main(args)
