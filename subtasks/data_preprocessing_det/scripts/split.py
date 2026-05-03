"""
Generate train/val/test split JSON from the detection data list.

Split is performed on individual images (consistent with classification pipeline).
Output format per entry: {image_path: [{"class": int, "bbox": [x, y, w, h]}, ...]}
Negative images have an empty list as the value.
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from project_src.arguments import AccessArgs


SIDE_CHOICES = ('front', 'back', 'both')


def force_label_continuity(all_annotations):
    """Remap class labels across all annotation lists to continuous integers starting from 0.

    Returns (remapped_annotations_list, label_map).
    """
    all_classes = sorted({ann['class'] for anns in all_annotations for ann in anns})
    if not all_classes:
        return all_annotations, {}
    label_map = {old: new for new, old in enumerate(all_classes)}
    remapped = [
        [{"class": label_map[ann['class']], "bbox": ann['bbox']} for ann in anns]
        for anns in all_annotations
    ]
    return remapped, label_map


def split_det(data_list_path, output_path, val_ratio=0.1, seed=42, side='both'):
    if side not in SIDE_CHOICES:
        raise ValueError(f"side must be one of {SIDE_CHOICES}, got '{side}'")

    with open(data_list_path, 'r', encoding='utf-8') as f:
        records = json.load(f)

    img_paths, all_annotations = [], []
    for r in records:
        if side != 'both' and r['side'] != side:
            continue
        img_paths.append(f"{r['file_dir']}/{r['filename']}")
        all_annotations.append(r['annotations'])

    all_annotations, label_map = force_label_continuity(all_annotations)

    train_pths, test_pths, train_anns, test_anns = train_test_split(
        img_paths, all_annotations, test_size=val_ratio, random_state=seed
    )
    train_pths, val_pths, train_anns, val_anns = train_test_split(
        train_pths, train_anns, test_size=val_ratio, random_state=seed
    )

    split = {
        'train': [{k: v} for k, v in zip(train_pths, train_anns)],
        'val': [{k: v} for k, v in zip(val_pths, val_anns)],
        'test': [{k: v} for k, v in zip(test_pths, test_anns)],
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(split, f, ensure_ascii=False, indent=2)

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
    defect_classes = sorted({
        ann['class']
        for split in split_names
        for entry in data[split]
        for anns in entry.values()
        for ann in anns
    })

    # 'neg' column at position -1 represents images with no annotations.
    x_labels = ['neg'] + defect_classes
    x_positions = list(range(len(x_labels)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle('Image/annotation count per split (detection)', fontsize=14)

    for ax, name in zip(axes, split_names):
        neg_count = sum(
            1 for entry in data[name]
            for anns in entry.values()
            if len(anns) == 0
        )
        ann_counts = Counter(
            ann['class']
            for entry in data[name]
            for anns in entry.values()
            for ann in anns
        )
        values = [neg_count] + [ann_counts.get(c, 0) for c in defect_classes]
        ax.bar(x_positions, values)
        ax.set_title(f'{name} (n={len(data[name])} imgs)')
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f'Saved to: {output_path}')
    else:
        plt.show()


def main(args):
    data_list = getattr(args, 'data_list', 'data/internal_train/data_list_det/data_list_det.json')
    output = getattr(args, 'split_output', 'data/internal_train/data_split_det.json')
    val_ratio = getattr(args, 'val_ratio', 0.2)
    seed = getattr(args, 'seed', 42)
    side = getattr(args, 'side', 'both')

    print(f"Splitting: {data_list}")
    split_det(data_list, output, val_ratio, seed, side)
    plot_split_hist(output, output_path=output.replace('.json', '_hist.png'))


if __name__ == '__main__':
    args = AccessArgs().get_args()
    main(args)
