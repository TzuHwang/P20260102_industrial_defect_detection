"""Convert a detection split JSON into RF-DETR's expected COCO directory layout.

Source format (written by split.py):
    {"train"/"val"/"test": [{image_path: [{"class": int, "bbox": [x, y, w, h]}, ...]}, ...]}
Class ids in the split JSON are already remapped to a continuous 0-based range
(see data_split_det_front_label_map.json for the raw → remapped mapping).

Output layout (RF-DETR / Roboflow COCO format):
    <out_dir>/
        train/_annotations.coco.json + symlinked images
        valid/_annotations.coco.json + symlinked images   (from the "val" split)
        test/_annotations.coco.json  + symlinked images

Images are symlinked (not copied) under generated names "{index:06d}{ext}"
to avoid duplicating ~18 GB of source data and to sidestep the deeply-nested,
non-ASCII source paths.

Usage:
    python -m subtasks.data_preprocessing_det.scripts.convert_to_coco \
        --data-split data/internal_train/data_list_det/data_split_det_front.json \
        --out-dir data/internal_train/rfdetr_coco_front
"""

import argparse
import json
import os
from pathlib import Path

from PIL import Image

from subtasks.data_preprocessing_det.scripts.draw_specimens import CLASS_NAMES

# split JSON key → (output directory name, COCO-convention name)
SPLIT_DIRS = {'train': 'train', 'val': 'valid', 'test': 'test'}


def _build_categories(label_map_path):
    """Return COCO categories [{id, name}], indexed by remapped class id + 1.

    label_map.json maps raw class id (string, 1-based) → remapped class id
    (0-based). CLASS_NAMES maps the same raw class ids → display names.
    """
    label_map = json.load(open(label_map_path, encoding='utf-8'))
    remapped_to_name = {remapped: CLASS_NAMES[int(raw)] for raw, remapped in label_map.items()}
    num_classes = len(remapped_to_name)
    return [{'id': i + 1, 'name': remapped_to_name[i]} for i in range(num_classes)]


def _convert_split(entries, split_dir):
    """Symlink images into split_dir and build COCO images/annotations lists."""
    split_dir.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    ann_id = 1

    for img_id, entry in enumerate(entries, start=1):
        src_path, anns = next(iter(entry.items()))
        ext = Path(src_path).suffix.lower()
        file_name = f'{img_id:06d}{ext}'
        link_path = split_dir / file_name
        if not link_path.exists():
            os.symlink(src_path, link_path)

        width, height = Image.open(src_path).size
        images.append({'id': img_id, 'file_name': file_name, 'width': width, 'height': height})

        for ann in anns:
            x, y, w, h = ann['bbox']
            annotations.append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': ann['class'] + 1,   # COCO category ids are 1-based
                'bbox': [x, y, w, h],
                'area': w * h,
                'iscrowd': 0,
            })
            ann_id += 1

    return images, annotations


def convert(data_split_path, label_map_path, out_dir):
    data_split = json.load(open(data_split_path, encoding='utf-8'))
    categories = _build_categories(label_map_path)
    out_dir = Path(out_dir)

    for split_key, dir_name in SPLIT_DIRS.items():
        split_dir = out_dir / dir_name
        images, annotations = _convert_split(data_split[split_key], split_dir)
        coco = {'images': images, 'annotations': annotations, 'categories': categories}
        with open(split_dir / '_annotations.coco.json', 'w', encoding='utf-8') as f:
            json.dump(coco, f, ensure_ascii=False)
        print(f'{split_key} -> {dir_name}: {len(images)} images, {len(annotations)} annotations')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-split', required=True)
    parser.add_argument('--label-map', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    convert(args.data_split, args.label_map, args.out_dir)
