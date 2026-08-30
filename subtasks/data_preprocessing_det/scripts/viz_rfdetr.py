"""Visualize RF-DETR predictions vs ground-truth on test-set images.

GT boxes are drawn in green; predicted boxes in red with confidence score.
Images with no GT and no predictions are skipped by default.

Usage:
    python -m subtasks.data_preprocessing_det.scripts.viz_rfdetr \
        --checkpoint outputs/rfdetr_medium_front/checkpoint_best_regular.pth \
        --dataset-dir data/internal_train/rfdetr_coco_front \
        --output-dir outputs/rfdetr_viz \
        --num-images 100 \
        --score-threshold 0.3
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from rfdetr import RFDETR

GT_COLOR = (0, 200, 0)       # green
PRED_COLOR = (220, 50, 50)   # red
BOX_WIDTH = 2
FONT_SIZE = 14
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"  # needed for Chinese class names


def _load_split(dataset_dir, split):
    split_dir = Path(dataset_dir) / split
    coco = json.load(open(split_dir / '_annotations.coco.json', encoding='utf-8'))
    categories = {c['id']: c['name'] for c in coco['categories']}
    num_classes = len(categories)

    anns_by_image: dict[int, list] = {}
    for ann in coco['annotations']:
        x, y, w, h = ann['bbox']
        anns_by_image.setdefault(ann['image_id'], []).append(
            (x, y, x + w, y + h, ann['category_id'] - 1)   # xyxy, 0-based class
        )

    records = []
    for img in sorted(coco['images'], key=lambda im: im['id']):
        records.append({
            'path': str(split_dir / img['file_name']),
            'anns': anns_by_image.get(img['id'], []),
        })

    class_names = [categories[i + 1] for i in range(num_classes)]
    return records, class_names


def _batched(seq, batch_size):
    for i in range(0, len(seq), batch_size):
        yield seq[i:i + batch_size]


def _draw_box(draw, xyxy, label, color, slot=0):
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=BOX_WIDTH)
    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except OSError:
        font = ImageFont.load_default()
    line_h = FONT_SIZE + 2
    label_y = y1 - (slot + 1) * line_h
    if label_y < 0:
        label_y = y1 + slot * line_h  # box touches the top edge: stack labels inside instead
    bbox = draw.textbbox((x1, label_y), label, font=font)
    draw.rectangle(bbox, fill=color)
    draw.text((bbox[0], bbox[1]), label, fill=(255, 255, 255), font=font)


def visualize(checkpoint, dataset_dir, split, output_dir, num_images, score_threshold, seed, skip_empty, batch_size):
    records, class_names = _load_split(dataset_dir, split)
    num_classes = len(class_names)

    candidates = [r for r in records if r['anns']] if skip_empty else records
    if num_images is not None:
        random.seed(seed)
        candidates = random.sample(candidates, min(num_images, len(candidates)))

    model = RFDETR.from_checkpoint(checkpoint)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    indexed = list(enumerate(candidates))
    done = 0
    for batch in _batched(indexed, batch_size):
        pending = [(i, meta) for i, meta in batch
                   if not (out_dir / f'{i:05d}_{Path(meta["path"]).stem}.jpg').exists()]
        if pending:
            preds = model.predict([meta['path'] for _, meta in pending], threshold=score_threshold)
            if not isinstance(preds, list):
                preds = [preds]

            for (i, meta), det in zip(pending, preds):
                img = Image.open(meta['path']).convert('RGB')
                draw = ImageDraw.Draw(img)

                for (x1, y1, x2, y2, cls_id) in meta['anns']:
                    _draw_box(draw, (x1, y1, x2, y2), class_names[cls_id], GT_COLOR, slot=0)

                keep = det.class_id < num_classes  # drop the "no-object" background slot
                for xyxy, conf, cls_id in zip(det.xyxy[keep], det.confidence[keep], det.class_id[keep]):
                    label = f'{class_names[int(cls_id)]} {conf:.2f}'
                    _draw_box(draw, xyxy, label, PRED_COLOR, slot=1)

                stem = Path(meta['path']).stem
                out_path = out_dir / f'{i:05d}_{stem}.jpg'
                img.save(out_path, quality=92)

        done += len(batch)
        print(f'[{done}/{len(candidates)}]')

    print(f'\nDone. {len(candidates)} images → {out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset-dir', required=True)
    parser.add_argument('--split', default='test', choices=['train', 'valid', 'test'])
    parser.add_argument('--output-dir', default='outputs/rfdetr_viz')
    parser.add_argument('--num-images', type=int, default=None,
                        help='limit to a random sample of N images; default: all')
    parser.add_argument('--score-threshold', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--include-empty', action='store_true',
                        help='also include images with no GT annotations')
    args = parser.parse_args()
    visualize(args.checkpoint, args.dataset_dir, args.split, args.output_dir,
              args.num_images, args.score_threshold, args.seed, not args.include_empty,
              args.batch_size)
