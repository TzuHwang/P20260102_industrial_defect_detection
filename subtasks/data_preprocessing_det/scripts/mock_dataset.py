"""Create a small synthetic detection dataset for pipeline verification.

Generates 256×256 images with coloured rectangles representing defects, then
writes a detection split JSON that mirrors the format produced by split.py.

Output layout (all paths relative to project root):
    data/test/mock_det/
        images/img_NNNN.jpg     — synthetic images
        mock_det_split.json     — train / val / test split

Usage:
    python -m subtasks.data_preprocessing_det.scripts.mock_dataset
"""

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUT_DIR = Path('data/test/mock_det')
IMG_DIR = OUT_DIR / 'images'
SPLIT_JSON = OUT_DIR / 'mock_det_split.json'

IMG_SIZE = 256          # square image side
NUM_CLASSES = 11        # 0..10, matching remapped real-data classes
TRAIN_PER_CLASS = 3     # positive train images per class
VAL_PER_CLASS = 1       # positive val images per class
TEST_PER_CLASS = 1      # positive test images per class
NEG_TRAIN = 4           # negative (no-defect) train images
NEG_VAL = 2
NEG_TEST = 2

SEED = 42

# One distinct RGB colour per class (vivid, easy to distinguish visually)
CLASS_COLORS = [
    (220, 50, 50),    # 0  表面脏污   red
    (50, 150, 220),   # 1  波浪       blue
    (50, 200, 50),    # 2  残缺       green
    (220, 150, 50),   # 3  表面损伤   orange
    (150, 50, 220),   # 4  烤焦起泡   purple
    (50, 200, 200),   # 5  烤漆色差   cyan
    (220, 220, 50),   # 6  空白接头   yellow
    (200, 100, 50),   # 7  黑块       brown
    (50, 50, 200),    # 8  前工序接头 dark blue
    (200, 50, 200),   # 9  印刷重     magenta
    (100, 200, 100),  # 10 上光起泡   light green
]

# ---------------------------------------------------------------------------
# Image generation helpers
# ---------------------------------------------------------------------------


def _noisy_background(rng) -> np.ndarray:
    """Gray noise background (uint8 HWC)."""
    base = int(rng.integers(100, 160))
    noise = rng.integers(-30, 30, (IMG_SIZE, IMG_SIZE, 3))
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def _random_box(rng, margin=20, min_side=30, max_side=80):
    """Return a random [x, y, w, h] box that fits within the image."""
    w = int(rng.integers(min_side, max_side))
    h = int(rng.integers(min_side, max_side))
    x = int(rng.integers(margin, IMG_SIZE - w - margin))
    y = int(rng.integers(margin, IMG_SIZE - h - margin))
    return [x, y, w, h]


def _draw_defect(img_arr: np.ndarray, box, class_id: int) -> None:
    """Draw a coloured rectangle for one defect (in-place)."""
    x, y, w, h = box
    color = CLASS_COLORS[class_id]
    # filled rectangle
    img_arr[y:y + h, x:x + w] = color
    # darker border
    border = 3
    border_color = tuple(max(0, c - 60) for c in color)
    img_arr[y:y + border, x:x + w] = border_color
    img_arr[y + h - border:y + h, x:x + w] = border_color
    img_arr[y:y + h, x:x + border] = border_color
    img_arr[y:y + h, x + w - border:x + w] = border_color


def _make_positive_image(rng, class_id: int, num_boxes: int = 2):
    """Create one image with `num_boxes` defect boxes of the given class."""
    img_arr = _noisy_background(rng)
    annotations = []
    for _ in range(num_boxes):
        box = _random_box(rng)
        _draw_defect(img_arr, box, class_id)
        annotations.append({'class': class_id, 'bbox': box})
    return img_arr, annotations


def _make_negative_image(rng):
    """Create one negative image (no defects)."""
    return _noisy_background(rng), []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_mock_dataset(seed: int = SEED):
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    IMG_DIR.mkdir(parents=True, exist_ok=True)

    img_counter = 0
    split = {'train': [], 'val': [], 'test': []}

    def _save(arr, annotations, split_name):
        nonlocal img_counter
        fname = f'img_{img_counter:04d}.jpg'
        img_counter += 1
        path = IMG_DIR / fname
        Image.fromarray(arr).save(str(path))
        split[split_name].append({str(path.resolve()): annotations})

    # --- positive images: one group per class ---
    for cls in range(NUM_CLASSES):
        for _ in range(TRAIN_PER_CLASS):
            arr, anns = _make_positive_image(rng, cls, num_boxes=py_rng.randint(1, 3))
            _save(arr, anns, 'train')
        for _ in range(VAL_PER_CLASS):
            arr, anns = _make_positive_image(rng, cls, num_boxes=1)
            _save(arr, anns, 'val')
        for _ in range(TEST_PER_CLASS):
            arr, anns = _make_positive_image(rng, cls, num_boxes=1)
            _save(arr, anns, 'test')

    # --- negative images ---
    for _ in range(NEG_TRAIN):
        _save(*_make_negative_image(rng), 'train')
    for _ in range(NEG_VAL):
        _save(*_make_negative_image(rng), 'val')
    for _ in range(NEG_TEST):
        _save(*_make_negative_image(rng), 'test')

    with open(SPLIT_JSON, 'w', encoding='utf-8') as f:
        json.dump(split, f, ensure_ascii=False, indent=2)

    print(f'Mock dataset written to: {OUT_DIR.resolve()}')
    print(f'  train : {len(split["train"])} images')
    print(f'  val   : {len(split["val"])} images')
    print(f'  test  : {len(split["test"])} images')
    print(f'  split : {SPLIT_JSON.resolve()}')
    return SPLIT_JSON


if __name__ == '__main__':
    build_mock_dataset()
