from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from project_src.dataloader.transform.augmentation.resize_and_crop import Resize


@pytest.fixture
def resize_args():
    return SimpleNamespace(target_size=200, interpolation='LINEAR', p=1.0)


@pytest.fixture
def image_subject():
    """RGB image (H=300, W=400, C=3) with no boxes."""
    return {
        'inputs': np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8),
        'targets': {'label': 0},
    }


@pytest.fixture
def detection_subject():
    """RGB image (H=300, W=400) with two boxes in xyxy format."""
    return {
        'inputs': np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8),
        'targets': {
            'boxes': [[40, 30, 200, 150], [250, 100, 380, 280]],
            'labels': [1, 2],
        },
    }


# ---------------------------------------------------------------------------
# Image-only resize
# ---------------------------------------------------------------------------

def test_resize_output_shape(resize_args, image_subject):
    aug = Resize(resize_args)
    out = aug(deepcopy(image_subject))
    assert out['inputs'].shape == (200, 200, 3)


def test_resize_preserves_targets(resize_args, image_subject):
    aug = Resize(resize_args)
    out = aug(deepcopy(image_subject))
    assert out['targets']['label'] == image_subject['targets']['label']


# ---------------------------------------------------------------------------
# Resize with bounding boxes
# ---------------------------------------------------------------------------

def test_resize_scales_boxes(resize_args, detection_subject):
    """Boxes must be scaled by the same ratio as the image dimensions."""
    aug = Resize(resize_args)
    out = aug(deepcopy(detection_subject))

    orig_h, orig_w = 300, 400
    new_h, new_w = 200, 200
    sx = new_w / orig_w   # 0.5
    sy = new_h / orig_h   # 0.667

    for orig_box, scaled_box in zip(detection_subject['targets']['boxes'],
                                    out['targets']['boxes']):
        x1, y1, x2, y2 = orig_box
        ex1 = pytest.approx(x1 * sx, abs=1.5)
        ey1 = pytest.approx(y1 * sy, abs=1.5)
        ex2 = pytest.approx(x2 * sx, abs=1.5)
        ey2 = pytest.approx(y2 * sy, abs=1.5)
        assert scaled_box[0] == ex1
        assert scaled_box[1] == ey1
        assert scaled_box[2] == ex2
        assert scaled_box[3] == ey2


def test_resize_preserves_box_count(resize_args, detection_subject):
    aug = Resize(resize_args)
    out = aug(deepcopy(detection_subject))
    assert len(out['targets']['boxes']) == len(detection_subject['targets']['boxes'])


def test_resize_preserves_labels(resize_args, detection_subject):
    aug = Resize(resize_args)
    out = aug(deepcopy(detection_subject))
    assert out['targets']['labels'] == detection_subject['targets']['labels']


def test_resize_image_shape_with_boxes(resize_args, detection_subject):
    aug = Resize(resize_args)
    out = aug(deepcopy(detection_subject))
    assert out['inputs'].shape == (200, 200, 3)
