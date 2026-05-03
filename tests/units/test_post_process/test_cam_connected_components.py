"""
Parity tests: CAM2BoxWrapper._connected_components / cam2box (PyTorch)
must produce the same bounding boxes as a cv2 reference implementation.
"""
import cv2
import numpy as np
import pytest
import torch
from types import SimpleNamespace

from project_src.post_process.wrapper.cam import CAM2BoxWrapper


# ---------------------------------------------------------------------------
# cv2 reference
# ---------------------------------------------------------------------------

def _cv2_cam2box(cam: np.ndarray, class_idx: int, cam_thr: float, box_thr: int) -> list:
    mask = (cam > cam_thr).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= box_thr:
            boxes.append({"bbox": [x, y, x + w, y + h], "class": class_idx})
    return boxes


# ---------------------------------------------------------------------------
# Minimal stand-in — binds cam2box / _connected_components without a real model
# ---------------------------------------------------------------------------

class _FakeWrapper:
    """
    Provides only the instance attributes that cam2box and _connected_components
    actually access, so we can call those methods without a real pytorch_grad_cam
    model.
    """
    def __init__(self, device: torch.device, cam_threshold: float = 0.5, box_threshold: int = 10):
        self.cam_threshold = cam_threshold
        self.box_threshold = box_threshold
        _param = torch.zeros(1, device=device)                # used to infer device in cam2box
        self.model = SimpleNamespace(
            model=SimpleNamespace(parameters=lambda: iter([_param]))
        )

    # Borrow the real implementations as bound methods
    _connected_components = CAM2BoxWrapper._connected_components
    cam2box = CAM2BoxWrapper.cam2box


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _cuda_functional() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        torch.zeros(1, device="cuda")
        return True
    except RuntimeError:
        return False

DEVICES = ["cpu"] + (["cuda"] if _cuda_functional() else [])


@pytest.fixture(params=DEVICES)
def wrapper(request):
    return _FakeWrapper(torch.device(request.param))


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------

def _sort(boxes: list) -> list:
    return sorted(boxes, key=lambda b: tuple(b["bbox"]))


def _assert_same(wrapper: _FakeWrapper, cam: np.ndarray, class_idx: int = 0):
    got = _sort(wrapper.cam2box(cam, class_idx))
    want = _sort(_cv2_cam2box(cam, class_idx, wrapper.cam_threshold, wrapper.box_threshold))
    assert len(got) == len(want), (
        f"box count: torch={len(got)}, cv2={len(want)}\n"
        f"  torch: {got}\n  cv2:   {want}"
    )
    for g, w in zip(got, want):
        assert g["bbox"] == w["bbox"], f"bbox mismatch: torch={g['bbox']}  cv2={w['bbox']}"
        assert g["class"] == w["class"], f"class mismatch: torch={g['class']} cv2={w['class']}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_empty_mask(wrapper):
    cam = np.zeros((64, 64), dtype=np.float32)
    _assert_same(wrapper, cam)


def test_single_square_blob(wrapper):
    cam = np.zeros((64, 64), dtype=np.float32)
    cam[10:25, 15:30] = 1.0          # 15×15 = 225 px
    _assert_same(wrapper, cam)


def test_two_separate_blobs(wrapper):
    cam = np.zeros((64, 64), dtype=np.float32)
    cam[5:15, 5:15] = 1.0            # 10×10 = 100 px
    cam[40:55, 40:55] = 1.0          # 15×15 = 225 px
    _assert_same(wrapper, cam)


def test_diagonal_is_single_component(wrapper):
    """8-connectivity: diagonal pixels are connected → one bbox."""
    cam = np.zeros((32, 32), dtype=np.float32)
    for i in range(15):
        cam[i, i] = 1.0              # 15 px diagonal
    _assert_same(wrapper, cam)


def test_blob_at_image_border(wrapper):
    cam = np.zeros((64, 64), dtype=np.float32)
    cam[0:12, 0:12] = 1.0            # top-left corner, 144 px
    _assert_same(wrapper, cam)


def test_area_filter(wrapper):
    """Blob < box_threshold must be dropped by both implementations."""
    cam = np.zeros((64, 64), dtype=np.float32)
    cam[5:10, 5:10] = 1.0            # 5×5 = 25 px  → kept  (≥ 10)
    cam[30:32, 30:31] = 1.0          # 2×1 = 2 px   → dropped (< 10)
    _assert_same(wrapper, cam)


def test_blobs_separated_by_one_pixel(wrapper):
    """A 1-px gap must keep two blobs as two separate components."""
    cam = np.zeros((64, 64), dtype=np.float32)
    cam[10:20, 5:15] = 1.0
    cam[10:20, 16:26] = 1.0          # gap at column 15
    _assert_same(wrapper, cam)


def test_class_index_passthrough(wrapper):
    """class index must be preserved regardless of value."""
    cam = np.zeros((32, 32), dtype=np.float32)
    cam[5:15, 5:15] = 1.0
    for cls in [0, 3, 7]:
        _assert_same(wrapper, cam, class_idx=cls)


def test_full_mask_is_single_component(wrapper):
    cam = np.ones((32, 32), dtype=np.float32)
    _assert_same(wrapper, cam)


def test_partial_threshold(wrapper):
    """Values ≤ cam_threshold (0.5) must be treated as background."""
    cam = np.zeros((64, 64), dtype=np.float32)
    cam[10:25, 10:25] = 0.8          # above threshold → foreground
    cam[10:25, 30:45] = 0.5          # at threshold    → background (not strictly greater)
    cam[10:25, 46:61] = 0.3          # below threshold → background
    _assert_same(wrapper, cam)


@pytest.mark.parametrize("seed", [0, 1, 2, 42, 99])
def test_random_masks(wrapper, seed):
    rng = np.random.default_rng(seed)
    cam = rng.uniform(0.0, 1.0, (64, 64)).astype(np.float32)
    _assert_same(wrapper, cam)


@pytest.mark.parametrize("seed", [0, 1, 2, 42, 99])
def test_random_masks_sparse(wrapper, seed):
    """Sparse activations — many disconnected tiny regions."""
    rng = np.random.default_rng(seed)
    cam = (rng.uniform(0.0, 1.0, (64, 64)) > 0.85).astype(np.float32)
    _assert_same(wrapper, cam)
