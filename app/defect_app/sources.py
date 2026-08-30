"""Frame sources: physical cameras and the demo (val-image) playback."""

import glob
import os

import cv2


def list_cameras(max_index: int = 5) -> list:
    """Return indices of cameras that open successfully."""
    found = []
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            found.append(i)
        cap.release()
    return found


class CameraSource:
    def __init__(self, index: int):
        self.cap = cv2.VideoCapture(index)

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self):
        self.cap.release()


class DemoSource:
    """Loops over a folder of images as a fake video (for when no camera is set).

    Defaults to a model's val/test split; on a deployed .exe point it at a
    bundled sample folder instead.
    """

    _EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")

    def __init__(self, image_dir: str, limit: int = 500):
        paths = []
        for ext in self._EXTS:
            paths.extend(glob.glob(os.path.join(image_dir, ext)))
        self.paths = sorted(paths)[:limit]
        self.i = 0

    def read(self):
        if not self.paths:
            return None
        frame = cv2.imread(self.paths[self.i % len(self.paths)])
        self.i += 1
        return frame

    def release(self):
        pass
