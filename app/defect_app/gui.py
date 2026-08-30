"""Main window for the defect-detection app.

Layout: front + rear live panels (with detection overlays), a 3x3 grid of the
nine most recent defect crops, camera dropdowns, an output-directory button, and
a demo button. Rear stream is only available with a second GPU (see gpu.py).
"""

import os
import time
from collections import deque
from datetime import datetime

import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QMainWindow, QPushButton, QVBoxLayout, QWidget,
)

from . import crypto, trt_engine
from .config import KEY_PATH, MODELS, demo_dir, fp16_enc_path
from .engine import RFDetrOnnx
from .gpu import gpu_count, rear_supported
from .pipeline import InferenceWorker
from .sources import CameraSource, DemoSource, list_cameras

PANEL_W, PANEL_H = 480, 360
SAVE_COOLDOWN_S = 1.0
SIDES = ("front", "back")


def bgr_to_pixmap(frame_bgr, w=None, h=None) -> QPixmap:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    hh, ww, ch = rgb.shape
    img = QImage(rgb.data, ww, hh, ch * ww, QImage.Format_RGB888).copy()
    pix = QPixmap.fromImage(img)
    if w:
        pix = pix.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return pix


class MainWindow(QMainWindow):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.setWindowTitle("Industrial Defect Detection")
        self.threshold = threshold

        self.n_gpus = gpu_count()
        self.rear_ok = rear_supported(self.n_gpus)
        self.cameras = list_cameras()

        self.engines = self._load_engines()
        self.workers: dict = {}
        self.latest_frames: dict = {}      # side -> annotated BGR frame
        self.output_dir = None
        self.last_save = 0.0
        self.recent_defects = deque(maxlen=9)   # annotated frames of the last 9 defect hits

        self.setWindowTitle(
            f"Industrial Defect Detection — {self.backend}, {self.n_gpus} GPU")
        self._build_ui()

    # ---- setup ---------------------------------------------------------
    def _load_engines(self):
        """Prefer the FP16 TensorRT backend (builds/caches per GPU on first run);
        fall back to the onnxruntime engine when TRT or the FP16 model is absent."""
        key = crypto.load_key(KEY_PATH) if os.path.exists(KEY_PATH) else None
        engines = {}
        if key is None:
            self.backend = "none"
            return engines
        use_trt = trt_engine.available()
        self.backend = "tensorrt" if use_trt else "onnxruntime"
        for i, side in enumerate(SIDES):
            if side == "back" and not self.rear_ok:
                continue
            spec = MODELS[side]
            eng = None
            if use_trt and os.path.exists(fp16_enc_path(spec)):
                try:
                    eng = trt_engine.TrtEngine.build_or_load(
                        spec, key, class_names=spec.class_names,
                        resolution=spec.resolution, device_id=i)
                except Exception as exc:
                    print(f"[{side}] TensorRT unavailable ({exc}); using onnxruntime")
                    eng = None
            if eng is None and os.path.exists(spec.encrypted):
                eng = RFDetrOnnx.from_encrypted(
                    spec.encrypted, key, class_names=spec.class_names,
                    resolution=spec.resolution, device_id=i)
            if eng is not None:
                engines[side] = eng
        return engines

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # --- video panels + grid ---
        panels = QHBoxLayout()
        self.video_labels, self.fps_labels = {}, {}
        for side in SIDES:
            box = QGroupBox("Front camera" if side == "front" else "Rear camera")
            v = QVBoxLayout(box)
            vid = QLabel()
            vid.setFixedSize(PANEL_W, PANEL_H)
            vid.setAlignment(Qt.AlignCenter)
            vid.setStyleSheet("background:#222;color:#888;")
            if side == "back" and not self.rear_ok:
                vid.setText(f"Rear disabled\n(needs 2 GPUs; found {self.n_gpus})")
            else:
                vid.setText("no signal")
            fps = QLabel("-- fps")
            self.video_labels[side], self.fps_labels[side] = vid, fps
            v.addWidget(vid)
            v.addWidget(fps)
            panels.addWidget(box)

        grid_box = QGroupBox("Recent defects (9)")
        self.grid = QGridLayout(grid_box)
        self.grid_cells = []
        for r in range(3):
            for c in range(3):
                cell = QLabel()
                cell.setFixedSize(120, 120)
                cell.setAlignment(Qt.AlignCenter)
                cell.setStyleSheet("background:#333;border:1px solid #555;")
                self.grid.addWidget(cell, r, c)
                self.grid_cells.append(cell)
        panels.addWidget(grid_box)
        root.addLayout(panels)

        # --- controls ---
        controls = QHBoxLayout()
        self.combos = {}
        for side in SIDES:
            controls.addWidget(QLabel(f"{side.capitalize()} cam:"))
            combo = QComboBox()
            combo.addItem("None", None)
            for idx in self.cameras:
                combo.addItem(f"Camera {idx}", idx)
            combo.currentIndexChanged.connect(lambda _, s=side: self._on_camera_changed(s))
            if side == "back" and not self.rear_ok:
                combo.setEnabled(False)
            self.combos[side] = combo
            controls.addWidget(combo)

        self.dir_btn = QPushButton("Select output dir")
        self.dir_btn.clicked.connect(self._choose_output_dir)
        controls.addWidget(self.dir_btn)
        self.dir_label = QLabel("(none)")
        controls.addWidget(self.dir_label)

        self.demo_btn = QPushButton("Demo")
        self.demo_btn.clicked.connect(self._start_demo)
        controls.addWidget(self.demo_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop_all)
        controls.addWidget(self.stop_btn)

        controls.addStretch()
        root.addLayout(controls)

    # ---- stream control ------------------------------------------------
    def _start_stream(self, side, source):
        self._stop_stream(side)
        engine = self.engines.get(side)
        if engine is None:
            self.video_labels[side].setText(f"{side} model unavailable")
            source.release()
            return
        worker = InferenceWorker(side, source, engine, self.threshold)
        worker.frame_ready.connect(self._on_frame)
        worker.defect_detected.connect(self._on_defect)
        worker.fps_ready.connect(self._on_fps)
        worker.start()
        self.workers[side] = worker

    def _stop_stream(self, side):
        worker = self.workers.pop(side, None)
        if worker is not None:
            worker.stop()

    def _stop_all(self):
        for side in list(self.workers):
            self._stop_stream(side)
        for side in SIDES:
            if not (side == "back" and not self.rear_ok):
                self.video_labels[side].setText("no signal")
                self.fps_labels[side].setText("-- fps")

    def _on_camera_changed(self, side):
        idx = self.combos[side].currentData()
        if idx is None:
            self._stop_stream(side)
            self.video_labels[side].setText("no signal")
        else:
            self._start_stream(side, CameraSource(idx))

    def _start_demo(self):
        # Demo runs when no camera is selected: play sample images through the models.
        for side in SIDES:
            if side not in self.engines:
                continue
            folder = demo_dir(MODELS[side])
            if not os.path.isdir(folder):
                self.video_labels[side].setText(f"no demo images\n({folder})")
                continue
            self.combos[side].setCurrentIndex(0)  # clear camera selection
            self._start_stream(side, DemoSource(folder))

    # ---- signal handlers (GUI thread) ----------------------------------
    def _on_frame(self, side, annotated, _detections):
        self.latest_frames[side] = annotated
        self.video_labels[side].setPixmap(bgr_to_pixmap(annotated, PANEL_W, PANEL_H))

    def _on_fps(self, side, fps):
        self.fps_labels[side].setText(f"{fps:.1f} fps")

    def _on_defect(self, side, annotated_frame):
        # Grid shows the 9 most recent whole frames that had a defect, with the
        # detection overlay drawn (not a crop of the box).
        self.recent_defects.appendleft(bgr_to_pixmap(annotated_frame, 120, 120))
        for cell, pix in zip(self.grid_cells, self.recent_defects):
            cell.setPixmap(pix)
        self._save_on_defect()

    def _save_on_defect(self):
        now = time.time()
        if not self.output_dir or now - self.last_save < SAVE_COOLDOWN_S:
            return
        self.last_save = now
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        for side, frame in self.latest_frames.items():
            cv2.imwrite(os.path.join(self.output_dir, f"{ts}_{side}.jpg"), frame)

    def _choose_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select output directory")
        if path:
            self.output_dir = path
            self.dir_label.setText(path)

    def closeEvent(self, event):
        self._stop_all()
        for eng in self.engines.values():
            if hasattr(eng, "close"):
                eng.close()
        super().closeEvent(event)
