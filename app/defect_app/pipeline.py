"""Background inference worker: one per camera stream.

Pulls frames from a source, runs the ONNX engine, draws overlays, and emits
results to the GUI thread via Qt signals so the UI stays responsive.
"""

import time

from PySide6.QtCore import QThread, Signal

from .draw import draw_detections


class InferenceWorker(QThread):
    # side, annotated BGR frame, list[Detection]
    frame_ready = Signal(str, object, object)
    # side, annotated BGR frame  (emitted only when a defect is found)
    defect_detected = Signal(str, object)
    fps_ready = Signal(str, float)

    def __init__(self, side, source, engine, threshold=0.5, parent=None):
        super().__init__(parent)
        self.side = side
        self.source = source
        self.engine = engine
        self.threshold = threshold
        self._running = True

    def run(self):
        n, t0 = 0, time.time()
        while self._running:
            frame = self.source.read()
            if frame is None:
                self.msleep(5)
                continue

            detections = self.engine.predict(frame, threshold=self.threshold)
            annotated = draw_detections(frame, detections)
            self.frame_ready.emit(self.side, annotated, detections)
            if detections:
                self.defect_detected.emit(self.side, annotated)

            n += 1
            if n >= 30:
                now = time.time()
                self.fps_ready.emit(self.side, n / (now - t0))
                n, t0 = 0, now

        self.source.release()

    def stop(self):
        self._running = False
        self.wait(2000)
