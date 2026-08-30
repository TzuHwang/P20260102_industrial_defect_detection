"""GPU capability detection.

Deployment rule: the rear camera stream is only enabled when there are at least
two GPUs (front -> device 0, rear -> device 1). With a single GPU the app runs
front-only; with no GPU it falls back to CPU (front-only, for demo/dev).
"""

import shutil
import subprocess

import onnxruntime as ort


def gpu_count() -> int:
    exe = shutil.which("nvidia-smi")
    if exe:
        try:
            out = subprocess.run([exe, "-L"], capture_output=True, text=True, timeout=5)
            n = sum(1 for line in out.stdout.splitlines() if line.startswith("GPU "))
            if n:
                return n
        except Exception:
            pass
    # nvidia-smi missing: onnxruntime only tells us whether CUDA works, not how many.
    if "CUDAExecutionProvider" in ort.get_available_providers():
        return 1
    return 0


def rear_supported(n_gpus: int) -> bool:
    """Rear camera needs a dedicated second GPU."""
    return n_gpus >= 2
