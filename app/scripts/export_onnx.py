"""Export the front / back RF-DETR checkpoints to ONNX.

Dev-only: run inside the CUDA Docker container where `rfdetr`, `onnx` and
`onnxsim` are installed (see app/README.md). Writes a static-batch-1, 576x576
ONNX per model.

FP32 (`model.onnx`) is the reference. `--fp16` additionally writes `model_fp16.onnx`
(weights + I/O in half); this is the input for the TensorRT engine build
(build_trt.py) which is where the real speedup lives — FP16 on the plain
onnxruntime CUDA EP gives no gain because several ops fall back to CPU.

Usage:
    python -m app.scripts.export_onnx                 # FP32, both models
    python -m app.scripts.export_onnx --fp16          # also write model_fp16.onnx
    python -m app.scripts.export_onnx --model front --fp16
"""

import argparse
import shutil
from pathlib import Path

from rfdetr import RFDETR

from app.defect_app.config import MODELS, fp16_onnx_path


def export_fp32(spec, model):
    out_path = Path(spec.onnx)
    tmp_dir = out_path.parent / "_export_tmp"
    produced = Path(model.export(output_dir=str(tmp_dir), opset_version=17))
    shutil.move(str(produced), str(out_path))
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"[{spec.name}] FP32 -> {out_path}")


def export_fp16(spec, model):
    # rfdetr has no FP16 ONNX flag (quantization= is TFLite-only), so drive its
    # low-level exporter with a half-cast model, tracing on GPU (FP16 CPU kernels
    # are incomplete).
    from copy import deepcopy

    from rfdetr.export.main import export_onnx as _export_onnx, make_infer_image

    out_path = Path(fp16_onnx_path(spec))
    inner = deepcopy(model.model.model).eval().cuda().half()
    x = make_infer_image(None, (spec.resolution, spec.resolution), 1, "cuda",
                         num_channels=model.model_config.num_channels).half()
    _export_onnx(output_dir=str(out_path.parent), model=inner,
                 input_names=["input"], input_tensors=x,
                 output_names=["dets", "labels"], dynamic_axes=None,
                 verbose=False, opset_version=17, variant_name=out_path.stem)
    print(f"[{spec.name}] FP16 -> {out_path}")


def export_one(spec, fp16):
    Path(spec.onnx).parent.mkdir(parents=True, exist_ok=True)
    model = RFDETR.from_checkpoint(spec.checkpoint)
    export_fp32(spec, model)
    if fp16:
        export_fp16(spec, model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS) + ["all"], default="all")
    parser.add_argument("--fp16", action="store_true", help="also write model_fp16.onnx")
    args = parser.parse_args()

    specs = MODELS.values() if args.model == "all" else [MODELS[args.model]]
    for spec in specs:
        export_one(spec, args.fp16)


if __name__ == "__main__":
    main()
