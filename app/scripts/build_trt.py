"""Install-time step: build the FP16 TensorRT engine on THIS machine's GPU and
cache it (AES-encrypted) next to the model.

Run once per target machine, after the encrypted FP16 ONNX (model_fp16.enc) is in
place. Decrypts the ONNX in memory, builds the engine on the given GPU, and writes
the encrypted engine cache (model.trt). The app also does this lazily on first run
(TrtEngine.build_or_load); this script just lets you pre-build it.

IMPORTANT: engines are GPU-arch / TRT-version specific — build on the deployment
3070s (one per GPU), not on the dev card. Needs `tensorrt` + `cuda-python`.

    python -m app.scripts.build_trt --model front --device-id 0
    python -m app.scripts.build_trt --model back  --device-id 1
"""

import argparse

from app.defect_app import crypto
from app.defect_app.config import KEY_PATH, MODELS
from app.defect_app.trt_engine import precache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS) + ["all"], default="all")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--workspace-gb", type=int, default=4)
    args = parser.parse_args()

    key = crypto.load_key(KEY_PATH)
    specs = MODELS.values() if args.model == "all" else [MODELS[args.model]]
    for spec in specs:
        cache = precache(spec, key, device_id=args.device_id, workspace_gb=args.workspace_gb)
        print(f"[{spec.name}] cached encrypted engine -> {cache}")


if __name__ == "__main__":
    main()
