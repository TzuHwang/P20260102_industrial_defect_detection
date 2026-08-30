"""Encrypt the exported ONNX models with AES-256-GCM.

Generates the key on first run (KEY_PATH) if it does not exist, then writes an
encrypted `.enc` next to each ONNX. Run export_onnx.py first.

`--fp16` encrypts model_fp16.onnx -> model_fp16.enc instead (the artifact shipped
for the TensorRT path). Run `export_onnx.py --fp16` first.

Usage:
    python -m app.scripts.encrypt_model            # FP32 model.onnx -> model.enc
    python -m app.scripts.encrypt_model --fp16     # FP16 model_fp16.onnx -> model_fp16.enc
    python -m app.scripts.encrypt_model --model back
"""

import argparse
import os

from app.defect_app import crypto
from app.defect_app.config import KEY_PATH, MODELS, fp16_enc_path, fp16_onnx_path


def _get_or_create_key(path):
    if os.path.exists(path):
        return crypto.load_key(path)
    key = crypto.generate_key()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    crypto.save_key(path, key)
    print(f"generated new key -> {path}")
    return key


def encrypt_one(spec, key, fp16):
    src = fp16_onnx_path(spec) if fp16 else spec.onnx
    dst = fp16_enc_path(spec) if fp16 else spec.encrypted
    with open(src, "rb") as f:
        blob = crypto.encrypt(f.read(), key)
    with open(dst, "wb") as f:
        f.write(blob)
    print(f"[{spec.name}] {src} -> {dst} ({len(blob)} bytes)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS) + ["all"], default="all")
    parser.add_argument("--fp16", action="store_true", help="encrypt model_fp16.onnx instead")
    args = parser.parse_args()

    key = _get_or_create_key(KEY_PATH)
    specs = MODELS.values() if args.model == "all" else [MODELS[args.model]]
    for spec in specs:
        encrypt_one(spec, key, args.fp16)


if __name__ == "__main__":
    main()
