"""Build a demo image set for the Demo button from a model's test split.

Copies real image files (not the dataset symlinks) into app/demo/<side>/NNNN.jpg,
preferring images that contain defects so the demo shows detections and fills the
3x3 grid. Bundle app/demo/ beside the .exe (build_exe.ps1 does this) so the Demo
button works on the deployment box where the dataset is absent.

The dataset entries are symlinks into the training data. Where they resolve
natively (the Linux CUDA container) no extra flag is needed; on Windows the
symlink targets (`/root/data/...`) are dangling, so pass `--data-root` to remap
that prefix to the real drive, e.g. `--data-root E:/cache_data`.

    # in the container:
    python -m app.scripts.make_demo
    # on Windows:
    python -m app.scripts.make_demo --data-root E:/cache_data
"""

import argparse
import json
import os
import shutil

from app.defect_app.config import MODELS

_SYMLINK_PREFIX = "/root/data"


def _resolve(path, data_root):
    if os.path.exists(path):                       # symlink resolves natively
        return path
    if data_root and os.path.islink(path):
        target = os.readlink(path)
        if target.startswith(_SYMLINK_PREFIX):
            return data_root + target[len(_SYMLINK_PREFIX):]
    return path


def make(side, count, dest_root, data_root):
    spec = MODELS[side]
    split = os.path.join(spec.dataset_dir, "test")
    coco = json.load(open(os.path.join(split, "_annotations.coco.json"), encoding="utf-8"))
    with_defect = {a["image_id"] for a in coco["annotations"]}
    imgs = [im for im in coco["images"] if im["id"] in with_defect] or coco["images"]
    imgs.sort(key=lambda im: im["id"])
    step = max(1, len(imgs) // count)              # spread the sample across the split
    sample = imgs[::step][:count]

    dest = os.path.join(dest_root, side)
    os.makedirs(dest, exist_ok=True)
    copied = 0
    for im in sample:
        src = _resolve(os.path.join(split, im["file_name"]), data_root)
        if not os.path.isfile(src):
            continue
        shutil.copyfile(src, os.path.join(dest, f"{copied:04d}.jpg"))
        copied += 1
    print(f"[{side}] {copied} images -> {dest}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS) + ["all"], default="all")
    parser.add_argument("--count", type=int, default=80, help="images per side")
    parser.add_argument("--dest-root", default="app/demo")
    parser.add_argument("--data-root", default=None,
                        help="remap the dataset symlink prefix /root/data (e.g. E:/cache_data)")
    args = parser.parse_args()

    sides = MODELS if args.model == "all" else {args.model: MODELS[args.model]}
    for side in sides:
        make(side, args.count, args.dest_root, args.data_root)


if __name__ == "__main__":
    main()
