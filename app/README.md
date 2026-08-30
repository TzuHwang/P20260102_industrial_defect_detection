# Industrial defect detection app

Package the front and back models into an application that provides predictions for each frame captured by the cameras.

## Pipeline

Camera --> app --> model --> gui

## GUI

The GUI must include the following components:

1. Two display panels for showing the live video streams from the front and rear cameras. Each stream should overlay the detected bounding boxes, confidence scores, and predicted classes.
2. One display panel showing the nine most recently detected product defects in a 3 × 3 grid.
3. Two drop-down menus for selecting the front and rear cameras. Both menus may select the same camera if desired.
4. One button for selecting the output directory. Whenever a product defect is detected, the two most recent images shall be saved to the selected directory using the button timestamp as the filename.
5. A demo buttom, display two mock videos composed by front and back val data if there is no video recorde by camera(camera not select)

## requirements

1. The fps must > 90
2. using python to build the app
3. dev in a docker container
4. Trun models to onnx and encryt it
5. I would like a exe file

## model
help me fill the model paths for the models corresponding the logs

"""front val
outputs\rfdetr_eval.log   (11 classes, images=9637, mAP=0.7866, macro AUROC=0.9997)
"""
"""front model
outputs\rfdetr_medium_front\checkpoint_best_total.pth
dataset: data\internal_train\rfdetr_coco_front  (11 classes)
"""
"""back val
outputs\rfdetr_eval_back_v3.log   (7 classes, images=2715, mAP=0.8200, macro AUROC=0.9697)
"""
"""back model
outputs\rfdetr_medium_back_v3\checkpoint_best_total.pth
dataset: data\internal_train\rfdetr_coco_back  (7 classes)
"""

---

## Build status

Delivered in stages. **Done: (1) model → ONNX → AES-encrypted, (2) PySide6 GUI,
(3) FP16/TensorRT speed path wired into the app, (4) .exe packaging scaffolding.**

### 4. .exe packaging (built + smoke-tested on Windows / RTX 5070 Ti)

A Windows `.exe` cannot be cross-built from the Linux/CUDA dev container, so packaging
is **built on the target Windows machine**. Verified there: the build succeeds, the
frozen exe launches, loads all native DLLs (onnxruntime-gpu / TensorRT / cuda-python /
PySide6 / cv2), reads `models/` beside the exe, and builds the encrypted TensorRT engine
to `%LOCALAPPDATA%\DefectDetection\cache\` (64 MB) — the full decrypt→build→encrypt path.
One-folder `dist/` is ~3.35 GB (mostly TensorRT/CUDA DLLs).

Build notes for this environment: used `uv` with an isolated Python 3.11 (the model
artifacts are Python-version-independent); the corporate network does SSL inspection so
uv needs `--system-certs` (already in the build script). A one-time spec fix anchors
paths to the repo root via `SPECPATH` (PyInstaller resolves script paths relative to the
.spec file, not the CWD).

What's committed:

- `app/packaging/app.spec` — PyInstaller spec (one-folder build; `collect_all` for
  onnxruntime / PySide6 / cv2 / PIL / TensorRT / cuda-python; excludes dev-only
  torch/rfdetr/onnx). `console=False` — flip to `True` to see tracebacks while
  debugging the build.
- `app/packaging/build_exe.ps1` — makes a build venv, installs
  `app/requirements.txt` + pyinstaller, and runs the spec. Output stays under
  `app/packaging/{dist,build}` (git-ignored).

```powershell
# on the target Windows machine, from the repo root:
.\app\packaging\build_exe.ps1
# -> app\packaging\dist\DefectDetection\DefectDetection.exe
```

Packaging design (already handled in `config.py`):
- **Paths resolve for source and frozen**: models are read from `<exe_dir>/models/`
  when frozen (`app/models/` from source). Ship a `models/` folder beside the exe:
  `models/model.key`, `models/{front,back}/model_fp16.enc`.
- **Writable cache**: the per-machine encrypted `model.trt` is written to
  `%LOCALAPPDATA%\DefectDetection\cache\` when frozen, so the app works even if
  installed under Program Files (read-only). Built lazily on first run (~40 s/GPU).
- **Demo**: `config.demo_dir` prefers a bundled `demo/<side>/` (beside the exe when
  frozen, under `app/` from source), falling back to the dataset split for dev.
  `app/scripts/make_demo.py` populates `app/demo/{front,back}/` with real defect
  images from the test split (run in the container, or on Windows with
  `--data-root E:/cache_data`); `build_exe.ps1` copies `app/demo` beside the exe.
  Verified: the Demo button plays the bundled images and fills the 3×3 grid with
  real detections. `app/demo/` images are git-ignored (internal data).

Done: TensorRT/CUDA DLL collection (the exe starts and builds the engine — no missing
DLLs), a headless smoke test of the frozen exe (engine cache written, single-GPU →
front only), and the bundled demo set (make_demo.py + auto-copy in build_exe.ps1;
demo plays and fills the grid). Remaining polish: an interactive run with real cameras
on the target 3070s, trimming the 3.35 GB dist if desired, and key-management hardening
(embed/obfuscate `model.key` into the binary rather than shipping it as a file — the
"basic protection" caveat from §1 still applies).

### 3. FP16 / TensorRT (measured on a single RTX 5070 Ti, batch=1, 576×576)

| backend | model-only (mean / p95) | verdict |
|---|---|---|
| FP32 onnxruntime CUDA EP | 99 / 78 fps (full pipeline ~74 fps) | borderline — misses >90 reliably |
| FP16 ONNX + onnxruntime CUDA EP | 96 fps | **no gain** — ops without FP16 CUDA kernels force ~12 CPU↔GPU memcpys |
| **FP16 TensorRT** | **426 / 371 fps** | ✓ ~4.3× faster; clears 90 with huge margin |

FP16 TRT parity vs the original checkpoint: 100% of reference boxes matched, mean IoU 0.994.

Scripts (dev, in the CUDA container; TRT build needs `tensorrt`):

```bash
python -m app.scripts.export_onnx --fp16      # writes model.onnx (FP32) + model_fp16.onnx
python -m app.scripts.build_trt               # model_fp16.onnx -> model.trt (strongly-typed FP16)
```

- FP16 ONNX export drives rfdetr's low-level exporter with a half-cast model
  (rfdetr's `quantization=` is TFLite-only). TensorRT 10+/11 is strongly-typed,
  so precision comes from the FP16 ONNX (no FP16 builder flag).
- **TRT engines are not portable** across GPU arch / TRT version. The dev card is
  Blackwell (5070 Ti); the deployment cards are Ampere (3070). Ship the encrypted
  FP16 ONNX and **build `model.trt` on each target 3070 at install/first run**
  (`*.trt` is git-ignored).

#### Runtime backend (wired — Option A: TensorRT + cuda-python, no torch)

`trt_engine.py` provides `TrtEngine` with the same `predict()` interface as
`RFDetrOnnx`, so `gui.py` picks it automatically when `tensorrt` + `cuda-python`
are present, else falls back to onnxruntime. The window title shows which backend
is active.

Deployment / install flow (all decryption is in-memory — no plaintext on disk):

```bash
# dev (CUDA container): produce the shippable encrypted FP16 model
python -m app.scripts.export_onnx --fp16
python -m app.scripts.encrypt_model --fp16          # -> model_fp16.enc  (ship this)

# on each target 3070 (once): build + cache the encrypted engine
python -m app.scripts.build_trt --model front --device-id 0
python -m app.scripts.build_trt --model back  --device-id 1
```

`TrtEngine.build_or_load` also builds+caches lazily on first run if `model.trt`
is missing (~40 s), then loads the encrypted cache in ~2 s thereafter. Verified on
the 5070 Ti: cache load 1.8 s, 100% parity vs FP32 (IoU 0.997),
`TrtEngine.predict` full pipeline ~115 fps; GUI auto-selects the `tensorrt` backend.
Pre/post-processing is shared with the onnxruntime path (`engine.preprocess` /
`engine.postprocess`) so outputs are identical.

### 2. GUI (done)

Run: `python -m app.main` (needs the deps in `app/requirements.txt`; models exported + encrypted first).

- `gui.py` — front + rear live panels with overlays, a 3×3 grid of the nine most
  recent defect frames (whole frame with the detection overlay drawn), front/rear
  camera dropdowns, output-dir button, demo button.
- `pipeline.py` — one `InferenceWorker` (QThread) per stream: source → engine →
  overlay → Qt signals; keeps the UI responsive and reports per-stream fps.
- `sources.py` — `CameraSource` (cv2) and `DemoSource` (loops a folder of val images).
- `gpu.py` — GPU count decides capability: **≥2 GPUs → front+rear (one GPU each);
  1 GPU → front only (rear dropdown/panel disabled); 0 → CPU, front only.**
- `draw.py` — overlay via PIL (Chinese class names need a CJK font).

On a detected defect: the newest annotated frame (whole frame with boxes) is pushed
into the 3×3 grid, and if an output dir is set, the latest front+rear frames are saved as
`YYYYMMDD_HHMMSS_ffffff_{front,back}.jpg` (1 s cooldown to avoid flooding).

Verified headless (offscreen Qt) on the 1-GPU dev box: rear correctly disabled,
demo playback renders + fills the grid, and the save/cooldown works.

Interpretations made (see the GUI spec above): "two most recent images" = latest
front + latest rear frame (with overlays); "9 most recent defects" = the 9 most recent
whole frames that had a defect, with the detection overlay drawn; demo builds its two
streams from the val/test image folders (bundle a sample folder for the .exe, where the
dataset is absent).

### 1. Model → encrypted ONNX (done)

`app/defect_app/` holds the runtime pieces:
- `config.py` — front/back model registry (paths, class names, resolution 576, ImageNet norm).
- `crypto.py` — AES-256-GCM encrypt/decrypt (`MAGIC || nonce || ciphertext+tag`).
- `engine.py` — `RFDetrOnnx`: decrypts to memory, runs onnxruntime (CUDA→CPU
  fallback), replicates RF-DETR pre/post-processing, returns pixel-space detections.

Pipeline (run inside the CUDA Docker container; needs `rfdetr onnx onnxsim onnxruntime cryptography`):

```bash
python -m app.scripts.export_onnx                 # .pth -> app/models/{front,back}/model.onnx
python -m app.scripts.encrypt_model               # model.onnx -> model.enc (+ generates app/models/model.key)
python -m app.scripts.verify_parity --model front # decrypted-ONNX vs original RF-DETR
```

Verified: decrypted-ONNX detections match the original checkpoints (mean IoU ≈ 0.997,
100% of reference boxes recovered at threshold 0.5); a wrong key fails with `InvalidTag`.

Shipped artifacts: `app/models/{front,back}/model.enc`. The plaintext `.onnx` and
`model.key` are git-ignored.

> **Key-management caveat.** AES here is basic protection: it stops the raw ONNX
> from being read/copied off disk, but the app must hold the key to decrypt at
> runtime. When packaging the `.exe`, embed/obfuscate the key in the binary (or
> derive it from a machine binding). It deters casual copying, not a determined
> reverse-engineer — for that, hardware-bound TensorRT or a DRM scheme is needed.

### Runtime target

Windows with **two NVIDIA RTX 3070** GPUs, inference via `onnxruntime-gpu`
(`app/requirements.txt`). Strategy for fps>90: pin one camera stream per GPU
(front → `device_id=0`, back → `device_id=1`, via `RFDetrOnnx(device_id=...)`)
so the streams don't share a device, plus FP16 / TensorRT EP. On a single 3070
FP32 is too tight for 90 fps; FP16/TensorRT is what makes it feasible.
