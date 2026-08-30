# PyInstaller spec for the defect-detection GUI.
#
# Build ON the target Windows machine (a .exe can't be cross-built from Linux),
# from the repository root, in a venv that has app/requirements.txt + pyinstaller:
#
#     pyinstaller app/packaging/app.spec
#
# Produces dist/DefectDetection/DefectDetection.exe (one-folder build — required
# for the large onnxruntime / TensorRT / CUDA native DLLs; one-file would unpack
# hundreds of MB to temp on every launch and complicate DLL loading).
#
# Models are NOT bundled: ship a `models/` folder beside the .exe (see README).

import os

from PyInstaller.utils.hooks import collect_all, collect_submodules

# SPECPATH is app/packaging (injected by PyInstaller); the repo root is two up.
# Anchor everything to it so paths don't depend on the current directory.
repo_root = os.path.abspath(os.path.join(SPECPATH, "..", ".."))

datas, binaries, hiddenimports = [], [], []

# Heavy native packages: pull in their DLLs, data, and submodules. Wrapped so a
# package that isn't installed (e.g. tensorrt on a CPU-only box) doesn't abort the
# build — the app falls back to onnxruntime at runtime when TRT is absent.
for pkg in ("onnxruntime", "PySide6", "cv2", "PIL",
            "tensorrt", "tensorrt_libs", "tensorrt_bindings",
            "cuda", "cuda_bindings", "cuda_pathfinder"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as exc:  # noqa: BLE001
        print(f"[app.spec] skipping {pkg}: {exc}")

hiddenimports += collect_submodules("app.defect_app")

a = Analysis(
    [os.path.join(repo_root, "app", "main.py")],
    pathex=[repo_root],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["torch", "rfdetr", "onnx", "onnxsim", "matplotlib", "pandas"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DefectDetection",
    debug=False,
    strip=False,
    upx=False,
    console=False,          # set True temporarily to see tracebacks while debugging the build
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="DefectDetection",
)
