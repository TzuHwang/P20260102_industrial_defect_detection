# Build the DefectDetection .exe on Windows (PowerShell).
#
# Uses `uv` to create a fully isolated build venv (uv fetches a standalone Python
# into its own cache — nothing is installed system-wide, no PATH/registry changes).
# Everything lives under app/packaging/ (git-ignored); delete .venv_build/, dist/,
# build/ to clean up completely.
#
# Prerequisites: `uv` on PATH, an NVIDIA driver/CUDA compatible with
# onnxruntime-gpu / TensorRT. Run from the repository root:
#   .\app\packaging\build_exe.ps1
#
# TensorRT engines are GPU-arch specific and built on first run, so ideally build
# on a machine whose GPU matches deployment (Ampere / RTX 3070).

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path "$PSScriptRoot\..\..").Path
$pkg  = $PSScriptRoot
Set-Location $repo   # run from repo root so the `app` package imports

$venv = Join-Path $pkg ".venv_build"
$py   = Join-Path $venv "Scripts\python.exe"

# --system-certs uses the OS trust store (needed behind corporate SSL inspection);
# link-mode=copy avoids hardlink warnings when uv's cache and the venv are on
# different drives.
$env:UV_LINK_MODE = "copy"
uv venv $venv --python 3.11
uv pip install --system-certs --python $py -r app\requirements.txt pyinstaller

# Keep all build output under app/packaging (git-ignored), not the repo root.
& "$venv\Scripts\pyinstaller.exe" app\packaging\app.spec --noconfirm `
    --distpath app\packaging\dist --workpath app\packaging\build

# Bundle the demo image set (from make_demo.py) beside the exe, if present.
$distDir = "app\packaging\dist\DefectDetection"
if (Test-Path "app\demo") {
    Copy-Item "app\demo" (Join-Path $distDir "demo") -Recurse -Force
    Write-Host "Bundled app\demo -> $distDir\demo"
}

Write-Host ""
Write-Host "Built app\packaging\dist\DefectDetection\DefectDetection.exe" -ForegroundColor Green
Write-Host "Next: put a models\ folder next to the exe with:" -ForegroundColor Yellow
Write-Host "  models\model.key"
Write-Host "  models\front\model_fp16.enc"
Write-Host "  models\back\model_fp16.enc"
Write-Host "and (optional, for the Demo button) demo\front\*.jpg, demo\back\*.jpg"
Write-Host "First launch builds+caches the TRT engine per GPU (~40s each)."
