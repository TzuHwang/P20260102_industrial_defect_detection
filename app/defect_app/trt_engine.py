"""TensorRT FP16 inference backend (Option A: TensorRT + cuda-python, no torch).

Deployment flow:
  ship  : encrypted FP16 ONNX (model_fp16.enc) — portable across machines
  install: on each target GPU, decrypt in memory -> build TRT engine -> cache it
           AES-encrypted (model.trt); the plaintext ONNX/engine never touch disk
  run   : load + decrypt the cached engine -> cuda-python device buffers -> infer

TensorRT engines are GPU-arch / TRT-version specific, so the cache is built
per-machine and never shipped. Pre/post-processing is shared with the onnxruntime
path (engine.preprocess / engine.postprocess) so results are identical.

`available()` reports whether tensorrt + cuda-python are importable; when they are
not, the app falls back to the onnxruntime `RFDetrOnnx` engine.
"""

import os

import numpy as np

from . import crypto
from .config import fp16_enc_path, trt_cache_path
from .engine import postprocess, preprocess


def available() -> bool:
    try:
        import tensorrt  # noqa: F401
        from cuda.bindings import runtime  # noqa: F401
        return True
    except Exception:
        return False


def build_engine(onnx_bytes: bytes, workspace_gb: int = 4) -> bytes:
    """Parse an FP16 ONNX (strongly-typed) and return a serialized TRT engine."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_bytes):
        errs = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed:\n{errs}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    return bytes(serialized)


def precache(spec, key: bytes, device_id: int = 0, workspace_gb: int = 4) -> str:
    """Build the TRT engine from the encrypted FP16 ONNX and write the encrypted
    cache. Returns the cache path. Run once per machine (install / first run)."""
    import tensorrt as trt  # noqa: F401  (fail early if TRT missing)
    from cuda.bindings import runtime as cudart

    _ck(cudart.cudaSetDevice(device_id))
    with open(fp16_enc_path(spec), "rb") as f:
        onnx_bytes = crypto.decrypt(f.read(), key)
    engine_bytes = build_engine(onnx_bytes, workspace_gb)
    cache = trt_cache_path(spec)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "wb") as f:
        f.write(crypto.encrypt(engine_bytes, key))
    return cache


def _ck(ret):
    """Unwrap a cuda-python (err, *values) return, raising on error."""
    err = ret[0] if isinstance(ret, tuple) else ret
    values = ret[1:] if isinstance(ret, tuple) else ()
    if int(err) != 0:
        raise RuntimeError(f"CUDA runtime error {int(err)}")
    return values[0] if len(values) == 1 else values


class TrtEngine:
    def __init__(self, engine_bytes, class_names, resolution=576, num_select=300, device_id=0):
        import tensorrt as trt
        from cuda.bindings import runtime as cudart

        self._cudart = cudart
        self.device_id = device_id
        self.class_names = list(class_names)
        self.resolution = resolution
        self.num_select = num_select

        _ck(cudart.cudaSetDevice(device_id))
        engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(engine_bytes)
        if engine is None:
            raise RuntimeError("failed to deserialize TensorRT engine")
        self.engine = engine
        self.ctx = engine.create_execution_context()
        self.stream = _ck(cudart.cudaStreamCreate())

        self.host, self.dev = {}, {}
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            shape = tuple(engine.get_tensor_shape(name))
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            arr = np.zeros(shape, dtype=dtype)
            self.host[name] = arr
            self.dev[name] = _ck(cudart.cudaMalloc(arr.nbytes))
            self.ctx.set_tensor_address(name, int(self.dev[name]))
        self.input_dtype = self.host["input"].dtype

    @classmethod
    def build_or_load(cls, spec, key, class_names, resolution=576, device_id=0):
        """Load the cached encrypted engine if present, else build + cache it."""
        cache = trt_cache_path(spec)
        if not os.path.exists(cache):
            precache(spec, key, device_id=device_id)
        with open(cache, "rb") as f:
            engine_bytes = crypto.decrypt(f.read(), key)
        return cls(engine_bytes, class_names, resolution=resolution, device_id=device_id)

    def predict(self, image_bgr, threshold=0.5):
        cudart = self._cudart
        h, w = image_bgr.shape[:2]
        _ck(cudart.cudaSetDevice(self.device_id))
        self.host["input"][:] = preprocess(image_bgr, self.resolution).astype(self.input_dtype)

        H2D = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        D2H = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        inp = self.host["input"]
        _ck(cudart.cudaMemcpyAsync(int(self.dev["input"]), inp.ctypes.data, inp.nbytes, H2D, self.stream))
        self.ctx.execute_async_v3(int(self.stream))
        for name in ("dets", "labels"):
            arr = self.host[name]
            _ck(cudart.cudaMemcpyAsync(arr.ctypes.data, int(self.dev[name]), arr.nbytes, D2H, self.stream))
        _ck(cudart.cudaStreamSynchronize(self.stream))

        return postprocess(self.host["dets"][0].astype(np.float32),
                           self.host["labels"][0].astype(np.float32),
                           w, h, threshold, self.class_names, self.num_select)

    def close(self):
        for ptr in self.dev.values():
            self._cudart.cudaFree(ptr)
        self.dev.clear()
