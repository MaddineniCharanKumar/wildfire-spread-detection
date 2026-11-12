# GPU and device documentation

Runtime GPU capabilities are discovered dynamically from the environment using `utils/device.py`.

The detection stack supports:
- CUDA available devices
- FP16 and mixed precision
- CPU fallback
- multi-GPU detection when present
- batch inference settings
- no hardcoded dataset classes

The API exposes live GPU state at `/api/system/gpu`.
