# Model Configuration & Scientific Limitations

## Model Configuration

The detector uses **Ultralytics YOLO**. Configure via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `FIREGUARD_MODEL_FAMILY` | `yolo26` | Model family: `yolo26` (current) or `yolo27` (when released). |
| `FIREGUARD_MODEL_SIZE` | `n` | Model size: `n`, `s`, `m`, `l`, `x`. |
| `FIREGUARD_MODEL_WEIGHTS` | *(empty)* | Optional path to custom trained weights (`.pt`). If set, takes priority over family+size. |

**Default weight resolution**: If `FIREGUARD_MODEL_WEIGHTS` is not set, the loader builds the weight filename as `{family}{size}.pt` (e.g., `yolo26n.pt`). Ultralytics will auto-download the official weights on first use.

**YOLO27 readiness**: When Ultralytics releases YOLO27, set `FIREGUARD_MODEL_FAMILY=yolo27` and `FIREGUARD_MODEL_SIZE` to your desired size. No code changes required. **Note**: YOLO27 has not been released or tested as of this writing.

### Minimum Ultralytics Version

`ultralytics>=8.3` is required for YOLOv8/YOLO11/YOLO26 compatibility. The `pyproject.toml` pins this minimum.

### GPU / CPU

- `FIREGUARD_DEVICE=cpu` (default) or `cuda` / `cuda:0` etc.
- On CUDA devices, mixed precision (FP16) is enabled automatically.
- The API degrades gracefully to CPU if CUDA is unavailable.

---

## Scientific Limitations

Bounding-box area is an **area proxy**, not physical fire area. Centroid and box motion are estimated visual spread direction, not true propagation speed. Risk is an environmental fusion estimate, not future-fire certainty. The system does not infer exact temperature. `DEMO DATA` is explicitly marked and must never be used for operational decisions.

Only evaluation runs against real labelled data may populate model metrics. Missing data is reported as unavailable rather than replaced with a fabricated value.

---

## License Notice

**Ultralytics code and models are licensed under AGPL-3.0**. If you distribute this software or use it as a network service, you must comply with AGPL-3.0 (source code availability, same license for derivatives). See https://github.com/ultralytics/ultralytics/blob/main/LICENSE for the full license text.