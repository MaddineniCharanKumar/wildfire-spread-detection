# Deployment

Use a process manager/container, PostgreSQL, Redis, TLS termination, JWT authentication, upload size/type validation, rate limiting, secret injection, and least-privilege database credentials. The included local API is intentionally safe-by-default and does not claim production security until those controls are configured.

## Model Configuration

The detector uses Ultralytics YOLO. Configure via environment variables:

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

### Docker

```bash
docker-compose up -d --build
```

The `docker-compose.yml` mounts `.env` and passes `FIREGUARD_DEVICE` through. For GPU, ensure the host has NVIDIA Container Toolkit and add `deploy.resources.reservations.devices` in compose.

### Secrets

Never commit `.env`. Inject secrets at runtime (Docker secrets, Kubernetes secrets, Vault, etc.). Required secrets:
- `FIREGUARD_FIRMS_MAP_KEY` (NASA FIRMS)
- Optional: `FIREGUARD_WEATHER_API_KEY` if using a keyed weather provider