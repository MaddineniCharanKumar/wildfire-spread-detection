from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Supported families and their minimum ultralytics versions
MODEL_FAMILY_MIN_VERSION = {
    "yolo26": "8.3.0",
    "yolo27": "8.3.0",  # placeholder; update when released
}


def _build_weight_name(family: str, size: str) -> str:
    """Build default weight filename from family and size (e.g., 'yolo26n.pt')."""
    fam = family.lower().strip()
    sz = size.lower().strip()
    if fam not in {"yolo26", "yolo27"}:
        logger.warning("Unknown model family '%s', defaulting to yolo26", fam)
        fam = "yolo26"
    if sz not in {"n", "s", "m", "l", "x"}:
        logger.warning("Unknown model size '%s', defaulting to n", sz)
        sz = "n"
    return f"{fam}{sz}.pt"


def _check_ultralytics_version(min_version: str) -> tuple[bool, str]:
    """Check if installed ultralytics meets minimum version."""
    try:
        import ultralytics
        from packaging import version
        installed = ultralytics.__version__
        ok = version.parse(installed) >= version.parse(min_version)
        return ok, installed
    except Exception as e:
        logger.warning("Could not check ultralytics version: %s", e)
        return False, "unknown"


def resolve_model_weights(
    family: str,
    size: str,
    custom_weights: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Resolve the model weights path and return metadata.

    Priority:
    1. Explicit `custom_weights` path (if file exists)
    2. Built-in weight name `{family}{size}.pt` (Ultralytics auto-downloads)

    Returns (weights_path, metadata_dict)
    """
    meta: dict[str, Any] = {
        "family": family,
        "size": size,
        "custom_weights": custom_weights,
        "ultralytics_version": "unknown",
        "version_ok": False,
        "status": "UNKNOWN",
    }

    # Check ultralytics version
    min_ver = MODEL_FAMILY_MIN_VERSION.get(family.lower(), "8.3.0")
    ver_ok, installed_ver = _check_ultralytics_version(min_ver)
    meta["ultralytics_version"] = installed_ver
    meta["version_ok"] = ver_ok

    if not ver_ok:
        meta["status"] = "MODEL_UNAVAILABLE"
        meta["error"] = f"ultralytics>={min_ver} required for {family}, found {installed_ver}"
        return "", meta

    # Custom weights take priority
    if custom_weights:
        p = Path(custom_weights).expanduser().resolve()
        if p.exists():
            meta["weights_path"] = str(p)
            meta["status"] = "READY"
            meta["source"] = "custom"
            return str(p), meta
        else:
            logger.warning("Custom weights not found at %s, falling back to default", p)
            meta["error"] = f"Custom weights not found: {p}"

    # Default: built-in weight name
    weight_name = _build_weight_name(family, size)
    meta["weights_path"] = weight_name
    meta["status"] = "READY"
    meta["source"] = "default"
    return weight_name, meta


def load_yolo_model(
    weights_path: str,
    device: str = "cpu",
) -> tuple[Any, dict[str, Any]]:
    """
    Load a YOLO model using the public Ultralytics API.

    Returns (model, metadata_dict) or (None, metadata_dict) on failure.
    """
    meta: dict[str, Any] = {
        "weights_path": weights_path,
        "device": device,
        "status": "UNKNOWN",
    }

    try:
        from ultralytics import YOLO
        model = YOLO(weights_path)
        meta["status"] = "LOADED"
        meta["model_names"] = getattr(model, "names", {})
        meta["task"] = getattr(model, "task", "detect")
        logger.info("Loaded YOLO model from %s on %s", weights_path, device)
        return model, meta
    except Exception as e:
        meta["status"] = "MODEL_UNAVAILABLE"
        meta["error"] = f"{e.__class__.__name__}: {e}"
        logger.error("Failed to load YOLO model from %s: %s", weights_path, e)
        return None, meta


def get_model_status(
    family: str,
    size: str,
    custom_weights: str | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """
    Get model status without loading the full model (for /api/model/status).
    """
    weights_path, resolve_meta = resolve_model_weights(family, size, custom_weights)

    status = {
        "family": family,
        "size": size,
        "weights_path": weights_path,
        "device": device,
        "ultralytics_version": resolve_meta.get("ultralytics_version"),
        "version_ok": resolve_meta.get("version_ok"),
        "source": resolve_meta.get("source"),
        "resolve_status": resolve_meta.get("status"),
        "resolve_error": resolve_meta.get("error"),
        "status": "UNKNOWN",
    }

    if resolve_meta.get("status") != "READY":
        status["status"] = "MODEL_UNAVAILABLE"
        status["error"] = resolve_meta.get("error")
        return status

    # Try a lightweight load to verify
    model, load_meta = load_yolo_model(weights_path, device)
    status["status"] = load_meta.get("status", "MODEL_UNAVAILABLE")
    if model is not None:
        status["model_names"] = load_meta.get("model_names") or getattr(model, "names", None)
        status["task"] = load_meta.get("task") or getattr(model, "task", None)
    else:
        status["error"] = load_meta.get("error")

    return status