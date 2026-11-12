from __future__ import annotations

from pathlib import Path
from typing import Any

from fireguard.config import settings
from fireguard.model_loader import _build_weight_name, load_yolo_model


class FireguardRuntime:
    def __init__(self, model_path: str | None = None, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self._model = None
        self._model_meta: dict[str, Any] = {}
    def load(self) -> bool:
        """Load configured YOLO weights or the local family/size weights."""
        if self._model is not None:
            return True
        weight_name = self.model_path or _build_weight_name(settings.model_family, settings.model_size)
        candidate = Path(weight_name).expanduser()
        if not candidate.is_absolute() and (Path(__file__).resolve().parents[1] / candidate).is_file():
            candidate = Path(__file__).resolve().parents[1] / candidate
        self.model_path = str(candidate.resolve()) if candidate.is_file() else weight_name
        self._model, self._model_meta = load_yolo_model(self.model_path, self.device)
        return self._model is not None

    def get_model(self):
        if self._model is None:
            self.load()
        return self._model

    def get_status(self) -> dict[str, Any]:
        if self._model is None and not self._model_meta:
            self.load()
        names = self._model_meta.get("model_names")
        class_names = list(names.values()) if isinstance(names, dict) else []
        has_wildfire_classes = any(
            str(name).strip().lower() in {"fire", "smoke"} for name in class_names
        )
        return {
            "model_path": self.model_path,
            "weights_path": self.model_path,
            "device": self.device,
            "family": settings.model_family,
            "size": settings.model_size,
            "status": self._model_meta.get("status", "MODEL_UNAVAILABLE"),
            "model_names": names,
            "class_names": class_names,
            "task": self._model_meta.get("task"),
            "fire_smoke_classes_available": has_wildfire_classes,
            "inference_ready": self._model is not None and has_wildfire_classes,
            "error": self._model_meta.get("error"),
        }


class ModelRuntime:
    """Legacy compatibility wrapper."""
    def __init__(self, weights_path: str | None = None):
        self.weights_path = weights_path

    def status(self) -> dict[str, Any]:
        return {
            "weights_path": self.weights_path,
            "ready": bool(self.weights_path),
            "note": "Real YOLO weights are required before production inference.",
        }