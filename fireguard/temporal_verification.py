from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, List, Tuple

from fireguard.config import settings
from fireguard.runtime import FireguardRuntime


@dataclass
class DetectionBox:
    class_id: int
    confidence: float
    bbox: Tuple[float, float, float, float]
    centroid: Tuple[float, float]
    timestamp: datetime

    @property
    def area_proxy(self) -> float:
        x, y, w, h = self.bbox
        return max(0.0, w) * max(0.0, h)


class YOLODetector:
    """Minimal YOLO-compatible inference wrapper using shared runtime."""

    def __init__(self, model_path: str | None = None, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self._runtime = FireguardRuntime(model_path=model_path, device=device)

    def load(self):
        return self._runtime.load()

    def get_model(self):
        return self._runtime.get_model()

    def predict_image(self, image_path: str) -> List[dict[str, Any]]:
        model = self.get_model()
        if model is None:
            return [{
                "class_id": 0,
                "class_name": "fire_or_smoke",
                "confidence": 0.0,
                "bbox": [0.0, 0.0, 0.0, 0.0],
                "centroid": [0.0, 0.0],
                "source": "demo",
                "note": "No trained YOLO model loaded; this is a placeholder implementation.",
            }]

        results = model(image_path, conf=0.25, device=self.device)
        detections: List[dict[str, Any]] = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                detections.append({
                    "class_id": cls,
                    "class_name": getattr(result.names, str(cls), f"class_{cls}"),
                    "confidence": conf,
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "centroid": [float(cx), float(cy)],
                    "source": "yolo",
                })
        return detections


class InferenceRunner:
    def __init__(self, model_path: str | None = None, device: str = "cpu"):
        self.detector = YOLODetector(model_path=model_path, device=device)

    def run_image(self, image_path: str) -> dict[str, Any]:
        detections = self.detector.predict_image(image_path)
        return {
            "status": "ok" if detections else "no_detection",
            "detections": detections,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "device": self.detector.device,
        }