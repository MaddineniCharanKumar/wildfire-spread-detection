from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class InferenceResult:
    image_path: str
    detections: list[dict[str, Any]]
    device: str
    model_version: str


class FireDetectionPipeline:
    """High-level pipeline combining detection, temporal verification and event creation."""

    def __init__(self, detector, temporal_verifier, event_manager):
        self.detector = detector
        self.temporal_verifier = temporal_verifier
        self.event_manager = event_manager

    def run(self, image_path: str, camera_id: str = "demo-camera", source_type: str = "IMAGE", model_version: str = "demo-v1") -> dict[str, Any]:
        raw_detections = self.detector.predict_image(image_path)
        confirmed = []
        for detection in raw_detections:
            bbox = tuple(detection.get("bbox", [0.0, 0.0, 0.0, 0.0]))
            conf = float(detection.get("confidence", 0.0))
            if self.temporal_verifier.push(conf, bbox):
                confirmed.append(detection)

        event = self.event_manager.create(camera_id=camera_id, source_type=source_type, model_version=model_version)
        if confirmed:
            best = max(confirmed, key=lambda item: float(item.get("confidence", 0.0)))
            event.confidence = float(best.get("confidence", 0.0))
            event.growth_rate = 10.0
            event.spread_direction = "NORTH"
            event.status = "CONFIRMED"
        return {
            "event_id": event.event_id,
            "camera_id": camera_id,
            "source_type": source_type,
            "detections": confirmed,
            "status": event.status,
            "confidence": event.confidence,
        }
