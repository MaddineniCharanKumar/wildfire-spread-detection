from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func
from sqlmodel import Session, select

from fireguard.database import engine, init_db
from fireguard.models import Detection, InferenceLog


def record_inference(
    *,
    camera_id: str,
    model_version: str,
    latency_ms: float,
    gpu_used: bool,
    image_width: int,
    image_height: int,
    detections: list[dict[str, Any]],
) -> None:
    init_db()
    timestamp = datetime.now(timezone.utc)
    inference_fps = 1000 / latency_ms if latency_ms > 0 else 0

    with Session(engine) as session:
        session.add(
            InferenceLog(
                camera_id=camera_id,
                model_version=model_version,
                timestamp=timestamp,
                created_at=timestamp,
                inference_latency_ms=latency_ms,
                fps=inference_fps,
                gpu_used=gpu_used,
                status="OK",
            )
        )
        for item in detections:
            x1, y1, x2, y2 = item["bbox"]
            session.add(
                Detection(
                    detection_id=item["detection_id"],
                    event_id=item.get("event_id"),
                    camera_id=camera_id,
                    source_type="IMAGE",
                    timestamp=timestamp,
                    created_at=timestamp,
                    confidence=item["confidence"],
                    class_name=item["class_name"],
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    visual_area_proxy=item["visual_area_proxy"],
                    image_width=image_width,
                    image_height=image_height,
                    inference_latency_ms=latency_ms,
                    model_version=model_version,
                    is_valid=True,
                )
            )
        session.commit()


def get_inference_performance() -> dict[str, Any]:
    init_db()
    with Session(engine) as session:
        sample_count, mean_latency, last_inference = session.exec(
            select(
                func.count(InferenceLog.id),
                func.avg(InferenceLog.inference_latency_ms),
                func.max(InferenceLog.timestamp),
            )
        ).one()
        detection_count = session.exec(select(func.count(Detection.id))).one()
        detections_by_class = session.exec(
            select(Detection.class_name, func.count(Detection.id))
            .group_by(Detection.class_name)
            .order_by(Detection.class_name)
        ).all()

    mean_latency_ms = round(float(mean_latency), 2) if mean_latency is not None else None
    if last_inference is not None and last_inference.tzinfo is None:
        last_inference = last_inference.replace(tzinfo=timezone.utc)
    return {
        "sample_count": sample_count,
        "total_detections": detection_count,
        "mean_latency_ms": mean_latency_ms,
        "single_image_fps_estimate": (
            round(1000 / mean_latency_ms, 2)
            if mean_latency_ms is not None and mean_latency_ms > 0
            else None
        ),
        "detections_by_class": {name: count for name, count in detections_by_class},
        "last_inference_at": last_inference.isoformat() if last_inference else None,
        "scope": "Persisted image-inference samples; prediction counts are not accuracy metrics.",
    }
