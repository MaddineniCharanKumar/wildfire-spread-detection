from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace

from fastapi.testclient import TestClient
from sqlmodel import create_engine

from fireguard import database, performance
from fireguard.api import app


def test_image_inference_is_saved_and_reflected_in_analytics(monkeypatch, tmp_path) -> None:
    test_engine = create_engine(
        f"sqlite:///{tmp_path / 'api-performance.db'}",
        connect_args={"check_same_thread": False},
    )
    monkeypatch.setattr(database, "engine", test_engine)
    monkeypatch.setattr(performance, "engine", test_engine)

    class FakeModel:
        def predict(self, **kwargs):
            box = SimpleNamespace(
                cls=SimpleNamespace(item=lambda: 0),
                conf=SimpleNamespace(item=lambda: 0.9),
                xyxy=[SimpleNamespace(tolist=lambda: [1.0, 2.0, 30.0, 40.0])],
            )
            return [SimpleNamespace(names={0: "person"}, boxes=[box])]

    from fireguard import api

    monkeypatch.setattr(api.runtime, "get_model", lambda: FakeModel())
    monkeypatch.setattr(
        api.runtime,
        "get_status",
        lambda: {"model_path": "test-model", "fire_smoke_classes_available": False},
    )
    monkeypatch.setattr(api, "_runtime_device", "cpu")

    from PIL import Image

    image = BytesIO()
    Image.new("RGB", (64, 48)).save(image, format="PNG")

    client = TestClient(app)
    response = client.post(
        "/api/detection/image",
        files={"file": ("test.png", image.getvalue(), "image/png")},
    )

    assert response.status_code == 200
    assert response.json()["detections"][0]["class_name"] == "person"
    runtime = client.get("/api/analytics/overview").json()["model_metrics"]["runtime"]
    assert runtime["sample_count"] == 1
    assert runtime["total_detections"] == 1
    assert runtime["detections_by_class"] == {"person": 1}


def test_inference_performance_starts_with_no_fabricated_samples(monkeypatch, tmp_path) -> None:
    test_engine = create_engine(
        f"sqlite:///{tmp_path / 'empty-performance.db'}",
        connect_args={"check_same_thread": False},
    )
    monkeypatch.setattr(database, "engine", test_engine)
    monkeypatch.setattr(performance, "engine", test_engine)

    summary = performance.get_inference_performance()

    assert summary["sample_count"] == 0
    assert summary["total_detections"] == 0
    assert summary["mean_latency_ms"] is None
    assert summary["single_image_fps_estimate"] is None
    assert summary["detections_by_class"] == {}
    assert summary["last_inference_at"] is None


def test_inference_performance_persists_samples_and_predictions(monkeypatch, tmp_path) -> None:
    test_engine = create_engine(
        f"sqlite:///{tmp_path / 'performance.db'}",
        connect_args={"check_same_thread": False},
    )
    monkeypatch.setattr(database, "engine", test_engine)
    monkeypatch.setattr(performance, "engine", test_engine)

    performance.record_inference(
        camera_id="test-camera",
        model_version="test-model",
        latency_ms=100,
        gpu_used=False,
        image_width=640,
        image_height=480,
        detections=[
            {
                "detection_id": "test-detection",
                "confidence": 0.9,
                "class_name": "fire",
                "bbox": [1.0, 2.0, 30.0, 40.0],
                "visual_area_proxy": 1102.0,
            }
        ],
    )
    performance.record_inference(
        camera_id="test-camera",
        model_version="test-model",
        latency_ms=200,
        gpu_used=False,
        image_width=640,
        image_height=480,
        detections=[],
    )

    summary = performance.get_inference_performance()

    assert summary["sample_count"] == 2
    assert summary["total_detections"] == 1
    assert summary["mean_latency_ms"] == 150
    assert summary["single_image_fps_estimate"] == 6.67
    assert summary["detections_by_class"] == {"fire": 1}
    assert summary["last_inference_at"] is not None
