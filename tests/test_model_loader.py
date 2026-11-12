from __future__ import annotations

from unittest.mock import MagicMock, patch

from fireguard.model_loader import (
    MODEL_FAMILY_MIN_VERSION,
    _build_weight_name,
    get_model_status,
    resolve_model_weights,
)
from fireguard.runtime import FireguardRuntime


class TestWeightNameBuilding:
    def test_yolo26_nano(self):
        assert _build_weight_name("yolo26", "n") == "yolo26n.pt"

    def test_yolo26_small(self):
        assert _build_weight_name("yolo26", "s") == "yolo26s.pt"

    def test_yolo27_medium(self):
        assert _build_weight_name("yolo27", "m") == "yolo27m.pt"

    def test_unknown_family_defaults_to_yolo26(self):
        assert _build_weight_name("unknown", "n") == "yolo26n.pt"

    def test_unknown_size_defaults_to_nano(self):
        assert _build_weight_name("yolo26", "unknown") == "yolo26n.pt"

    def test_case_insensitive(self):
        assert _build_weight_name("YOLO26", "N") == "yolo26n.pt"


class TestResolveModelWeights:
    def test_default_weights_resolution(self):
        path, meta = resolve_model_weights("yolo26", "n")
        assert path == "yolo26n.pt"
        assert meta["status"] == "READY"
        assert meta["source"] == "default"

    def test_custom_weights_priority_when_exists(self, tmp_path):
        custom = tmp_path / "my_model.pt"
        custom.write_text("dummy")
        path, meta = resolve_model_weights("yolo26", "n", custom_weights=str(custom))
        assert path == str(custom)
        assert meta["source"] == "custom"

    def test_custom_weights_fallback_when_missing(self):
        path, meta = resolve_model_weights("yolo26", "n", custom_weights="/nonexistent/path.pt")
        assert path == "yolo26n.pt"
        assert meta["source"] == "default"
        assert "error" in meta

    def test_version_check_in_meta(self):
        _, meta = resolve_model_weights("yolo26", "n")
        assert "ultralytics_version" in meta
        assert "version_ok" in meta


class TestGetModelStatus:
    @patch("fireguard.model_loader.load_yolo_model")
    def test_status_ready_when_load_succeeds(self, mock_load):
        mock_model = MagicMock()
        mock_model.names = {0: "fire", 1: "smoke"}
        mock_model.task = "detect"
        mock_load.return_value = (mock_model, {"status": "LOADED"})

        status = get_model_status("yolo26", "n", device="cpu")
        assert status["status"] == "LOADED"
        assert status["model_names"] == {0: "fire", 1: "smoke"}
        assert status["task"] == "detect"

    @patch("fireguard.model_loader.load_yolo_model")
    def test_status_unavailable_when_load_fails(self, mock_load):
        mock_load.return_value = (None, {"status": "MODEL_UNAVAILABLE", "error": "load failed"})
        status = get_model_status("yolo26", "n", device="cpu")
        assert status["status"] == "MODEL_UNAVAILABLE"
        assert "error" in status

    @patch("fireguard.model_loader.load_yolo_model")
    def test_status_unavailable_when_ultralytics_version_too_old(self, mock_load):
        with patch("fireguard.model_loader._check_ultralytics_version", return_value=(False, "8.2.0")):
            status = get_model_status("yolo26", "n", device="cpu")
            assert status["status"] == "MODEL_UNAVAILABLE"
            assert "8.3.0" in status["error"]


class TestModelFamilyMinVersion:
    def test_yolo26_has_min_version(self):
        assert "yolo26" in MODEL_FAMILY_MIN_VERSION
        assert MODEL_FAMILY_MIN_VERSION["yolo26"] == "8.3.0"

    def test_yolo27_placeholder(self):
        assert "yolo27" in MODEL_FAMILY_MIN_VERSION


class TestRuntime:
    @patch("fireguard.runtime.load_yolo_model")
    def test_default_runtime_loads_local_yolo26_and_reports_fire_classes(self, mock_load):
        mock_load.return_value = (
            MagicMock(),
            {"status": "LOADED", "model_names": {0: "fire", 1: "smoke"}, "task": "detect"},
        )
        runtime = FireguardRuntime(device="cpu")
        assert runtime.load()
        status = runtime.get_status()
        assert status["family"] == "yolo26"
        assert status["status"] == "LOADED"
        assert status["fire_smoke_classes_available"] is True
        assert status["inference_ready"] is True

    @patch("fireguard.runtime.load_yolo_model")
    def test_coco_weights_do_not_claim_wildfire_detection(self, mock_load):
        mock_load.return_value = (
            MagicMock(),
            {"status": "LOADED", "model_names": {0: "person", 1: "bicycle"}, "task": "detect"},
        )
        runtime = FireguardRuntime(device="cpu")
        assert runtime.load()
        status = runtime.get_status()
        assert status["status"] == "LOADED"
        assert status["fire_smoke_classes_available"] is False
        assert status["inference_ready"] is False