from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from fireguard.api import app
from fireguard.domain import area_proxy, growth_rate
from fireguard.environment import DemoProvider, UnavailableProvider
from fireguard.forecast import wind_advection_forecast
from fireguard.risk import calculate_location_risk, calculate_risk


def test_risk_is_bounded() -> None:
    result = calculate_risk({"visual": 100})
    assert 0 <= result["score"] <= 100


def test_location_risk_is_weather_index_and_requires_fresh_inputs() -> None:
    result = calculate_location_risk({
        "freshness": "LIVE",
        "source": "TEST_WEATHER",
        "timestamp": "2026-10-23T12:00:00Z",
        "temperature": 35,
        "humidity": 20,
        "wind_speed": 30,
        "rainfall": 0,
    })
    assert result["status"] == "ESTIMATE"
    assert result["score"] == 74.0
    assert result["basis"].startswith("Weather-only heuristic")
    assert "probability" in result["basis"]


def test_location_risk_never_estimates_from_stale_or_demo_data() -> None:
    stale = calculate_location_risk({
        "freshness": "STALE",
        "temperature": 35,
        "humidity": 20,
        "wind_speed": 30,
    })
    demo = calculate_location_risk({
        "freshness": "DEMO_MODE",
        "demo": True,
        "temperature": 35,
        "humidity": 20,
    })
    assert stale["status"] == "UNAVAILABLE" and stale["score"] is None
    assert demo["status"] == "DEMO_MODE" and demo["score"] is None


def test_area_proxy_is_not_physical_area() -> None:
    assert area_proxy((0, 0, 0.5, 0.2)) == 0.1


def test_growth() -> None:
    assert growth_rate(10, 15) == 50


def test_wind_forecast_projects_downwind_from_observed_wind() -> None:
    result = wind_advection_forecast(12.0, 77.0, 12.0, 0.0, 30)
    assert result["model"] == "wind_advection_baseline_v1"
    assert result["drift_km"] == 6
    assert result["projected_latitude"] < 12.0
    assert abs(result["projected_longitude"] - 77.0) < 1e-8
    assert result["status"] == "ESTIMATE"


def test_forecast_rejects_invalid_coordinates() -> None:
    try:
        wind_advection_forecast(91.0, 77.0, 12.0, 0.0, 30)
    except ValueError as error:
        assert "latitude" in str(error)
    else:
        raise AssertionError("invalid latitude must be rejected")


def test_unavailable_provider_marks_data_unavailable() -> None:
    provider = UnavailableProvider()
    payload = asyncio.run(provider.current())
    assert payload["freshness"] == "UNAVAILABLE"
    assert payload["status"] == "UNAVAILABLE"


def test_demo_provider_is_explicitly_labeled() -> None:
    payload = asyncio.run(DemoProvider().current())
    assert payload["demo"] is True
    assert payload["status"] == "DEMO_MODE"


def test_api_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_forecast_endpoint_returns_not_found_for_unknown_event() -> None:
    response = TestClient(app).get("/api/forecast/unknown-event?horizon_minutes=30")
    assert response.status_code == 404


def test_analytics_omits_unavailable_accuracy_metrics() -> None:
    result = TestClient(app).get("/api/analytics/overview").json()
    assert "evaluation_status" not in result["model_metrics"]
    assert "runtime" in result["model_metrics"]


def test_analytics_exposes_latest_dataset_evaluation_and_provider_reports(monkeypatch) -> None:
    from fireguard import api

    monkeypatch.setattr(api, "extract_evaluation_report", lambda: {"precision": 0.9})
    monkeypatch.setattr(api, "extract_dataset_report", lambda: {"counts": {"images": 12}})
    monkeypatch.setattr(api, "extract_live_provider_report", lambda: {"status": "PASS"})

    result = TestClient(app).get("/api/analytics/overview").json()

    assert result["model_metrics"]["validation"]["precision"] == 0.9
    assert result["dataset_report"]["counts"]["images"] == 12
    assert result["provider_check"]["status"] == "PASS"


def test_location_risk_endpoint_uses_requested_coordinates(monkeypatch) -> None:
    from fireguard import api

    async def fake_weather(latitude, longitude):
        assert latitude == 13.2
        assert longitude == 77.8
        return {
            "freshness": "LIVE",
            "source": "TEST_WEATHER",
            "timestamp": "2026-10-23T12:00:00Z",
            "temperature": 35,
            "humidity": 20,
            "wind_speed": 30,
        }

    monkeypatch.setattr(api.environment_provider, "current", fake_weather)
    response = TestClient(app).get("/api/risk/location?latitude=13.2&longitude=77.8")
    assert response.status_code == 200
    assert response.json()["latitude"] == 13.2
    assert response.json()["status"] == "ESTIMATE"
    assert response.json()["score"] is not None


def test_historical_endpoint_uses_requested_site_and_radius(monkeypatch) -> None:
    from fireguard import api

    async def fake_history(start_date, days, latitude, longitude, limit, radius_km):
        return {
            "status": "AVAILABLE",
            "source": "NASA_FIRMS",
            "requested": [latitude, longitude, radius_km],
            "records": [],
        }

    monkeypatch.setattr(api, "fetch_firms_history", fake_history)
    response = TestClient(app).get(
        "/api/satellite/history?start_date=2026-10-20&days=3&limit=50"
        "&latitude=13.2&longitude=77.8&radius_km=25"
    )
    assert response.status_code == 200
    assert response.json()["requested"] == [13.2, 77.8, 25]
    assert TestClient(app).get(
        "/api/satellite/history?start_date=2026-10-20&latitude=13.2"
    ).status_code == 422


def test_historical_endpoint_exposes_provider_setup_without_fabricated_records(monkeypatch) -> None:
    from fireguard.config import settings

    monkeypatch.setattr(settings, "firms_map_key", None)
    result = TestClient(app).get("/api/satellite/history?start_date=2026-01-01&days=5&limit=6")
    assert result.status_code == 200
    assert result.json()["status"] == "CONFIG_REQUIRED"
    assert result.json()["records"] == []


def test_monitoring_location_is_persisted_on_api_host(monkeypatch, tmp_path) -> None:
    from fireguard import api

    async def no_refresh():
        return None

    monkeypatch.setattr(api, "_local_settings_path", tmp_path / "local_settings.json")
    monkeypatch.setattr(api, "collect_once", no_refresh)
    result = TestClient(app).put(
        "/api/system/location",
        json={"latitude": 13.0, "longitude": 77.6, "radius_km": 80},
    )
    assert result.status_code == 200
    assert result.json()["status"] == "SAVED"
    assert api._local_settings_path.exists()
    assert api._local_settings_path.read_text(encoding="utf-8").find('"latitude": 13.0') >= 0
    invalid = TestClient(app).put(
        "/api/system/location",
        json={"latitude": 91, "longitude": 77.6, "radius_km": 80},
    )
    assert invalid.status_code == 422


def test_dashboard_summary_is_serializable_and_never_synthetic(monkeypatch) -> None:
    from fireguard.config import settings
    monkeypatch.setattr(settings, "firms_map_key", None)  # no key -> no network call, explicit status
    monkeypatch.setattr(settings, "demo_mode", False)
    body = TestClient(app).get("/api/dashboard/summary").json()
    assert isinstance(body["live"], dict) and "sources" in body["live"]
    assert body["live"]["config"]["firms_key_configured"] is False
    assert body["satellite_meta"]["satellite_status"] == "CONFIG_REQUIRED"
    assert body["satellite_fires"] == [] and body["fires"] == [] and body["alerts"] == []
    assert body["environment"].get("demo") is False
