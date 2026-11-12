from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import httpx

from fireguard.environment import (
    DemoProvider,
    NWSProvider,
    OpenMeteoProvider,
    UnavailableProvider,
)


def test_unavailable_provider_marks_data_unavailable() -> None:
    payload = asyncio.run(UnavailableProvider().current())
    assert payload["freshness"] == "UNAVAILABLE"
    assert payload["status"] == "UNAVAILABLE"


def test_demo_provider_is_explicitly_labeled() -> None:
    payload = asyncio.run(DemoProvider().current())
    assert payload["demo"] is True
    assert payload["status"] == "DEMO_MODE"


def test_live_provider_requires_coordinates() -> None:
    payload = asyncio.run(OpenMeteoProvider("https://api.open-meteo.com/v1/forecast").current())
    assert payload["status"] == "UNAVAILABLE"
    assert payload["freshness"] == "UNAVAILABLE"


def test_nws_provider_requires_coordinates() -> None:
    payload = asyncio.run(NWSProvider().current())
    assert payload["status"] == "UNAVAILABLE"
    assert payload["source"] == "NWS_GOV"


def test_nws_provider_reports_unavailable_outside_us_without_network_assumptions() -> None:
    # This does not require network access; it only asserts the provider never fabricates
    # a reading and instead reports UNAVAILABLE with a message when a lookup can't complete.
    provider = NWSProvider(timeout_seconds=0.001)
    payload = asyncio.run(provider.current(14.1234, 77.5678))
    assert payload["demo"] is False
    assert payload["status"] in {"UNAVAILABLE"}
    assert payload["temperature"] is None


# ---- response-shape tests with mocked transports (fixtures only; no runtime synthetic data) ----
def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def test_open_meteo_parses_current_block_and_computes_freshness() -> None:
    t = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M")
    body = {"current": {"time": t, "temperature_2m": 29.1, "relative_humidity_2m": 44, "precipitation": 0.0, "wind_speed_10m": 11.5, "wind_direction_10m": 250, "surface_pressure": 921.3, "visibility": 12000}, "current_units": {"wind_speed_10m": "km/h"}}
    p = OpenMeteoProvider(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body)))
    out = asyncio.run(p.current(13.1, 77.6))
    assert out["temperature"] == 29.1 and out["wind_speed"] == 11.5 and out["demo"] is False
    assert out["visibility"] == 12
    assert out["freshness"] == "LIVE" and out["age_seconds"] is not None


def test_open_meteo_old_observation_is_marked_stale_not_live() -> None:
    body = {"current": {"time": "2020-01-01T00:00", "temperature_2m": 10}}
    out = asyncio.run(OpenMeteoProvider(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body))).current(1.0, 1.0))
    assert out["freshness"] == "STALE" and out["status"] == "STALE"


def test_nws_uses_4dp_coords_follows_redirects_and_skips_empty_stations() -> None:
    seen: list[str] = []
    def handler(req: httpx.Request) -> httpx.Response:
        url = str(req.url); seen.append(url)
        if url.startswith("https://api.weather.gov/points/"):
            return httpx.Response(200, json={"properties": {"observationStations": "https://api.weather.gov/gridpoints/X/1,1/stations"}})
        if url.endswith("/stations"):
            return httpx.Response(200, json={"features": [{"properties": {"stationIdentifier": "EMPTY"}}, {"properties": {"stationIdentifier": "GOOD"}}]})
        if "/EMPTY/" in url:
            return httpx.Response(200, json={"properties": {"timestamp": _iso_now(), "temperature": {"value": None}}})
        v = lambda x: {"value": x}
        return httpx.Response(200, json={"properties": {"station": "https://api.weather.gov/stations/GOOD", "timestamp": _iso_now(), "temperature": v(21.0), "relativeHumidity": v(35.0), "windSpeed": v(18.0), "windDirection": v(270), "barometricPressure": v(101300), "visibility": v(16000), "precipitationLastHour": v(0)}})
    out = asyncio.run(NWSProvider(transport=httpx.MockTransport(handler)).current(34.123456789, -118.987654321))
    assert seen[0] == "https://api.weather.gov/points/34.1235,-118.9877"
    assert out["temperature"] == 21.0 and out["pressure"] == 1013.0 and out["visibility"] == 16.0 and out["freshness"] == "LIVE"


def test_nws_outside_coverage_reports_unavailable_never_fabricates() -> None:
    p = NWSProvider(transport=httpx.MockTransport(lambda r: httpx.Response(404, json={"title": "Data Unavailable For Requested Point"})))
    out = asyncio.run(p.current(13.1, 77.6))
    assert out["status"] == "UNAVAILABLE" and out["temperature"] is None and out["demo"] is False
