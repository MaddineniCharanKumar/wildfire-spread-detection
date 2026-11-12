"""FIRMS client tests. Responses below are *test fixtures* in FIRMS's documented CSV format,
served through httpx.MockTransport; runtime code never generates data."""
from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone

import httpx
import pytest

from fireguard import live_data
from fireguard.config import settings

HEADER = "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight"


def _row(lat, lon, conf="n", frp="3.2", sat="N21"):
    now = datetime.now(timezone.utc)
    return f"{lat},{lon},330.1,0.4,0.4,{now:%Y-%m-%d},{now:%H%M},{sat},VIIRS,{conf},2.0NRT,295.0,{frp},D"


@pytest.fixture(autouse=True)
def _cfg(monkeypatch):
    live_data._firms_cache.clear()
    monkeypatch.setattr(settings, "firms_map_key", "TESTKEY")
    monkeypatch.setattr(settings, "firms_source", "VIIRS_NOAA21_NRT,VIIRS_SNPP_NRT")
    monkeypatch.setattr(settings, "firms_radius_km", 100.0)
    monkeypatch.setattr(settings, "firms_days", 1)
    monkeypatch.setattr(settings, "firms_poll_seconds", 600)


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_bbox_is_west_south_east_north_and_scaled_by_latitude():
    w, s, e, n = live_data.firms_bbox(13.0, 77.6, 100)
    assert w < 77.6 < e and s < 13.0 < n
    assert abs((n - s) / 2 - 100 / 111.32) < 1e-6
    assert (e - w) > (n - s)  # a degree of longitude is shorter than latitude away from the equator


def test_bbox_is_clamped_to_valid_range():
    w, s, e, n = live_data.firms_bbox(89.9, 179.9, 500)
    assert -180 <= w and e <= 180 and -90 <= s and n <= 90


def test_no_key_reports_config_required_without_network(monkeypatch):
    monkeypatch.setattr(settings, "firms_map_key", None)
    out = asyncio.run(live_data.fetch_firms(13.0, 77.6))
    assert out["satellite_status"] == "CONFIG_REQUIRED" and out["fires"] == []


def test_history_no_key_reports_config_required_without_network(monkeypatch):
    monkeypatch.setattr(settings, "firms_map_key", None)
    out = asyncio.run(live_data.fetch_firms_history(date(2026, 1, 1), 5, 13.0, 77.6))
    assert out["status"] == "CONFIG_REQUIRED"
    assert out["records"] == []


def test_history_queries_dated_standard_processing_and_returns_measured_metrics():
    seen = []
    def handler(request: httpx.Request):
        seen.append(str(request.url))
        return httpx.Response(200, text=f"{HEADER}\n{_row(13.05, 77.60, conf='h', frp='6.2')}")

    out = asyncio.run(live_data.fetch_firms_history(
        date(2026, 1, 1), 5, 13.0, 77.6, client=_client(handler)
    ))
    assert out["status"] == "AVAILABLE"
    assert len(out["records"]) == 2
    assert out["metrics"]["returned_records"] == 2
    assert out["metrics"]["mean_frp_mw"] == 6.2
    assert out["metrics"]["high_confidence_records"] == 2
    assert out["metrics"]["metric_scope"].startswith("Descriptive satellite observations")
    assert "/VIIRS_NOAA21_SP/" in seen[0] and seen[0].endswith("/5/2026-01-01")


def test_history_filters_records_using_requested_radius():
    def handler(request: httpx.Request):
        return httpx.Response(
            200,
            text=f"{HEADER}\n{_row(13.05, 77.60)}\n{_row(13.2, 77.7, conf='h')}",
        )

    out = asyncio.run(live_data.fetch_firms_history(
        date(2026, 1, 1),
        5,
        13.0,
        77.6,
        radius_km=10,
        client=_client(handler),
    ))
    assert out["status"] == "AVAILABLE"
    assert len(out["records"]) == 2
    assert all(record["distance_km"] <= 10 for record in out["records"])


def test_parse_csv_fields_and_distance():
    fires = live_data.parse_firms_csv(f"{HEADER}\n{_row(13.05, 77.60)}", "VIIRS_NOAA21_NRT", 13.0, 77.6)
    assert len(fires) == 1
    f = fires[0]
    assert f["source"] == "NASA_FIRMS" and f["latitude"] == 13.05 and 5 < f["distance_km"] < 6
    assert f["observed_at"] and f["age_minutes"] is not None and f["age_minutes"] < 5


def test_plain_text_error_is_not_parsed_as_data():
    with pytest.raises(live_data.FirmsError):
        live_data.parse_firms_csv("Invalid MAP_KEY.", "VIIRS_SNPP_NRT")


def test_fetch_uses_bbox_key_source_and_day_range_and_dedupes_sources():
    seen = []
    def handler(req: httpx.Request):
        seen.append(str(req.url))
        return httpx.Response(200, text=f"{HEADER}\n{_row(13.05, 77.60)}\n{_row(13.2, 77.7, conf='h')}")
    out = asyncio.run(live_data.fetch_firms(13.0, 77.6, client=_client(handler)))
    assert out["satellite_status"] == "LIVE" and out["sources"] == ["VIIRS_NOAA21_NRT", "VIIRS_SNPP_NRT"]
    assert len(seen) == 2 and seen[0].startswith("https://firms.modaps.eosdis.nasa.gov/api/area/csv/TESTKEY/VIIRS_NOAA21_NRT/")
    assert seen[0].endswith("/1") and len(seen[0].split("/")[-2].split(",")) == 4
    # same point reported by two sources has different event_ids (source is in the id) -> both kept, not silently merged
    assert out["total_detections"] == 4
    assert out["fires"][0]["distance_km"] <= out["fires"][-1]["distance_km"]  # nearest first


def test_second_call_is_served_from_cache():
    calls = []
    def handler(req):
        calls.append(1); return httpx.Response(200, text=f"{HEADER}\n{_row(13.05, 77.6)}")
    asyncio.run(live_data.fetch_firms(13.0, 77.6, client=_client(handler)))
    n = len(calls)
    asyncio.run(live_data.fetch_firms(13.0, 77.6, client=_client(handler)))
    assert len(calls) == n


def test_header_only_csv_is_live_with_zero_fires():
    out = asyncio.run(live_data.fetch_firms(13.0, 77.6, client=_client(lambda r: httpx.Response(200, text=HEADER + "\n"))))
    assert out["satellite_status"] == "LIVE" and out["fires"] == []


def test_invalid_key_is_reported_and_key_not_leaked_in_result():
    out = asyncio.run(live_data.fetch_firms(13.0, 77.6, client=_client(lambda r: httpx.Response(200, text="Invalid MAP_KEY."))))
    assert out["satellite_status"] == "INVALID_KEY" and out["fires"] == []
    assert "TESTKEY" not in str(out)


def test_partial_when_one_source_fails():
    def handler(req):
        if "SNPP" in str(req.url): return httpx.Response(500, text="Internal error")
        return httpx.Response(200, text=f"{HEADER}\n{_row(13.05, 77.6)}")
    out = asyncio.run(live_data.fetch_firms(13.0, 77.6, client=_client(handler)))
    assert out["satellite_status"] == "PARTIAL" and len(out["fires"]) == 1


def test_stale_real_data_is_kept_and_flagged_when_refresh_fails():
    asyncio.run(live_data.fetch_firms(13.0, 77.6, client=_client(lambda r: httpx.Response(200, text=f"{HEADER}\n{_row(13.05, 77.6)}"))))
    out = asyncio.run(live_data.fetch_firms(13.0, 77.6, client=_client(lambda r: httpx.Response(503, text="down")), use_cache=False))
    assert out["satellite_status"] == "STALE" and len(out["fires"]) == 2


def test_network_failure_reports_unavailable_not_empty_live():
    def boom(req): raise httpx.ConnectError("no route")
    out = asyncio.run(live_data.fetch_firms(13.0, 77.6, client=_client(boom)))
    assert out["satellite_status"] == "UNAVAILABLE" and out["fires"] == []


def test_elevation_open_meteo_format(monkeypatch):
    # Open-Meteo elevation returns {"elevation": [...]} and takes latitude/longitude params.
    monkeypatch.setattr(settings, "elevation_api_url", "https://api.open-meteo.com/v1/elevation")
    seen = []
    def handler(req: httpx.Request):
        seen.append(str(req.url))
        return httpx.Response(200, json={"elevation": [910.0]})
    out = asyncio.run(live_data.fetch_elevation(13.0, 77.6, client=_client(handler)))
    assert out["terrain_status"] == "LIVE" and out["elevation_m"] == 910.0
    assert "latitude=13.0" in seen[0] and "longitude=77.6" in seen[0]


def test_elevation_open_elevation_legacy_format(monkeypatch):
    # open-elevation.com (legacy) returns {"results": [{"elevation": ...}]} and takes a locations string.
    monkeypatch.setattr(settings, "elevation_api_url", "https://api.open-elevation.com/api/v1/lookup")
    seen = []
    def handler(req: httpx.Request):
        seen.append(str(req.url))
        return httpx.Response(200, json={"results": [{"latitude": 13.0, "longitude": 77.6, "elevation": 920.0}]})
    out = asyncio.run(live_data.fetch_elevation(13.0, 77.6, client=_client(handler)))
    assert out["terrain_status"] == "LIVE" and out["elevation_m"] == 920.0
    assert "locations=13.0%2C77.6" in seen[0]


def test_elevation_down_reports_unavailable_not_fake_value(monkeypatch):
    monkeypatch.setattr(settings, "elevation_api_url", "https://api.open-meteo.com/v1/elevation")
    def boom(req): raise httpx.ConnectError("no route")
    out = asyncio.run(live_data.fetch_elevation(13.0, 77.6, client=_client(boom)))
    assert out["terrain_status"] == "UNAVAILABLE" and out["elevation_m"] is None


def test_open_meteo_visibility_is_reported_with_freshness_and_source():
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M")
    body = {"current": {"time": timestamp, "visibility": 12500}}
    out = asyncio.run(live_data.fetch_visibility(
        13.0,
        77.6,
        client=_client(lambda request: httpx.Response(200, json=body)),
    ))
    assert out["visibility_km"] == 12.5
    assert out["visibility_status"] == "LIVE"
    assert out["visibility_source"] == "OPEN_METEO"


def test_missing_visibility_remains_explicitly_unavailable():
    out = asyncio.run(live_data.fetch_visibility(
        13.0,
        77.6,
        client=_client(lambda request: httpx.Response(200, json={"current": {}})),
    ))
    assert out["visibility_km"] is None
    assert out["visibility_status"] == "UNAVAILABLE"
