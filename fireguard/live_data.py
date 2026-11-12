"""Live, real-data sources: NASA FIRMS (satellite hotspots), Open-Meteo air quality and elevation.

Nothing in this module ever fabricates a value. When a source cannot be reached, is not configured,
or rejects the request, the result carries an explicit status and an empty payload.
"""
from __future__ import annotations

import csv
import io
import logging
import time
from datetime import date, datetime, timezone
from math import asin, cos, isfinite, radians, sin, sqrt
from typing import Any

import httpx

from fireguard.config import settings
from fireguard.environment import classify_age

log = logging.getLogger(__name__)

FIRMS_AREA_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
FIRMS_MAX_DAYS = 5          # documented range: 1..5
FIRMS_MAX_RETURNED = 500    # cap what is pushed to the dashboard
FIRMS_STALE_AFTER_S = 3 * 3600  # keep last good FIRMS data (flagged STALE) for at most 3 h if a refresh fails

_firms_cache: dict[str, Any] = {}


def ts() -> str:
    return datetime.now(timezone.utc).isoformat()


class FirmsError(Exception):
    """Raised when FIRMS answers with something that is not a CSV of detections."""


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * 6371.0088 * asin(sqrt(a))


def firms_bbox(lat: float, lon: float, radius_km: float) -> tuple[float, float, float, float]:
    """(west, south, east, north) box around a point. Clamped to valid ranges (no antimeridian wrap)."""
    dlat = radius_km / 111.32
    dlon = radius_km / (111.32 * max(cos(radians(lat)), 0.01))
    return (max(-180.0, lon - dlon), max(-90.0, lat - dlat), min(180.0, lon + dlon), min(90.0, lat + dlat))


def _observed_at(acq_date: str | None, acq_time: str | None) -> str | None:
    """FIRMS acq_date is YYYY-MM-DD and acq_time is HHMM, both UTC."""
    try:
        hhmm = str(acq_time or "").zfill(4)
        return datetime.strptime(f"{acq_date} {hhmm}", "%Y-%m-%d %H%M").replace(tzinfo=timezone.utc).isoformat()
    except (TypeError, ValueError):
        return None


def parse_firms_csv(text: str, source: str, lat: float | None = None, lon: float | None = None) -> list[dict[str, Any]]:
    """Parse a FIRMS area CSV. Raises FirmsError if the body is not a detections CSV
    (FIRMS reports bad keys and quota problems as plain text, sometimes with HTTP 200)."""
    body = (text or "").strip()
    if not body:
        raise FirmsError("empty response from FIRMS")
    first = body.splitlines()[0].lower()
    if "latitude" not in first or "longitude" not in first:
        raise FirmsError(body[:160])
    now = datetime.now(timezone.utc)
    fires: list[dict[str, Any]] = []
    for row in csv.DictReader(io.StringIO(body)):
        try:
            la, lo = float(row["latitude"]), float(row["longitude"])
        except (KeyError, TypeError, ValueError):
            continue
        observed = _observed_at(row.get("acq_date"), row.get("acq_time"))
        age_min = round((now - datetime.fromisoformat(observed)).total_seconds() / 60, 1) if observed else None
        fires.append({
            "event_id": f"firms-{source}-{row.get('acq_date')}-{row.get('acq_time')}-{la:.4f}-{lo:.4f}",
            "latitude": la, "longitude": lo,
            "confidence": row.get("confidence"),
            "frp": row.get("frp"),
            "brightness": row.get("bright_ti4") or row.get("brightness"),
            "daynight": row.get("daynight"),
            "satellite": row.get("satellite"),
            "instrument": row.get("instrument"),
            "acq_date": row.get("acq_date"), "acq_time": row.get("acq_time"),
            "observed_at": observed, "age_minutes": age_min,
            "distance_km": round(haversine_km(lat, lon, la, lo), 1) if lat is not None and lon is not None else None,
            "firms_source": source,
            "source": "NASA_FIRMS", "status": "SATELLITE_OBSERVATION",
        })
    return fires


def _firms_sources() -> list[str]:
    return [s.strip() for s in settings.firms_source.split(",") if s.strip()]


async def fetch_firms_history(
    start_date: date,
    days: int,
    latitude: float | None,
    longitude: float | None,
    limit: int = 6,
    radius_km: float | None = None,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Fetch an explicitly dated NASA FIRMS Standard Processing archive window."""
    if not settings.firms_map_key:
        return {
            "status": "CONFIG_REQUIRED",
            "source": "NASA_FIRMS",
            "records": [],
            "message": "Configure FIREGUARD_FIRMS_MAP_KEY on the API server to query the NASA archive.",
        }

    search_radius_km = radius_km or settings.firms_radius_km
    if latitude is not None and longitude is not None:
        area = ",".join(f"{value:.4f}" for value in firms_bbox(latitude, longitude, search_radius_km))
    else:
        area = settings.firms_region

    sources = [
        source.replace("_NRT", "_SP")
        for source in _firms_sources()
    ] or ["VIIRS_NOAA21_SP", "VIIRS_NOAA20_SP", "VIIRS_SNPP_SP"]
    unique_sources = list(dict.fromkeys(sources))
    detections: dict[str, dict[str, Any]] = {}
    source_status: dict[str, str] = {}

    own_client = client is None
    client = client or httpx.AsyncClient(timeout=settings.environment_timeout_seconds)
    try:
        for source in unique_sources:
            url = f"{FIRMS_AREA_URL}/{settings.firms_map_key}/{source}/{area}/{days}/{start_date:%Y-%m-%d}"
            try:
                response = await client.get(url)
                response.raise_for_status()
                for detection in parse_firms_csv(response.text, source, latitude, longitude):
                    if (
                        latitude is not None
                        and longitude is not None
                        and detection["distance_km"] > search_radius_km
                    ):
                        continue
                    detections[detection["event_id"]] = detection
                source_status[source] = "AVAILABLE"
            except FirmsError as exc:
                source_status[source] = _classify_firms_error(str(exc))
                log.warning("FIRMS archive %s response was not usable", source)
            except httpx.HTTPStatusError as exc:
                source_status[source] = "RATE_LIMITED" if exc.response.status_code == 429 else "UNAVAILABLE"
                log.warning("FIRMS archive %s returned HTTP %s", source, exc.response.status_code)
            except httpx.HTTPError as exc:
                source_status[source] = "UNAVAILABLE"
                log.warning("FIRMS archive %s request failed: %s", source, exc.__class__.__name__)
    finally:
        if own_client:
            await client.aclose()

    records = sorted(
        detections.values(),
        key=lambda item: item.get("distance_km")
        if item.get("distance_km") is not None
        else float("inf"),
    )
    records.sort(key=lambda item: item.get("observed_at") or "", reverse=True)
    selected = records[:limit]
    frp_values = []
    for record in selected:
        try:
            value = float(record.get("frp"))
        except (TypeError, ValueError):
            continue
        if value >= 0:
            frp_values.append(value)
    confidence_high = sum(
        str(record.get("confidence", "")).strip().lower() in {"h", "high"}
        or _is_numeric_high_confidence(record.get("confidence"))
        for record in selected
    )
    status = "AVAILABLE" if any(value == "AVAILABLE" for value in source_status.values()) else "UNAVAILABLE"
    return {
        "status": status,
        "source": "NASA_FIRMS",
        "window_start": f"{start_date:%Y-%m-%d}",
        "window_days": days,
        "record_count": len(records),
        "records": selected,
        "sources": source_status,
        "metrics": {
            "returned_records": len(selected),
            "high_confidence_records": confidence_high,
            "mean_frp_mw": round(sum(frp_values) / len(frp_values), 2) if frp_values else None,
            "frp_sample_count": len(frp_values),
            "metric_scope": "Descriptive satellite observations only; not model accuracy or verified incident counts.",
        },
        "timestamp": ts(),
    }


def _is_numeric_high_confidence(value: Any) -> bool:
    try:
        return float(value) >= 80
    except (TypeError, ValueError):
        return False


def _classify_firms_error(message: str) -> str:
    m = message.lower()
    if "invalid" in m and "key" in m:
        return "INVALID_KEY"
    if "exceed" in m or "limit" in m or "too many" in m:
        return "RATE_LIMITED"
    return "UNAVAILABLE"


async def fetch_firms(lat: float | None = None, lon: float | None = None, client: httpx.AsyncClient | None = None, use_cache: bool = True) -> dict[str, Any]:
    if not settings.firms_map_key:
        return {"fires": [], "satellite_status": "CONFIG_REQUIRED", "satellite_source": "NASA_FIRMS",
                "satellite_message": "Set FIREGUARD_FIRMS_MAP_KEY (free key: https://firms.modaps.eosdis.nasa.gov/api/area/).",
                "satellite_timestamp": ts()}

    if lat is not None and lon is not None:
        area = ",".join(f"{v:.4f}" for v in firms_bbox(lat, lon, settings.firms_radius_km))
        scope = {"bbox": area, "radius_km": settings.firms_radius_km}
    else:
        area = settings.firms_region
        scope = {"bbox": area, "radius_km": None}
    days = min(max(int(settings.firms_days), 1), FIRMS_MAX_DAYS)
    sources = _firms_sources()
    cache_key = f"{area}|{days}|{','.join(sources)}"

    cached = _firms_cache.get("value") if _firms_cache.get("key") == cache_key else None
    if use_cache and cached and time.monotonic() - _firms_cache["at"] < max(30, settings.firms_poll_seconds):
        return cached

    own = client is None
    client = client or httpx.AsyncClient(timeout=settings.environment_timeout_seconds)
    per_source: dict[str, dict[str, Any]] = {}
    merged: dict[str, dict[str, Any]] = {}
    try:
        for src in sources:
            url = f"{FIRMS_AREA_URL}/{settings.firms_map_key}/{src}/{area}/{days}"
            try:
                r = await client.get(url)
                text = r.text
                if r.status_code >= 400 and "latitude" not in text[:200].lower():
                    raise FirmsError(text.strip()[:160] or f"HTTP {r.status_code}")
                found = parse_firms_csv(text, src, lat, lon)
                in_radius = [
                    f for f in found
                    if lat is None or lon is None or f["distance_km"] <= settings.firms_radius_km
                ]
                per_source[src] = {"status": "LIVE", "count": len(in_radius)}
                for f in in_radius:
                    merged[f["event_id"]] = f
            except FirmsError as e:
                # never log the URL: it contains the MAP_KEY
                per_source[src] = {"status": _classify_firms_error(str(e)), "message": str(e)[:160]}
            except Exception as e:  # network/timeout/etc.
                per_source[src] = {"status": "UNAVAILABLE", "message": e.__class__.__name__}
    finally:
        if own:
            await client.aclose()

    ok = [s for s, v in per_source.items() if v["status"] == "LIVE"]
    fires = sorted(merged.values(), key=lambda f: (f["distance_km"] if f["distance_km"] is not None else 0, f.get("age_minutes") or 0))
    total = len(fires)
    fires = fires[:FIRMS_MAX_RETURNED]

    if ok:
        status = "LIVE" if len(ok) == len(sources) else "PARTIAL"
        result = {"fires": fires, "satellite_status": status, "satellite_source": "NASA_FIRMS", "satellite_timestamp": ts(),
                  "satellite_message": None if status == "LIVE" else "Some FIRMS sources failed; see per_source.",
                  "sources": sources, "per_source": per_source, "total_detections": total, "days": days, **scope}
        _firms_cache.update(key=cache_key, at=time.monotonic(), value=result)
        return result

    # Every source failed. Prefer real-but-older data (clearly flagged) over erasing what we know.
    worst = next(iter(per_source.values()), {"status": "UNAVAILABLE", "message": "no sources configured"})
    status = worst["status"]
    log.warning("FIRMS unavailable: %s", {k: v["status"] for k, v in per_source.items()})
    if cached and time.monotonic() - _firms_cache["at"] < FIRMS_STALE_AFTER_S and status != "INVALID_KEY":
        return {**cached, "satellite_status": "STALE", "satellite_message": f"Refresh failed ({status}); showing last successful FIRMS data.", "per_source": per_source}
    return {"fires": [], "satellite_status": status, "satellite_source": "NASA_FIRMS", "satellite_timestamp": ts(),
            "satellite_message": worst.get("message"), "sources": sources, "per_source": per_source, **scope}


async def fetch_air_quality(lat: float, lon: float):
    try:
        async with httpx.AsyncClient(timeout=settings.environment_timeout_seconds) as c:
            r = await c.get(settings.air_quality_api_url, params={"latitude": lat, "longitude": lon, "current": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide", "timezone": "UTC"}); r.raise_for_status(); cur = r.json().get("current", {})
        return {"pm25": cur.get("pm2_5"), "pm10": cur.get("pm10"), "air_quality_status": "LIVE", "air_quality_source": "OPEN_METEO_AIR_QUALITY", "air_quality_timestamp": cur.get("time") or ts()}
    except Exception as e:
        log.warning("air quality failed: %s", e.__class__.__name__); return {"pm25": None, "pm10": None, "air_quality_status": "UNAVAILABLE", "air_quality_source": "OPEN_METEO_AIR_QUALITY", "air_quality_timestamp": ts()}


async def fetch_elevation(lat: float, lon: float, client: httpx.AsyncClient | None = None):
    url = settings.elevation_api_url
    source = "OPEN_METEO_ELEVATION" if "open-meteo.com" in url else ("OPEN_ELEVATION" if "open-elevation.com" in url else "ELEVATION_API")
    try:
        # Open-Meteo elevation takes latitude/longitude; open-elevation takes a locations string.
        params = {"latitude": lat, "longitude": lon} if "open-meteo.com" in url else {"locations": f"{lat},{lon}"}
        own = client is None
        client = client or httpx.AsyncClient(timeout=settings.environment_timeout_seconds)
        try:
            r = await client.get(url, params=params); r.raise_for_status(); d = r.json()
        finally:
            if own:
                await client.aclose()
        # Open-Meteo returns {"elevation": [910.0]}; open-elevation returns {"results": [{"elevation": 910.0}]}.
        if isinstance(d.get("elevation"), list) and d["elevation"]:
            e = d["elevation"][0]
        else:
            results = d.get("results", [])
            e = results[0].get("elevation") if results else None
        return {"elevation_m": e, "terrain_status": "LIVE" if e is not None else "UNAVAILABLE", "terrain_source": source, "terrain_timestamp": ts()}
    except Exception as ex:
        log.warning("elevation failed: %s", ex.__class__.__name__); return {"elevation_m": None, "terrain_status": "UNAVAILABLE", "terrain_source": source, "terrain_timestamp": ts()}


async def fetch_visibility(
    lat: float,
    lon: float,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Use Open-Meteo's current visibility when the selected weather provider lacks it."""
    own = client is None
    client = client or httpx.AsyncClient(timeout=settings.environment_timeout_seconds)
    try:
        response = await client.get(
            settings.weather_api_url,
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "visibility",
                "timezone": "UTC",
            },
        )
        response.raise_for_status()
        payload = response.json()
        current = payload.get("current", {}) if isinstance(payload, dict) else {}
        visibility_m = current.get("visibility")
        timestamp = current.get("time")
        if visibility_m is None or not timestamp:
            raise ValueError("missing current visibility measurement")
        visibility_km = float(visibility_m) / 1000
        if not isfinite(visibility_km) or visibility_km < 0:
            raise ValueError("invalid current visibility measurement")
        freshness, age = classify_age(timestamp, settings.environment_max_age_seconds)
        return {
            "visibility_km": round(visibility_km, 3),
            "visibility_status": freshness,
            "visibility_source": "OPEN_METEO",
            "visibility_timestamp": timestamp,
            "visibility_age_seconds": age,
        }
    except (httpx.HTTPError, TypeError, ValueError) as exc:
        log.warning("visibility unavailable: %s", exc.__class__.__name__)
        return {
            "visibility_km": None,
            "visibility_status": "UNAVAILABLE",
            "visibility_source": "OPEN_METEO",
            "visibility_timestamp": None,
        }
    finally:
        if own:
            await client.aclose()


async def collect_live_sources(lat, lon, include_satellite: bool = True):
    result: dict[str, Any] = {"timestamp": ts()}
    if include_satellite:
        result["satellite"] = await fetch_firms(lat, lon)
    if lat is not None and lon is not None:
        result["air_quality"] = await fetch_air_quality(lat, lon)
        result["terrain"] = await fetch_elevation(lat, lon)
        result["visibility"] = await fetch_visibility(lat, lon)
    return result
