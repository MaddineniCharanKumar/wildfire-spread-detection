"""Weather providers. Every provider returns real observations or an explicit UNAVAILABLE payload;
none ever invents a reading. DemoProvider exists only for opt-in demos and is always labeled."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

logger = logging.getLogger(__name__)
DEFAULT_MAX_AGE_S = 1800


def _now() -> datetime:
    return datetime.now(timezone.utc)


def classify_age(timestamp: str | None, max_age_seconds: int = DEFAULT_MAX_AGE_S) -> tuple[str, float | None]:
    """Freshness from the observation's own timestamp: LIVE <= max_age, RECENT <= 6x, else STALE."""
    if not timestamp:
        return "UNAVAILABLE", None
    try:
        dt = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
    except ValueError:
        return "UNAVAILABLE", None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    age = max(0.0, (_now() - dt).total_seconds())
    label = "LIVE" if age <= max_age_seconds else "RECENT" if age <= max_age_seconds * 6 else "STALE"
    return label, round(age, 1)


class EnvironmentalProvider:
    name = "UNAVAILABLE"
    async def current(self, latitude=None, longitude=None): raise NotImplementedError


def base(latitude, longitude):
    return {"temperature": None, "humidity": None, "wind_speed": None, "wind_direction": None, "rainfall": None, "pressure": None,
            "pm25": None, "pm10": None, "visibility": None, "source": "UNAVAILABLE", "timestamp": _now().isoformat(), "demo": False,
            "freshness": "UNAVAILABLE", "status": "UNAVAILABLE", "location_available": latitude is not None and longitude is not None}


class UnavailableProvider(EnvironmentalProvider):
    async def current(self, latitude=None, longitude=None):
        p = base(latitude, longitude); p["message"] = "No environmental provider is configured."; return p


class DemoProvider(EnvironmentalProvider):
    """Opt-in (FIREGUARD_DEMO_MODE=true) fixed values for UI demos. Always labeled; never used by default."""
    name = "DEMO"
    async def current(self, latitude=None, longitude=None):
        return {"temperature": 32.5, "humidity": 31.0, "wind_speed": 24.0, "wind_direction": 210.0, "rainfall": 0.0, "pressure": 1012.0, "pm25": 18.5, "pm10": 25.0, "visibility": 8.5,
                "source": "DEMO", "timestamp": _now().isoformat(), "demo": True, "freshness": "DEMO", "status": "DEMO_MODE",
                "message": "Demo data is active. This is not live monitoring data.", "location_available": latitude is not None and longitude is not None}


class OpenMeteoProvider(EnvironmentalProvider):
    """api.open-meteo.com - global coverage, no key. Aggregates national weather services' models/observations."""
    name = "OPEN_METEO"
    def __init__(self, url="https://api.open-meteo.com/v1/forecast", timeout_seconds=10, max_age_seconds=DEFAULT_MAX_AGE_S, transport=None):
        self.url = url; self.timeout = httpx.Timeout(timeout_seconds); self.max_age = max_age_seconds; self.transport = transport

    async def current(self, latitude=None, longitude=None):
        if latitude is None or longitude is None:
            p = base(latitude, longitude); p["source"] = self.name; p["message"] = "Configure FIREGUARD_LATITUDE and FIREGUARD_LONGITUDE for live weather."; return p
        try:
            async with httpx.AsyncClient(timeout=self.timeout, transport=self.transport) as c:
                r = await c.get(self.url, params={"latitude": latitude, "longitude": longitude, "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m,surface_pressure,visibility", "timezone": "UTC"})
                r.raise_for_status(); d = r.json()
            cur = d.get("current", {}); t = cur.get("time")
            if not t: raise ValueError("missing observation time")
            freshness, age = classify_age(t, self.max_age)
            return {"temperature": cur.get("temperature_2m"), "humidity": cur.get("relative_humidity_2m"), "wind_speed": cur.get("wind_speed_10m"), "wind_direction": cur.get("wind_direction_10m"),
                    "rainfall": cur.get("precipitation"), "pressure": cur.get("surface_pressure"), "pm25": None, "pm10": None,
                    "visibility": cur.get("visibility") / 1000 if cur.get("visibility") is not None else None,
                    "source": self.name, "timestamp": t, "age_seconds": age, "demo": False, "freshness": freshness, "status": freshness, "location_available": True, "units": d.get("current_units", {})}
        except Exception as e:
            logger.warning("weather failed: %s", e.__class__.__name__)
            p = base(latitude, longitude); p["source"] = self.name; p["message"] = f"Live weather unavailable: {e.__class__.__name__}"; return p


class MetNoProvider(EnvironmentalProvider):
    """Live global weather from MET Norway's Locationforecast service. No API key.

    MET Norway asks clients to identify themselves with a User-Agent. This provider
    is used for the Bengaluru deployment so the dashboard does not depend on an
    Open-Meteo request quota.
    """
    name = "MET_NO"

    def __init__(self, timeout_seconds=10, max_age_seconds=DEFAULT_MAX_AGE_S, user_agent="FireGuard-Bengaluru/1.0 local", transport=None):
        self.timeout = httpx.Timeout(timeout_seconds)
        self.max_age = max_age_seconds
        self.user_agent = user_agent
        self.transport = transport

    async def current(self, latitude=None, longitude=None):
        if latitude is None or longitude is None:
            p = base(latitude, longitude); p["source"] = self.name
            p["message"] = "Configure FIREGUARD_LATITUDE and FIREGUARD_LONGITUDE for live weather."
            return p
        try:
            headers = {"User-Agent": self.user_agent, "Accept": "application/json"}
            url = "https://api.met.no/weatherapi/locationforecast/2.0/compact"
            async with httpx.AsyncClient(timeout=self.timeout, transport=self.transport, follow_redirects=True) as c:
                r = await c.get(url, params={"lat": latitude, "lon": longitude}, headers=headers)
                r.raise_for_status(); d = r.json()
            series = d.get("properties", {}).get("timeseries", [])
            if not series:
                raise ValueError("missing forecast timeseries")
            current = series[0]
            details = current.get("data", {}).get("instant", {}).get("details", {})
            next_hour = current.get("data", {}).get("next_1_hours", {}).get("details", {})
            t = current.get("time")
            if not t: raise ValueError("missing observation time")
            freshness, age = classify_age(t, self.max_age)
            return {
                "temperature": details.get("air_temperature"),
                "humidity": details.get("relative_humidity"),
                "wind_speed": details.get("wind_speed"),
                "wind_direction": details.get("wind_from_direction"),
                "rainfall": next_hour.get("precipitation_amount"),
                "pressure": details.get("air_pressure_at_sea_level"),
                "pm25": None, "pm10": None,
                "visibility": details.get("visibility") / 1000 if details.get("visibility") is not None else None,
                "source": self.name, "timestamp": t, "age_seconds": age, "demo": False,
                "freshness": freshness, "status": freshness, "location_available": True,
                "attribution": "MET Norway Locationforecast",
            }
        except Exception as e:
            logger.warning("MET Norway weather failed: %s", e.__class__.__name__)
            p = base(latitude, longitude); p["source"] = self.name
            p["message"] = f"Live weather unavailable: {e.__class__.__name__}"
            return p


class NWSProvider(EnvironmentalProvider):
    """Live weather straight from the U.S. National Weather Service (api.weather.gov, NOAA), no key.
    U.S. territory only: elsewhere NWS has no station, and this provider reports UNAVAILABLE."""
    name = "NWS_GOV"
    STATIONS_TRIED = 3

    def __init__(self, timeout_seconds=10, user_agent="EcoSpread-YOLO (set FIREGUARD_NWS_USER_AGENT)", max_age_seconds=DEFAULT_MAX_AGE_S, transport=None):
        self.timeout = httpx.Timeout(timeout_seconds); self.max_age = max_age_seconds; self.transport = transport
        self.headers = {"User-Agent": user_agent, "Accept": "application/geo+json"}
        self._station_cache: dict[tuple[float, float], list[str]] = {}

    async def _station_urls(self, c, latitude, longitude):
        # NWS /points wants <= 4 decimals and redirects otherwise.
        lat, lon = round(latitude, 4), round(longitude, 4)
        key = (round(lat, 2), round(lon, 2))
        if key in self._station_cache: return self._station_cache[key]
        r = await c.get(f"https://api.weather.gov/points/{lat},{lon}", headers=self.headers); r.raise_for_status()
        stations_url = r.json().get("properties", {}).get("observationStations")
        if not stations_url: raise ValueError("no observation stations for this location")
        r2 = await c.get(stations_url, headers=self.headers); r2.raise_for_status()
        ids = [f["properties"]["stationIdentifier"] for f in r2.json().get("features", [])][: self.STATIONS_TRIED]
        if not ids: raise ValueError("no nearby NWS station")
        urls = [f"https://api.weather.gov/stations/{i}/observations/latest" for i in ids]
        self._station_cache[key] = urls; return urls

    async def current(self, latitude=None, longitude=None):
        if latitude is None or longitude is None:
            p = base(latitude, longitude); p["source"] = self.name; p["message"] = "Configure FIREGUARD_LATITUDE and FIREGUARD_LONGITUDE for live NWS weather."; return p
        try:
            async with httpx.AsyncClient(timeout=self.timeout, transport=self.transport, follow_redirects=True) as c:
                props = None
                for url in await self._station_urls(c, latitude, longitude):
                    r = await c.get(url, headers=self.headers)
                    if r.status_code >= 400: continue
                    cand = r.json().get("properties", {})
                    if (cand.get("temperature") or {}).get("value") is not None:  # skip stations reporting nothing
                        props = cand; break
                if props is None: raise ValueError("no NWS station returned a usable observation")
            t = props.get("timestamp")
            if not t: raise ValueError("missing observation time")
            val = lambda f: (props.get(f) or {}).get("value")
            pa, vis = val("barometricPressure"), val("visibility")
            freshness, age = classify_age(t, self.max_age)
            return {"temperature": val("temperature"), "humidity": val("relativeHumidity"), "wind_speed": val("windSpeed"), "wind_direction": val("windDirection"), "rainfall": val("precipitationLastHour"),
                    "pressure": pa / 100.0 if pa is not None else None, "pm25": None, "pm10": None, "visibility": vis / 1000.0 if vis is not None else None,
                    "source": self.name, "timestamp": t, "age_seconds": age, "demo": False, "freshness": freshness, "status": freshness, "location_available": True,
                    "station": props.get("station"), "attribution": "U.S. National Weather Service (NOAA), api.weather.gov"}
        except Exception as e:
            logger.warning("NWS weather failed: %s", e.__class__.__name__)
            p = base(latitude, longitude); p["source"] = self.name; p["message"] = f"Live NWS weather unavailable: {e.__class__.__name__}"; return p


class WeatherAPIProvider(OpenMeteoProvider): name = "OPEN_METEO"
class SensorProvider(UnavailableProvider): name = "SENSOR_PROVIDER"
class CSVProvider(UnavailableProvider): name = "CSV_PROVIDER"
class IoTSensorProvider(UnavailableProvider): name = "IOT_SENSOR_PROVIDER"
class AirQualityProvider(UnavailableProvider): name = "AIR_QUALITY_PROVIDER"
