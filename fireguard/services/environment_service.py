from __future__ import annotations

from datetime import datetime
from typing import Any


class EnvironmentalService:
    def __init__(self, provider: Any):
        self.provider = provider

    def classify_freshness(self, timestamp: datetime, max_age_seconds: int = 300) -> str:
        age = (datetime.utcnow() - timestamp).total_seconds()
        if age <= 60:
            return "LIVE"
        if age <= max_age_seconds:
            return "RECENT"
        return "STALE"

    def normalize_snapshot(self, raw: dict[str, Any]) -> dict[str, Any]:
        return {
            "timestamp": raw.get("timestamp", datetime.utcnow()),
            "source": raw.get("source", "UNAVAILABLE"),
            "freshness": raw.get("freshness", "UNAVAILABLE"),
            "status": raw.get("status", "UNAVAILABLE"),
            "temperature_c": raw.get("temperature_c"),
            "humidity_pct": raw.get("humidity_pct"),
            "wind_speed_kmh": raw.get("wind_speed_kmh"),
            "wind_direction_deg": raw.get("wind_direction_deg"),
            "rainfall_mm": raw.get("rainfall_mm"),
            "pressure_hpa": raw.get("pressure_hpa"),
            "pm25": raw.get("pm25"),
            "pm10": raw.get("pm10"),
            "visibility_km": raw.get("visibility_km"),
            "is_demo": bool(raw.get("is_demo", False)),
        }
