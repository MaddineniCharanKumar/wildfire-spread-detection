from __future__ import annotations

from math import asin, atan2, cos, isfinite, pi, radians, sin


def wind_advection_forecast(
    latitude: float,
    longitude: float,
    wind_speed_kmh: float,
    wind_from_degrees: float,
    horizon_minutes: int,
) -> dict[str, float | int | str]:
    """Project a point using current wind only; this is not a fire-perimeter model."""
    if not all(isfinite(value) for value in (latitude, longitude, wind_speed_kmh, wind_from_degrees)):
        raise ValueError("forecast inputs must be finite numbers")
    if wind_speed_kmh < 0:
        raise ValueError("wind speed cannot be negative")
    if not -90 <= latitude <= 90:
        raise ValueError("latitude must be between -90 and 90 degrees")
    if not -180 <= longitude <= 180:
        raise ValueError("longitude must be between -180 and 180 degrees")

    earth_radius_km = 6371.0088
    distance_km = wind_speed_kmh * horizon_minutes / 60
    bearing = radians((wind_from_degrees + 180) % 360)
    lat1, lon1 = radians(latitude), radians(longitude)
    angular_distance = distance_km / earth_radius_km
    lat2 = asin(
        sin(lat1) * cos(angular_distance)
        + cos(lat1) * sin(angular_distance) * cos(bearing)
    )
    lon2 = lon1 + atan2(
        sin(bearing) * sin(angular_distance) * cos(lat1),
        cos(angular_distance) - sin(lat1) * sin(lat2),
    )
    projected_longitude = (lon2 + pi) % (2 * pi) - pi

    return {
        "model": "wind_advection_baseline_v1",
        "horizon_minutes": horizon_minutes,
        "origin_latitude": latitude,
        "origin_longitude": longitude,
        "projected_latitude": lat2 * 180 / pi,
        "projected_longitude": projected_longitude * 180 / pi,
        "wind_speed_kmh": wind_speed_kmh,
        "wind_from_degrees": wind_from_degrees % 360,
        "drift_km": round(distance_km, 3),
        "status": "ESTIMATE",
        "limitations": "Point advection from current wind; no validated perimeter, fuel, slope, or future weather model.",
    }
