from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin


@dataclass
class GeolocationRecord:
    latitude: float | None = None
    longitude: float | None = None
    altitude_m: float | None = None
    source: str = "UNAVAILABLE"
    estimated: bool = False
    status: str = "MISSING"
    confidence: float | None = None

    def to_dict(self) -> dict[str, float | str | bool | None]:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude_m": self.altitude_m,
            "source": self.source,
            "estimated": self.estimated,
            "status": self.status,
            "confidence": self.confidence,
        }


class GeolocationService:
    """Basic geolocation utilities for UAV or camera-derived estimates."""

    @staticmethod
    def from_uav(latitude: float | None, longitude: float | None, altitude_m: float | None = None, source: str = "UAV") -> GeolocationRecord:
        if latitude is None or longitude is None:
            return GeolocationRecord(latitude=None, longitude=None, altitude_m=altitude_m, source=source, estimated=False, status="MISSING")
        return GeolocationRecord(
            latitude=latitude,
            longitude=longitude,
            altitude_m=altitude_m,
            source=source,
            estimated=False,
            status="CONFIRMED",
            confidence=1.0,
        )

    @staticmethod
    def estimate_from_image(center_x: float, center_y: float, latitude: float | None, longitude: float | None, altitude_m: float | None = None, heading_deg: float = 0.0) -> GeolocationRecord:
        if latitude is None or longitude is None:
            return GeolocationRecord(latitude=None, longitude=None, altitude_m=altitude_m, source="IMAGE_PROJECTION", estimated=True, status="ESTIMATED")

        # Lightweight projection for a prototype system; it is a placeholder estimate and should not be treated as ground truth.
        earth_radius_m = 6_371_000.0
        heading = radians(heading_deg)
        north_m = (center_y - 0.5) * 200.0
        east_m = (center_x - 0.5) * 200.0
        lat_delta = (north_m * cos(heading) + east_m * sin(heading)) / earth_radius_m
        lon_delta = (east_m * cos(heading) - north_m * sin(heading)) / (earth_radius_m * cos(radians(latitude)))
        return GeolocationRecord(
            latitude=latitude + lat_delta * 180.0 / 3.141592653589793,
            longitude=longitude + lon_delta * 180.0 / 3.141592653589793,
            altitude_m=altitude_m,
            source="IMAGE_PROJECTION",
            estimated=True,
            status="ESTIMATED",
            confidence=0.5,
        )
