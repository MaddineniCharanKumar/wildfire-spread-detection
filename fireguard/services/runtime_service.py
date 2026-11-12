from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fireguard.services import (
    AlertService,
    DetectionService,
    EnvironmentalService,
    ForecastService,
    GeolocationService,
    RiskService,
)

from fireguard.config import settings
from fireguard.environment import DemoProvider, UnavailableProvider, WeatherAPIProvider


class EcoSpreadRuntime:
    def __init__(self) -> None:
        self.detection_service = DetectionService(model_path=settings.model_weights, device=settings.device)
        self.risk_service = RiskService()
        self.alert_service = AlertService()
        self.geolocation_service = GeolocationService()
        if settings.demo_mode:
            self.environment_service = EnvironmentalService(DemoProvider())
        elif settings.environment_provider.lower() == "weather":
            self.environment_service = EnvironmentalService(WeatherAPIProvider())
        else:
            self.environment_service = EnvironmentalService(UnavailableProvider())
        self.forecast_service = ForecastService()

    async def fetch_environment(self, latitude: float | None = None, longitude: float | None = None) -> dict[str, Any]:
        return await self.environment_service.fetch(latitude, longitude)

    def compute_risk(self, features: dict[str, Any]) -> dict[str, Any]:
        return self.risk_service.compute(features)

    def estimate_location(self, center_x: float, center_y: float, latitude: float | None, longitude: float | None, altitude_m: float | None = None, heading_deg: float = 0.0) -> dict[str, Any]:
        return self.geolocation_service.estimate_from_image(center_x, center_y, latitude, longitude, altitude_m, heading_deg).to_dict()

    def predict_spread(self, **kwargs: Any) -> dict[str, Any]:
        return self.forecast_service.predict(**kwargs)

    def get_health(self) -> dict[str, Any]:
        return {
            "status": "healthy",
            "demo_mode": settings.demo_mode,
            "model_configured": bool(settings.model_weights),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
