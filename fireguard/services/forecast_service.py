from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from sqlmodel import Session

from fireguard.models import FireForecast


class ForecastService:
    def __init__(self, session: Session):
        self.session = session

    def create_forecast(self, event_id: str, horizon_minutes: int, predicted_polygon: str) -> FireForecast:
        forecast = FireForecast(
            forecast_id=f"fc-{uuid4().hex[:12]}",
            event_id=event_id,
            horizon_minutes=horizon_minutes,
            forecast_time=datetime.utcnow(),
            input_observation_time=datetime.utcnow(),
            model_version="physics-hybrid-v1",
            predicted_polygon=predicted_polygon,
            confidence=0.8,
            uncertainty=0.2,
        )
        self.session.add(forecast)
        self.session.commit()
        self.session.refresh(forecast)
        return forecast
