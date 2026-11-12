from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class DetectionCreate(BaseModel):
    camera_id: str
    source_type: str = "IMAGE"
    confidence: float = 0.0
    class_name: str = "fire"
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0
    visual_area_proxy: Optional[float] = None


class DetectionRead(BaseModel):
    detection_id: str
    event_id: Optional[str]
    camera_id: str
    timestamp: datetime
    confidence: float
    class_name: str
    visual_area_proxy: Optional[float]
    model_version: Optional[str]


class EnvironmentalSnapshot(BaseModel):
    timestamp: datetime
    source: str
    freshness: str
    status: str
    temperature_c: Optional[float] = None
    humidity_pct: Optional[float] = None
    wind_speed_kmh: Optional[float] = None
    wind_direction_deg: Optional[float] = None
    rainfall_mm: Optional[float] = None
    pressure_hpa: Optional[float] = None
    pm25: Optional[float] = None
    pm10: Optional[float] = None
    visibility_km: Optional[float] = None
    is_demo: bool = False


class FireEventRead(BaseModel):
    event_id: str
    camera_id: str
    status: str
    latitude: Optional[float]
    longitude: Optional[float]
    confidence: float
    risk_score: float
    growth_rate: float
    spread_direction: str
    visual_area_proxy: Optional[float]
    created_at: datetime
    updated_at: datetime


class ForecastRead(BaseModel):
    forecast_id: str
    event_id: str
    horizon_minutes: int
    forecast_time: datetime
    predicted_polygon: str
    confidence: Optional[float]
    uncertainty: Optional[float]


class RiskRead(BaseModel):
    event_id: str
    risk_score: float
    level: str
    explanation: str
    contributing_factors: str
    timestamp: datetime


class AlertRead(BaseModel):
    alert_id: str
    event_id: str
    title: str
    message: str
    level: str
    acknowledged: bool
    created_at: datetime


class HealthRead(BaseModel):
    status: str
    demo_mode: bool
    environment_provider: str
    model_status: str
    timestamp: datetime


class GPURead(BaseModel):
    device: str
    precision: str
    cuda_available: bool
    gpu_memory_mb: Optional[int]
    gpu_utilization_pct: Optional[float]
    temperature_c: Optional[float]
    status: str
