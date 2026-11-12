from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class CameraSource(SQLModel, table=True):
    __tablename__ = "camera_sources"

    id: Optional[int] = Field(default=None, primary_key=True)
    camera_id: str = Field(index=True, unique=True)
    name: str
    source_type: str = "RTSP"
    url: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude_m: Optional[float] = None
    active: bool = True
    gps_locked: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class SensorSource(SQLModel, table=True):
    __tablename__ = "sensor_sources"

    id: Optional[int] = Field(default=None, primary_key=True)
    source_id: str = Field(index=True, unique=True)
    name: str
    provider: str = "OPEN_METEO"
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Detection(SQLModel, table=True):
    __tablename__ = "detections"

    id: Optional[int] = Field(default=None, primary_key=True)
    detection_id: str = Field(index=True, unique=True)
    event_id: Optional[str] = Field(default=None, index=True)
    camera_id: str = Field(index=True)
    source_type: str = "IMAGE"
    timestamp: datetime = Field(default_factory=datetime.utcnow, index=True)
    confidence: float = 0.0
    class_name: str = "fire"
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0
    visual_area_proxy: Optional[float] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    inference_latency_ms: Optional[float] = None
    model_version: Optional[str] = None
    is_valid: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Track(SQLModel, table=True):
    __tablename__ = "tracks"

    id: Optional[int] = Field(default=None, primary_key=True)
    track_id: str = Field(index=True, unique=True)
    camera_id: str = Field(index=True)
    event_id: Optional[str] = Field(default=None, index=True)
    first_seen: datetime = Field(default_factory=datetime.utcnow)
    last_seen: datetime = Field(default_factory=datetime.utcnow)
    persistence_count: int = 0
    confidence_mean: float = 0.0
    growth_rate: float = 0.0
    status: str = "ACTIVE"
    created_at: datetime = Field(default_factory=datetime.utcnow)


class FireEvent(SQLModel, table=True):
    __tablename__ = "fire_events"

    id: Optional[int] = Field(default=None, primary_key=True)
    event_id: str = Field(index=True, unique=True)
    camera_id: str = Field(index=True)
    source_type: str = "IMAGE"
    status: str = "ACTIVE"
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude_m: Optional[float] = None
    confidence: float = 0.0
    risk_score: float = 0.0
    growth_rate: float = 0.0
    spread_direction: str = "UNKNOWN"
    visual_area_proxy: Optional[float] = None
    current_temperature: Optional[float] = None
    current_humidity: Optional[float] = None
    current_wind_speed: Optional[float] = None
    current_wind_direction: Optional[float] = None
    model_version: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class FireForecast(SQLModel, table=True):
    __tablename__ = "fire_forecasts"

    id: Optional[int] = Field(default=None, primary_key=True)
    forecast_id: str = Field(index=True, unique=True)
    event_id: str = Field(index=True)
    horizon_minutes: int = 30
    forecast_time: datetime = Field(default_factory=datetime.utcnow, index=True)
    input_observation_time: datetime = Field(default_factory=datetime.utcnow)
    model_version: str = "unknown"
    predicted_polygon: str = ""
    confidence: Optional[float] = None
    uncertainty: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class EnvironmentalReading(SQLModel, table=True):
    __tablename__ = "environmental_readings"

    id: Optional[int] = Field(default=None, primary_key=True)
    reading_id: str = Field(index=True, unique=True)
    camera_id: Optional[str] = Field(default=None, index=True)
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow, index=True)
    temperature_c: Optional[float] = None
    humidity_pct: Optional[float] = None
    wind_speed_kmh: Optional[float] = None
    wind_direction_deg: Optional[float] = None
    rainfall_mm: Optional[float] = None
    pressure_hpa: Optional[float] = None
    pm25: Optional[float] = None
    pm10: Optional[float] = None
    visibility_km: Optional[float] = None
    source: str = "UNAVAILABLE"
    freshness: str = "UNAVAILABLE"
    status: str = "UNAVAILABLE"
    is_demo: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)


class RiskScore(SQLModel, table=True):
    __tablename__ = "risk_scores"

    id: Optional[int] = Field(default=None, primary_key=True)
    event_id: str = Field(index=True)
    timestamp: datetime = Field(default_factory=datetime.utcnow, index=True)
    risk_score: float = 0.0
    level: str = "LOW"
    explanation: str = ""
    contributing_factors: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Alert(SQLModel, table=True):
    __tablename__ = "alerts"

    id: Optional[int] = Field(default=None, primary_key=True)
    alert_id: str = Field(index=True, unique=True)
    event_id: str = Field(index=True)
    level: str = "INFO"
    title: str
    message: str
    reason: str = ""
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ModelVersion(SQLModel, table=True):
    __tablename__ = "model_versions"

    id: Optional[int] = Field(default=None, primary_key=True)
    version: str = Field(index=True, unique=True)
    model_type: str = "YOLO"
    architecture: str = "fire-detection"
    mAP50: Optional[float] = None
    mAP50_95: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    fps: Optional[float] = None
    latency_ms: Optional[float] = None
    parameters_m: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class InferenceLog(SQLModel, table=True):
    __tablename__ = "inference_logs"

    id: Optional[int] = Field(default=None, primary_key=True)
    camera_id: str = Field(index=True)
    model_version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow, index=True)
    inference_latency_ms: float = 0.0
    fps: float = 0.0
    gpu_used: bool = False
    status: str = "OK"
    created_at: datetime = Field(default_factory=datetime.utcnow)


# Re-export model loader functions for backward compatibility
from fireguard.model_loader import (
    _build_weight_name,
    resolve_model_weights,
    load_yolo_model,
    get_model_status,
)