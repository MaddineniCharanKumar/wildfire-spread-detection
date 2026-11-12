from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="FIREGUARD_", extra="ignore")

    # App
    app_name: str = "EcoSpread-YOLO"
    demo_mode: bool = False
    database_url: str = "sqlite:///./fireguard.db"
    device: str = "cpu"
    dataset_path: str = ""
    cors_origins: str = "http://localhost:5173"

    # Model configuration (YOLO26 default, YOLO27-ready via config switch)
    model_family: str = "yolo26"          # yolo26 | yolo27 (when released)
    model_size: str = "n"                 # n, s, m, l, x
    model_weights: str | None = None      # Optional: path to custom trained weights (takes priority)

    # Detection
    confidence_threshold: float = 0.35
    temporal_window: int = 12
    minimum_persistence: int = 3
    camera_frame_skip: int = 0

    # Location & live polling
    latitude: float | None = None
    longitude: float | None = None
    live_poll_seconds: int = 60

    # Weather / environment
    environment_provider: str = "met_no"
    weather_api_url: str = "https://api.open-meteo.com/v1/forecast"
    air_quality_api_url: str = "https://air-quality-api.open-meteo.com/v1/air-quality"
    weather_api_key: str | None = None
    met_no_user_agent: str = "FireGuard-Bengaluru/1.0 local"
    environment_max_age_seconds: int = 1800
    environment_timeout_seconds: float = 10.0
    nws_user_agent: str = "EcoSpread-YOLO (set FIREGUARD_NWS_USER_AGENT to your contact email)"

    # Alerting
    alert_cooldown_seconds: int = 60
    upload_max_mb: int = 100

    # Camera
    camera_url: str | None = None
    camera_id: str = "default-camera"

    # NASA FIRMS
    firms_map_key: str | None = None
    firms_source: str = "VIIRS_NOAA21_NRT,VIIRS_NOAA20_NRT,VIIRS_SNPP_NRT"
    firms_region: str = "world"
    firms_days: int = 1
    firms_radius_km: float = 250.0
    firms_poll_seconds: int = 600
    satellite_provider: str = "firms"

    # Terrain / elevation
    elevation_api_url: str = "https://api.open-meteo.com/v1/elevation"
    terrain_provider: str = "open_meteo_elevation"

    # Vegetation
    vegetation_api_url: str | None = None
    vegetation_provider: str = "none"

    # Risk / forecasting
    alert_threshold: float = 0.6
    forecast_horizons: str = "30,60"

    # Corroboration (Step 4)
    corroboration_firms_radius_km: float = 25.0
    corroboration_firms_time_window_hours: int = 6
    corroboration_humidity_threshold: float = 70.0  # % above which smoke detections are down-weighted

    # Dynamic threshold (Step 5)
    dynamic_threshold_enabled: bool = True
    dynamic_threshold_min: float = 0.25
    dynamic_threshold_max: float = 0.60

    # Smoke exposure (Step 6)
    smoke_spread_enabled: bool = True
    smoke_spread_horizons_min: str = "30,60"


settings = Settings()