from __future__ import annotations

import asyncio
import io
import json
import logging
import time
from collections import deque
from contextlib import asynccontextmanager, suppress
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError

from fireguard.config import settings
from fireguard.database import init_db
from fireguard.environment import (
    DemoProvider,
    MetNoProvider,
    NWSProvider,
    OpenMeteoProvider,
    UnavailableProvider,
)
from fireguard.forecast import wind_advection_forecast
from fireguard.live_data import collect_live_sources, fetch_firms_history
from fireguard.performance import get_inference_performance, record_inference
from fireguard.reporting import (
    extract_dataset_report,
    extract_evaluation_report,
    extract_live_provider_report,
)
from fireguard.risk import calculate_location_risk, calculate_risk
from fireguard.runtime import FireguardRuntime
from utils.device import (
    configure_precision,
    detect_device,
    get_gpu_info,
    get_gpu_memory,
    get_gpu_utilization,
)


def _build_environment_provider():
    if settings.demo_mode:
        return DemoProvider()
    choice = settings.environment_provider.lower()
    if choice in {"nws", "nws_gov", "noaa", "government", "gov"}:
        return NWSProvider(settings.environment_timeout_seconds, settings.nws_user_agent, settings.environment_max_age_seconds)
    if choice in {"met_no", "met-no", "metno", "norway", "live"}:
        return MetNoProvider(settings.environment_timeout_seconds, settings.environment_max_age_seconds, settings.met_no_user_agent)
    if choice in {"open_meteo", "open-meteo"}:
        return OpenMeteoProvider(settings.weather_api_url, settings.environment_timeout_seconds, settings.environment_max_age_seconds)
    return UnavailableProvider()


environment_provider = _build_environment_provider()

# Initialize runtime with resolved weights (custom or default family+size)
_runtime_device = settings.device or detect_device()
runtime = FireguardRuntime(model_path=settings.model_weights, device=_runtime_device)
_local_settings_path = Path(__file__).resolve().parent / "local_settings.json"


class MonitoringLocationUpdate(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    radius_km: float = Field(default=250, gt=0, le=5000)


try:
    if _local_settings_path.exists():
        _saved_site = json.loads(_local_settings_path.read_text(encoding="utf-8"))
        settings.latitude = float(_saved_site["latitude"])
        settings.longitude = float(_saved_site["longitude"])
        settings.firms_radius_km = float(_saved_site.get("radius_km", settings.firms_radius_km))
except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
    logging.getLogger(__name__).warning(
        "Could not load saved monitoring location: %s", exc.__class__.__name__
    )

# Ensure model is loaded on startup (non-blocking, logged)
@asynccontextmanager
async def _model_load_lifespan(app: FastAPI):
    # Model loading happens in background to not block startup
    import asyncio
    asyncio.create_task(_load_model_async())
    yield

async def _load_model_async():
    """Load model asynchronously after startup."""
    try:
        loaded = runtime.load()
        if loaded:
            print(f"[model] Loaded successfully on {_runtime_device}")
        else:
            print(f"[model] Not loaded: {runtime.get_status().get('error', 'unknown')}")
    except Exception as e:
        print(f"[model] Load failed: {e}")

environment_history: deque[dict[str, Any]] = deque(maxlen=288)
risk_history: deque[dict[str, Any]] = deque(maxlen=288)
fires: dict[str, dict[str, Any]] = {}
alerts: dict[str, dict[str, Any]] = {}
live_state: dict[str, Any] = {"timestamp": None, "satellite": {"fires": [], "satellite_status": "NOT_STARTED"}}
connections: list[WebSocket] = []
log = logging.getLogger(__name__)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def broadcast(payload):
    stale = []
    for ws in connections:
        try:
            await ws.send_json(payload)
        except Exception:
            stale.append(ws)
    for ws in stale:
        if ws in connections:
            connections.remove(ws)


async def collect_once():
    lat, lon = settings.latitude, settings.longitude
    try:
        env = await environment_provider.current(lat, lon)
        live = await collect_live_sources(lat, lon)
        env.update(live.get("air_quality", {}))
        visibility = live.get("visibility", {})
        if env.get("visibility") is None:
            env["visibility"] = visibility.get("visibility_km")
            env["visibility_status"] = visibility.get("visibility_status", "UNAVAILABLE")
            env["visibility_source"] = visibility.get("visibility_source", "OPEN_METEO")
            env["visibility_timestamp"] = visibility.get("visibility_timestamp")
        else:
            env["visibility_status"] = env.get("freshness", "UNAVAILABLE")
            env["visibility_source"] = env.get("source", "UNAVAILABLE")
            env["visibility_timestamp"] = env.get("timestamp")
        env["terrain"] = live.get("terrain", {})
        environment_history.append(env)
        location_risk = calculate_location_risk(env)
        if location_risk["status"] == "ESTIMATE":
            risk_history.append({
                "time": env.get("timestamp"),
                "score": location_risk["score"],
                "level": location_risk["level"],
                "source": location_risk["source"],
            })
        live_state.clear()
        live_state.update(live)
        live_state["timestamp"] = now()
        await broadcast({"type": "live_update", "timestamp": live_state["timestamp"], "environment": env, "location_risk": location_risk, "risk_history": list(risk_history)[-48:], "satellite": live.get("satellite", {}), "terrain": live.get("terrain", {})})
    except Exception as exc:
        live_state["error"] = exc.__class__.__name__
        await broadcast({"type": "live_error", "timestamp": now(), "error": exc.__class__.__name__})


async def poller():
    while True:
        await collect_once()
        await asyncio.sleep(max(15, settings.live_poll_seconds))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start model loading and poller
    init_db()
    asyncio.create_task(_load_model_async())
    task = asyncio.create_task(poller())
    yield
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


app = FastAPI(title="EcoSpread-YOLO", version="1.2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=[x.strip() for x in settings.cors_origins.split(",") if x.strip()], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.get("/")
async def root():
    return {"service": "FireGuard Wildfire Intelligence API", "status": "healthy", "docs": "/docs", "location": {"name": "Monitoring site", "latitude": settings.latitude, "longitude": settings.longitude}, "demo_mode": settings.demo_mode}


@app.get("/api/health")
async def health():
    return {"status": "healthy", "demo_mode": settings.demo_mode, "environment_provider": environment_provider.name, "model_status": runtime.get_status()["status"], "live_poll_seconds": settings.live_poll_seconds, "timestamp": now()}


@app.get("/api/system/gpu")
async def system_gpu():
    device = detect_device()
    return {"device": device, "precision": configure_precision(device), "gpu_info": get_gpu_info(), "gpu_memory": get_gpu_memory(), "gpu_utilization": get_gpu_utilization(), "status": "AVAILABLE" if device.startswith("cuda") else "CPU_MODE"}


@app.get("/api/model/status")
async def model_status():
    return runtime.get_status()


@app.post("/api/detection/image")
async def detection_image(file: UploadFile = File(...), camera_id: str | None = None):
    model = runtime.get_model()
    if model is None:
        status = runtime.get_status()
        raise HTTPException(503, detail={"message": "Model not loaded", "model_status": status})

    if not file.filename:
        raise HTTPException(400, "A filename is required.")
    if Path(file.filename).suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        raise HTTPException(400, "Unsupported image type.")

    try:
        from PIL import Image
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        start = time.perf_counter()
        results = model.predict(source=image, device=_runtime_device, conf=settings.confidence_threshold, verbose=False)
        latency = round((time.perf_counter() - start) * 1000, 2)
    except ImportError as exc:
        raise HTTPException(503, f"ML dependencies unavailable: {exc}") from exc
    except Exception as exc:
        raise HTTPException(500, f"Inference failed: {exc.__class__.__name__}") from exc

    detections = []
    event_detections = []
    model_version = runtime.get_status().get("model_path", "unknown")
    for result in results:
        names = getattr(result, "names", {}) or {}
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for i, box in enumerate(boxes):
            cls = int(box.cls.item())
            conf = float(box.conf.item())
            xyxy = [float(x) for x in box.xyxy[0].tolist()]
            area = max(0, xyxy[2] - xyxy[0]) * max(0, xyxy[3] - xyxy[1])
            detections.append({
                "detection_id": f"det-{uuid4().hex}",
                "camera_id": camera_id or settings.camera_id,
                "timestamp": now(),
                "confidence": round(conf, 4),
                "class_name": names.get(cls, str(cls)),
                "bbox": xyxy,
                "visual_area_proxy": round(area, 2),
                "inference_latency_ms": latency,
                "model_version": model_version,
            })
            if str(names.get(cls, "")).strip().lower() in {"fire", "smoke"}:
                event_detections.append(detections[-1])

    event = None
    if event_detections and runtime.get_status().get("fire_smoke_classes_available"):
        strongest = max(event_detections, key=lambda item: item["confidence"])
        environment = environment_history[-1] if environment_history else {}
        risk = calculate_risk({
            "visual": strongest["confidence"] * 100,
            "wind": min(float(environment.get("wind_speed") or 0) * 1.5, 100),
            "temperature": min(max(float(environment.get("temperature") or 0) - 20, 0) * 4, 100),
            "humidity": max(100 - float(environment.get("humidity") or 100), 0),
        })
        event_id = f"event-{int(time.time() * 1000)}"
        event = {
            "event_id": event_id,
            "camera_id": camera_id or settings.camera_id,
            "source_type": "UAV_IMAGE",
            "status": "ACTIVE",
            "timestamp": now(),
            "confidence": strongest["confidence"],
            "visual_area_proxy": strongest["visual_area_proxy"],
            "latitude": None,
            "longitude": None,
            "risk_score": risk["score"],
            "risk_level": risk["level"],
            "growth_rate": None,
            "model_version": model_version,
        }
        fires[event_id] = event
        alert_id = f"alert-{event_id}"
        alerts[alert_id] = {
            "alert_id": alert_id,
            "event_id": event_id,
            "level": risk["level"],
            "title": f"{strongest['class_name'].title()} detection",
            "message": f"{strongest['class_name'].title()} detected with {strongest['confidence']:.0%} model confidence. Location and spread observations are not available from this image.",
            "created_at": event["timestamp"],
            "acknowledged": False,
        }
        for detection in event_detections:
            detection["event_id"] = event_id

    try:
        record_inference(
            camera_id=camera_id or settings.camera_id,
            model_version=str(model_version),
            latency_ms=latency,
            gpu_used=_runtime_device.startswith("cuda"),
            image_width=image.width,
            image_height=image.height,
            detections=detections,
        )
    except SQLAlchemyError as exc:
        log.exception("Could not save inference performance to the database.")
        raise HTTPException(
            503,
            "Inference completed but could not be saved to the performance database.",
        ) from exc

    payload = {"status": "completed", "file": file.filename, "latency_ms": latency, "count": len(detections), "detections": detections, "event": event}
    await broadcast({"type": "detection_update", "payload": payload, "timestamp": now()})
    return payload


@app.get("/api/cameras")
async def cameras():
    return [{"camera_id": settings.camera_id, "name": "Primary UAV", "source_type": "UAV", "url": settings.camera_url, "active": bool(settings.camera_url), "gps_locked": settings.latitude is not None and settings.longitude is not None}]


@app.get("/api/fires")
async def list_fires():
    return list(fires.values())


@app.get("/api/satellite/fires")
async def satellite_fires():
    return live_state.get("satellite", {})


@app.get("/api/satellite/history")
async def satellite_history(
    start_date: date = Query(default_factory=lambda: datetime.now(timezone.utc).date()),
    days: int = Query(default=5, ge=1, le=5),
    limit: int = Query(default=50, ge=1, le=100),
    latitude: float | None = Query(default=None, ge=-90, le=90),
    longitude: float | None = Query(default=None, ge=-180, le=180),
    radius_km: float | None = Query(default=None, gt=0, le=5000),
):
    if start_date > datetime.now(timezone.utc).date():
        raise HTTPException(422, "Historical archive start date must not be in the future.")
    if (latitude is None) != (longitude is None):
        raise HTTPException(422, "Latitude and longitude must be provided together.")
    return await fetch_firms_history(
        start_date,
        days,
        settings.latitude if latitude is None else latitude,
        settings.longitude if longitude is None else longitude,
        limit,
        radius_km,
    )


@app.get("/api/system/location")
async def monitoring_location():
    live_sources = live_state.get("satellite", {})
    return {
        "latitude": settings.latitude,
        "longitude": settings.longitude,
        "radius_km": settings.firms_radius_km,
        "providers": {
            "weather": environment_provider.name,
            "weather_status": environment_history[-1].get("status", "UNAVAILABLE") if environment_history else "WAITING",
            "air_quality": live_state.get("air_quality", {}).get("air_quality_status", "UNAVAILABLE"),
            "terrain": live_state.get("terrain", {}).get("terrain_status", "UNAVAILABLE"),
            "firms": live_sources.get("satellite_status", "UNAVAILABLE"),
            "firms_key_configured": bool(settings.firms_map_key),
            "weather_credentials_required": False,
            "firms_credentials_required": True,
            "secrets_configuration": "Set provider keys in the API server .env file. Keys are never returned to the dashboard.",
        },
    }


@app.put("/api/system/location")
async def update_monitoring_location(location: MonitoringLocationUpdate):
    settings.latitude = location.latitude
    settings.longitude = location.longitude
    settings.firms_radius_km = location.radius_km
    try:
        _local_settings_path.write_text(
            json.dumps(location.model_dump(), indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        log.error("Could not persist monitoring location: %s", exc.__class__.__name__)
        raise HTTPException(500, "Could not save monitoring location on the API server.") from exc
    live_state["timestamp"] = None
    asyncio.create_task(collect_once())
    return {"status": "SAVED", "latitude": location.latitude, "longitude": location.longitude, "radius_km": location.radius_km}


@app.get("/api/fires/{event_id}")
async def fire(event_id: str):
    if event_id not in fires:
        raise HTTPException(404, "Fire event not found.")
    return fires[event_id]


@app.get("/api/forecast/{event_id}")
async def forecast(event_id: str, horizon_minutes: int = Query(30, ge=1, le=360)):
    event = fires.get(event_id)
    if event is None:
        raise HTTPException(404, "Fire event not found.")
    latitude, longitude = event.get("latitude"), event.get("longitude")
    env = environment_history[-1] if environment_history else {}
    wind_speed, wind_direction = env.get("wind_speed"), env.get("wind_direction")
    if latitude is None or longitude is None:
        raise HTTPException(409, "A geolocated fire event is required for a forecast.")
    if wind_speed is None or wind_direction is None:
        raise HTTPException(409, "Current wind speed and direction are required for a forecast.")
    prediction = wind_advection_forecast(
        float(latitude),
        float(longitude),
        float(wind_speed),
        float(wind_direction),
        horizon_minutes,
    )
    return {"event_id": event_id, **prediction, "input_timestamp": env.get("timestamp")}


@app.get("/api/environment/current")
async def environment_current(latitude: float | None = None, longitude: float | None = None):
    if latitude is None:
        latitude = settings.latitude
    if longitude is None:
        longitude = settings.longitude
    env = await environment_provider.current(latitude, longitude)
    extra = await collect_live_sources(latitude, longitude, include_satellite=False)
    env.update(extra.get("air_quality", {}))
    visibility = extra.get("visibility", {})
    if env.get("visibility") is None:
        env["visibility"] = visibility.get("visibility_km")
        env["visibility_status"] = visibility.get("visibility_status", "UNAVAILABLE")
        env["visibility_source"] = visibility.get("visibility_source", "OPEN_METEO")
        env["visibility_timestamp"] = visibility.get("visibility_timestamp")
    else:
        env["visibility_status"] = env.get("freshness", "UNAVAILABLE")
        env["visibility_source"] = env.get("source", "UNAVAILABLE")
        env["visibility_timestamp"] = env.get("timestamp")
    env["terrain"] = extra.get("terrain", {})
    return env


@app.get("/api/risk/location")
async def location_risk(
    latitude: float | None = Query(default=None, ge=-90, le=90),
    longitude: float | None = Query(default=None, ge=-180, le=180),
):
    if (latitude is None) != (longitude is None):
        raise HTTPException(422, "Latitude and longitude must be provided together.")
    latitude = settings.latitude if latitude is None else latitude
    longitude = settings.longitude if longitude is None else longitude
    environment = await environment_provider.current(latitude, longitude)
    return {
        **calculate_location_risk(environment),
        "latitude": latitude,
        "longitude": longitude,
    }


@app.get("/api/environment/history")
async def environment_history_endpoint():
    return list(environment_history)


@app.get("/api/live/status")
async def live_status():
    return {"timestamp": live_state.get("timestamp"), "location": {"latitude": settings.latitude, "longitude": settings.longitude}, "sources": {"weather": environment_provider.name, "visibility": live_state.get("visibility", {}).get("visibility_status", "UNAVAILABLE"), "air_quality": live_state.get("air_quality", {}).get("air_quality_status", "UNAVAILABLE"), "terrain": live_state.get("terrain", {}).get("terrain_status", "UNAVAILABLE"), "satellite": live_state.get("satellite", {}).get("satellite_status", "UNAVAILABLE")}, "config": {"firms_key_configured": bool(settings.firms_map_key), "firms_sources": [x.strip() for x in settings.firms_source.split(",") if x.strip()], "firms_radius_km": settings.firms_radius_km, "weather_provider": environment_provider.name, "demo_mode": settings.demo_mode}}


@app.get("/api/alerts")
async def list_alerts():
    return list(alerts.values())


@app.post("/api/alerts/{alert_id}/acknowledge")
async def acknowledge(alert_id: str):
    if alert_id not in alerts:
        raise HTTPException(404, "Alert not found.")
    alerts[alert_id]["acknowledged"] = True
    alerts[alert_id]["acknowledged_at"] = now()
    return alerts[alert_id]


@app.get("/api/analytics/overview")
async def analytics():
    active = [f for f in fires.values() if f.get("status") == "ACTIVE"]
    score = max((f.get("risk_score", 0) for f in active), default=0)
    runtime_metrics = get_inference_performance()
    return {
        "total_events": len(fires),
        "active_events": len(active),
        "alert_count": len(alerts),
        "satellite_events": len(live_state.get("satellite", {}).get("fires", [])),
        "risk_level": "CRITICAL" if score >= 80 else "HIGH" if score >= 60 else "MODERATE" if score >= 40 else "LOW",
        "source_status": environment_provider.name,
        "model_metrics": {
            "runtime": runtime_metrics,
            "validation": extract_evaluation_report(),
        },
        "dataset_report": extract_dataset_report(),
        "provider_check": extract_live_provider_report(),
        "timestamp": now(),
    }


@app.get("/api/dashboard/summary")
async def summary():
    if not live_state.get("timestamp"):
        await collect_once()
    env = environment_history[-1] if environment_history else await environment_provider.current(settings.latitude, settings.longitude)
    gpu = await system_gpu()
    model = runtime.get_status()
    stats = await analytics()
    sat = live_state.get("satellite", {})
    return {"timestamp": now(), "health": await health(), "environment": env, "location_risk": calculate_location_risk(env), "environment_history": list(environment_history)[-48:], "fires": list(fires.values()), "satellite_fires": sat.get("fires", []), "satellite_meta": {k: v for k, v in sat.items() if k != "fires"}, "alerts": list(alerts.values()), "analytics": stats, "gpu": gpu, "model": model, "live": await live_status(), "providers": {"satellite": sat.get("satellite_status", "UNAVAILABLE"), "terrain": live_state.get("terrain", {}).get("terrain_status", "UNAVAILABLE"), "vegetation": "UNAVAILABLE"}, "risk_history": list(risk_history)[-48:]}


@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    await websocket.accept()
    connections.append(websocket)
    current_environment = environment_history[-1] if environment_history else None
    await websocket.send_json({"type": "live_update", "timestamp": now(), "environment": current_environment, "location_risk": calculate_location_risk(current_environment or {}), "risk_history": list(risk_history)[-48:], "satellite": live_state.get("satellite", {}), "terrain": live_state.get("terrain", {})})
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in connections:
            connections.remove(websocket)