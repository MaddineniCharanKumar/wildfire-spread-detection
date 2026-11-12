# EcoSpread-YOLO

EcoSpread-YOLO is a research prototype for real-time UAV wildfire detection and predictive wildfire intelligence. It combines a lightweight YOLO-based detector, environmental telemetry, geospatial context, and short-horizon spread forecasting in a modular backend architecture.

## Mission

Move from reactive visual fire detection toward predictive wildfire intelligence by integrating:
- YOLO fire/smoke detection
- UAV geolocation and telemetry
- live or recent environmental data
- satellite corroboration
- terrain and vegetation context
- risk scoring and alerting
- short-horizon spread forecasts

## Core requirements enforced by this project

- No fabricated detections, fire events, GPS, weather values, or evaluation metrics.
- Demo-mode must be explicit and clearly labeled.
- Environmental data must preserve freshness metadata and source information.
- If a provider is missing, the system must report UNAVAILABLE instead of inventing data.
- All model and evaluation claims must be based on actual runs, not placeholders.
- The code must remain modular and testable.

## Repository structure

- `fireguard/` — backend, configuration, runtime, and service logic
- `training/` — dataset validation, training, evaluation, and export scripts
- `scripts/` — dataset preparation and inspection helpers
- `dashboard/` — frontend for map, alert, risk, and telemetry visualization
- `docs/` — architecture and operational notes
- `tests/` — unit and API tests

## Local development

1. Create a virtual environment
2. Install dependencies
3. Copy `.env.example` to `.env`
4. (Optional) Set model weights or use defaults
5. Start the API

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[ml,dev]"
cp .env.example .env
# Optional: set custom weights or use defaults (yolo26n.pt)
# FIREGUARD_MODEL_WEIGHTS=/path/to/best.pt
uvicorn fireguard.api:app --reload --host 0.0.0.0 --port 8000
```

API docs:
- http://localhost:8000/docs

### Windows PowerShell: run the API and dashboard

Run the API and dashboard in **separate terminals**. Start the API from the project directory so it loads that directory's `.env` file:

```powershell
cd .\wildfire-project-bengaluru
.\.venv\Scripts\python.exe -m uvicorn fireguard.api:app --reload --host 127.0.0.1 --port 8000
```

In a second terminal, from the workspace directory:

```powershell
cd .\wildfire-project-bengaluru\dashboard
npm run dev
```

Open the Vite URL printed in the second terminal. If Vite reports `ECONNREFUSED 127.0.0.1:8000`, the API terminal is not running or failed to start; check its output and confirm `http://localhost:8000/docs` opens.

## Key endpoints

- GET `/api/health`
- GET `/api/system/gpu`
- GET `/api/model/status`
- POST `/api/detection/image`
- POST `/api/detection/video`
- GET `/api/cameras`
- POST `/api/camera/{id}/start`
- POST `/api/camera/{id}/stop`
- GET `/api/fires`
- GET `/api/fires/{id}`
- GET `/api/environment/current`
- GET `/api/environment/history`
- GET `/api/risk/location?latitude=12.97&longitude=77.59`
- GET `/api/risk/{event_id}`
- GET `/api/alerts`
- POST `/api/alerts/{id}/acknowledge`
- GET `/api/analytics/overview`
- GET `/api/forecast/{event_id}?horizon_minutes=30`
- GET `/api/satellite/history?start_date=YYYY-MM-DD&days=5&limit=50&latitude=12.97&longitude=77.59&radius_km=25`
- GET/PUT `/api/system/location` (monitoring location is saved on the API host)
- WebSocket `/ws/live`

## Model Configuration

The detector uses Ultralytics YOLO. Configure via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `FIREGUARD_MODEL_FAMILY` | `yolo26` | Model family: `yolo26` (current) or `yolo27` (when released). |
| `FIREGUARD_MODEL_SIZE` | `n` | Model size: `n`, `s`, `m`, `l`, `x`. |
| `FIREGUARD_MODEL_WEIGHTS` | *(empty)* | Optional path to custom trained weights (`.pt`). If set, takes priority over family+size. |

**Default weight resolution**: If `FIREGUARD_MODEL_WEIGHTS` is not set, the loader builds the weight filename as `{family}{size}.pt` (e.g., `yolo26n.pt`). Ultralytics will auto-download the official weights on first use.

**YOLO27 readiness**: When Ultralytics releases YOLO27, set `FIREGUARD_MODEL_FAMILY=yolo27` and `FIREGUARD_MODEL_SIZE` to your desired size. No code changes required. **Note**: YOLO27 has not been released or tested as of this writing.

### Minimum Ultralytics Version

`ultralytics>=8.3` is required for YOLOv8/YOLO11/YOLO26 compatibility. The `pyproject.toml` pins this minimum.

### GPU / CPU

- `FIREGUARD_DEVICE=cpu` (default) or `cuda` / `cuda:0` etc.
- On CUDA devices, mixed precision (FP16) is enabled automatically.
- The API degrades gracefully to CPU if CUDA is unavailable.

## Data and evaluation policy

- Dataset inspection must happen before training or metric reporting.
- No model metrics may be reported without real validation data and actual execution.
- The dashboard omits model-accuracy metrics because no validated wildfire evaluation is configured. Runtime latency and single-image throughput are calculated only from completed inference requests, and appear only when actual samples exist.
- Dataset inspection reports object counts from YOLO, Pascal VOC, and COCO annotations. If annotation formats overlap, it selects one canonical format rather than summing duplicates.
- `training/evaluate.py` writes measured precision, recall, F1, mAP, per-class results, and validation FPS to `reports/eval/metrics.json` and `reports/latest_evaluation.json`; unsupported or unavailable metrics remain `null`, never zero placeholders.
- `scripts/verify_live_sources.py` persists its sanitized provider results to `reports/live_provider_report.json`.
- Do not parse bounding-box area as precise physical fire area unless calibration supports it.
- Keep research modules isolated and configurable.

See [dataset preparation and evaluation](docs/dataset.md) for the full inspect → prepare → train → evaluate workflow. The included pretrained COCO weights are not fire/smoke weights; a real image dataset and a trained checkpoint are required for valid precision/recall/mAP/FPS results.

## Dashboard configuration and local inference

- Set or change the monitoring coordinates and radius from the Command Center, Live Map, or **Monitoring Site** page. The API host persists only the coordinates and FIRMS radius in `fireguard/local_settings.json`; provider secrets are not exposed to the browser.
- Configure `FIREGUARD_FIRMS_MAP_KEY` in the API server `.env` for NASA FIRMS current/historical detection requests. Historical Data queries NASA's date-specific Standard Processing area API for a user-selected window of 1–5 days at the selected site/radius. Results are source satellite records, not verified incident history.
- NASA GIBS map layers and MODIS NDVI imagery are keyless. MET Norway, Open-Meteo weather/air quality, and Open-Meteo elevation do not require API keys.
- A local `yolo26n.pt` initializes the YOLO26 runtime when no custom `FIREGUARD_MODEL_WEIGHTS` is configured. The included weights were inspected and contain COCO classes, not fire/smoke classes. They are useful for checking runtime/inference but **are not a wildfire detector**. Configure fire/smoke-trained weights before enabling operational wildfire alerts. Evaluation values remain unreported until those weights are validated on labeled data.
- The main dashboard monitors a selected geographic area; it does not display an implied UAV/camera feed.
- The location risk is a transparent weather-only heuristic index, not a calibrated fire-occurrence probability. The index is withheld when live/recent weather is unavailable or insufficient. Validation/accuracy metrics are not shown without a real labeled evaluation report.

## Deployment note

The project supports GPU acceleration when available, but it must continue safely on CPU or with unavailable providers. It must degrade gracefully and clearly report missing resources instead of generating fake results.

### Deploy the dashboard to Vercel

The Vercel project should serve the dashboard only. The FastAPI backend runs a continuous live-data poller and WebSocket endpoint, so deploy it separately to an always-on host that supports WebSockets and persistent storage.

1. Import this repository into Vercel and set **Root Directory** to `dashboard`.
2. Keep the framework preset as **Vite**. The build command is `npm run build` and the output directory is `dist`.
3. Add the `VITE_API_BASE_URL` environment variable in Vercel, set to the backend's public HTTPS origin with no trailing slash (for example, `https://your-api-host.example.com`), then redeploy.
4. On the backend host, set `FIREGUARD_CORS_ORIGINS` to the exact deployed dashboard origin (for example, `https://your-project.vercel.app`). Configure the live-data variables and secrets from `.env.example` there, not in Vercel.
5. Confirm the backend's `/api/health` endpoint is reachable, then open the Vercel deployment. The dashboard uses the API origin for both HTTP requests and its secure WebSocket connection.

`dashboard/vercel.json` rewrites client-side routes to the Vite entry page, so direct navigation to dashboard pages works. The backend's default SQLite database and saved monitoring location are local files; configure persistent storage or a managed database on the backend host if data must survive redeployments. Deploying the backend as a Vercel function is not supported by this setup because its poller, WebSocket service, and local persistence require a long-running process.


## Realtime live-data configuration

EcoSpread-YOLO now polls live external sources and pushes updates to the dashboard over WebSocket.

### 1. Set the monitoring location

Copy `.env.example` to `.env` and set:

```env
FIREGUARD_LATITUDE=<monitoring latitude>
FIREGUARD_LONGITUDE=<monitoring longitude>
FIREGUARD_LIVE_POLL_SECONDS=60
```

### 2. Live weather and air quality

Three real, non-synthetic weather providers are available — pick one with `FIREGUARD_ENVIRONMENT_PROVIDER`:

```env
# Option A (this distribution) — global coverage, no API key, no quota.
# MET Norway Locationforecast (api.met.no). Set a contact User-Agent:
FIREGUARD_ENVIRONMENT_PROVIDER=met_no
FIREGUARD_MET_NO_USER_AGENT=FireGuard-Bengaluru/1.0 (you@example.com)

# Option B — global coverage, aggregates national meteorological agencies
# (NOAA/NWS, DWD, ECMWF, MeteoFrance and others), no API key required.
FIREGUARD_ENVIRONMENT_PROVIDER=open_meteo
FIREGUARD_WEATHER_API_URL=https://api.open-meteo.com/v1/forecast
FIREGUARD_AIR_QUALITY_API_URL=https://air-quality-api.open-meteo.com/v1/air-quality

# Option C — direct from the U.S. National Weather Service / NOAA's own federal API
# (api.weather.gov). U.S. locations only, no API key, but requires a contact User-Agent.
FIREGUARD_ENVIRONMENT_PROVIDER=nws
FIREGUARD_NWS_USER_AGENT=EcoSpread-YOLO (you@example.com)
```

Air-quality readings (PM2.5/PM10) always come from Open-Meteo's air-quality API regardless of which weather provider is selected, since NWS does not publish air quality. Elevation/terrain comes from Open-Meteo's keyless elevation API:

```env
FIREGUARD_ELEVATION_API_URL=https://api.open-meteo.com/v1/elevation
FIREGUARD_TERRAIN_PROVIDER=open_meteo_elevation
```

### 3. NASA FIRMS satellite fire detections

Create a free NASA FIRMS MAP_KEY and set:

```env
FIREGUARD_FIRMS_MAP_KEY=<your-map-key>
FIREGUARD_FIRMS_SOURCE=VIIRS_NOAA21_NRT
FIREGUARD_FIRMS_REGION=world
FIREGUARD_FIRMS_DAYS=1
```

The backend exposes satellite detections at `/api/satellite/fires` and includes them in the live map.

### 4. Live map (NASA GIBS map)

The dashboard renders a real NASA GIBS map `hybrid` map on the Live Map and Spread Prediction pages — the fire event, FIRMS hotspots, and wind vector are drawn from the live data above; no placeholder or synthetic map/pins are used. Copy `dashboard/.env.example` to `dashboard/.env` and set:

```env
VITE_API_BASE_URL=http://localhost:8000
VITE_MAP_LATITUDE=<default map center latitude, used only when no fire event exists yet>
VITE_MAP_LONGITUDE=<default map center longitude>
VITE_MAP_LOCATION_LABEL=Configured monitoring area
```


### 5. Run backend

```bash
uvicorn fireguard.api:app --host 0.0.0.0 --port 8000
```

### 6. Run dashboard

```bash
cd dashboard
npm install
npm run dev
```

The dashboard connects to `/ws/live` and receives live environment, satellite, terrain and detection updates. The REST summary is refreshed every 60 seconds as a fallback.

If a provider is not configured or fails, the UI shows `UNAVAILABLE`/`CONFIG_REQUIRED` rather than inventing measurements.
FIRMS detections require a valid `FIREGUARD_FIRMS_MAP_KEY` in the API `.env`; a valid query can still return no hotspots if NASA has no detections in the selected area and date range. Precision, recall, F1, mAP, and validation FPS require an actual fire/smoke model evaluation and are intentionally unavailable until one is completed.
Completed image inferences and their predicted bounding boxes are stored in the configured SQLModel database (`fireguard.db` by default) and summarized on the Analytics page. Stored prediction counts and measured inference throughput are operational statistics, not accuracy metrics; calculating precision, recall, F1, and mAP still requires labeled validation data.


## Live data setup (real sources only)

1. `cp .env.example .env`; set `FIREGUARD_LATITUDE`, `FIREGUARD_LONGITUDE` and `FIREGUARD_FIRMS_MAP_KEY` (free NASA key).
2. Weather: `met_no` (global, no key) or `open_meteo` (global, no key); `nws` is U.S.-only (UNAVAILABLE elsewhere). Air quality and elevation: Open-Meteo.
3. Hotspots: NASA FIRMS, bounding box around your site, 3 VIIRS satellites, cached.
4. Verify against the real servers from your machine: `python scripts/verify_live_sources.py`
5. Run `uvicorn fireguard.api:app` and `cd dashboard && npm install && npm run dev`.

Demo mode is off by default and labelled DEMO everywhere if enabled.

## Bengaluru live configuration

This distribution is configured for **Bengaluru, Karnataka** (12.9716° N, 77.5946° E) with `FIREGUARD_DEMO_MODE=false` and **MET Norway** (`met_no`) as the live weather provider. Weather (temperature, humidity, wind, pressure, rainfall), air quality (PM2.5/PM10) and terrain elevation are pulled live from real keyless APIs — verified:

```text
PASS  MET Norway weather   RECENT T=22.1C wind=6.4km/h
PASS  Open-Meteo weather   LIVE T=22.0C wind=17.3km/h
PASS  Air quality          PM2.5=8.8
PASS  Elevation            910.0 m
SKIP  NWS (U.S. only)      UNAVAILABLE (US-only service)
```

Run `python scripts/verify_live_sources.py` any time to re-check every live source against the real upstream servers. No Google Maps key is required. NASA GIBS imagery is keyless. NASA FIRMS hotspots remain optional until a FIRMS MAP_KEY is configured (`FIREGUARD_FIRMS_MAP_KEY`, free from https://firms.modaps.eosdis.nasa.gov/api/area/).

Run the API on port 8000 and the Vite dashboard on port 5173. The dashboard uses a formal light operational theme.
