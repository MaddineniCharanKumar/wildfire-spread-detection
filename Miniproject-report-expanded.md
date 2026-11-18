# EcoSpread-YOLO: UAV Wildfire Detection and Predictive Wildfire Intelligence

**Mini-project report**  
**Repository:** [MaddineniCharanKumar/wildfire-spread-detection](https://github.com/MaddineniCharanKumar/wildfire-spread-detection)  
**Report version:** Expanded project report  
**Prepared:** 18 November 2026

> **Scope and evidence note.** This report describes the implementation currently present in the repository. It does not claim that the default detector has been trained or validated for wildfire use. The repository states that its available default YOLO weights are COCO weights, while the fire/smoke dataset images are not included. No model accuracy values are reported here because no verified fire/smoke evaluation run and report were available in the repository at the time this report was prepared. [1][2]

## Abstract

Wildfires can be monitored using visual detection, environmental observations, and satellite records. EcoSpread-YOLO is a research prototype that combines these sources in a Python FastAPI backend and a React dashboard. Its intended workflow includes image-based object detection with Ultralytics YOLO, location-based weather and NASA FIRMS satellite data, transparent heuristic risk indices, alert handling, and a short-horizon wind-advection estimate. The prototype emphasizes data provenance and explicitly displays unavailable or stale measurements instead of substituting invented readings.

The configured detector defaults to the `yolo26n.pt` model name. This is a general pretrained starting point, not, by itself, a wildfire detector. A custom model trained on labeled fire/smoke images is required before interpreting its predictions as wildfire detections. The repository supplies training and evaluation scripts and a `fire`/`smoke` dataset class configuration, but users must supply and prepare the actual dataset. Consequently, this report evaluates the implemented system workflow and describes a reproducible evaluation plan; it does not invent performance results.

**Keywords:** wildfire monitoring, object detection, YOLO, environmental data, NASA FIRMS, FastAPI, React, risk index, wind advection

## 1. Introduction

Wildfire monitoring can benefit from combining observations that differ in coverage and update frequency. Camera or UAV images may provide local visual evidence; weather observations describe environmental conditions; satellite products can provide broader-area hotspot records. A system that presents these inputs together can support situational awareness, but its estimates should be distinguished from verified incidents or operational forecasts.

EcoSpread-YOLO is designed as a modular research prototype for this kind of multi-source workflow. The repository describes a pipeline involving a source adapter, YOLO adapter, tracking and temporal verification, event management, environmental providers, risk/early warning, and alerting. Its dashboard includes a command center, live map, event list, environmental view, spread-estimate view, historical satellite data, alerts, analytics, data-source status, and monitoring-site configuration. [1][3]

### 1.1 Objectives

- Provide an API for image inference and wildfire-monitoring data.
- Visualize available environmental and satellite data together with source and freshness metadata.
- Present risk as an explainable heuristic index rather than a calibrated probability.
- Provide a simple downwind point-advection estimate where event coordinates and wind inputs exist.
- Support user-supplied fire/smoke data, training, and evaluation without reporting unsupported metrics.
- Display unavailable, stale, or configuration-required data honestly.

### 1.2 Project scope and limitations

This is a research prototype, not a certified wildfire-warning or firefighting decision system. A bounding box is not a calibrated physical fire area; visual motion is not a measured propagation rate; a weather-risk index is not the probability of fire; and the wind-advection baseline is not a predicted fire perimeter. Satellite hotspot records also are not, by themselves, independently verified wildfire incidents. [2][4]

## 2. Existing and proposed approaches

### 2.1 Existing/general approach

A basic image-only detection workflow applies an object detector to an image and returns class labels, confidence scores, and bounding boxes. It can be useful for screening imagery, but it does not inherently establish a fire incident, provide location or weather context, corroborate with satellite data, or describe the freshness and provenance of external observations. General YOLO literature describes single-stage object detection; the cited YOLO paper is background on the detector family and is not a paper validating this repository's configured YOLO26 weights for wildfire detection. [5]

### 2.2 Proposed EcoSpread-YOLO approach

The proposed project workflow integrates multiple independently sourced components:

1. **Visual inference:** Ultralytics YOLO processes a submitted image and returns predicted classes, confidence values, bounding boxes, and measured request latency.
2. **Wildfire event handling:** The backend creates fire-related events and alerts only when model classes include `fire` or `smoke`; the runtime exposes whether these classes are present.
3. **Environmental context:** Configurable providers supply weather, air-quality, visibility, and elevation/terrain data where available.
4. **Satellite context:** NASA FIRMS records can be queried for an area and date range when its API key is configured; NASA GIBS imagery is also presented in the map workflow.
5. **Risk summary:** A transparent feature-weighted event risk score and a separate weather-only location index are computed and labeled as estimates.
6. **Spread baseline:** Given a geolocated event and usable wind observations, the API projects a point downwind over a selected time horizon.
7. **Presentation:** The frontend combines REST summaries with `/ws/live` updates and displays missing or stale input states.

### 2.3 Qualitative comparison

| Capability | Basic image-only detector | EcoSpread-YOLO proposed workflow |
|---|---|---|
| Image object predictions | Yes, if model and dependencies load | Yes, through the image-inference API |
| Weather and air-quality context | Not inherent | Integrated through configured external providers |
| Satellite hotspot context | Not inherent | NASA FIRMS integration when configured; status is reported when unavailable |
| Location-based weather-risk index | Not inherent | Available when sufficiently fresh weather factors exist; heuristic only |
| Spread estimate | Not inherent | Wind-only point advection; not a fire-perimeter model |
| Live dashboard and alerts | Usually outside an image-only detector | React/Vite dashboard, API, and WebSocket updates |
| Wildfire accuracy evidence | Requires labeled wildfire evaluation | Still requires a fire/smoke-trained checkpoint and actual held-out evaluation |

This comparison is architectural and qualitative; it is not a numerical benchmark against another implemented product. No relative accuracy or speed advantage is claimed without comparable data, hardware, thresholds, and measured runs.

## 3. System architecture and implementation

### 3.1 Technology stack

- **Backend:** Python 3.10+; FastAPI and Uvicorn.
- **ML/runtime:** Ultralytics YOLO with optional PyTorch and OpenCV dependencies.
- **Data/configuration:** Pydantic Settings, SQLModel, SQLite by default, NumPy, Pillow, and PyYAML.
- **Frontend:** JavaScript, React, Vite, React Router, React Leaflet/Leaflet, and Recharts.
- **External inputs:** Configurable weather providers, Open-Meteo air-quality/elevation services, NASA FIRMS hotspot records, and NASA GIBS imagery. [1][6][7]

### 3.2 Important repository modules

| Path | Role |
|---|---|
| `fireguard/api.py` | FastAPI application, endpoints, background live polling, image inference, and WebSocket updates |
| `fireguard/config.py` | Environment-based model, device, provider, location, and threshold settings |
| `fireguard/model_loader.py` and `fireguard/runtime.py` | Resolve and load model weights; expose model and wildfire-class readiness status |
| `fireguard/risk.py` | Event risk score, location weather-risk heuristic, and risk-level mapping |
| `fireguard/forecast.py` | Wind-advection point projection |
| `fireguard/live_data.py` and `fireguard/environment.py` | Live source collection and weather-provider adapters |
| `training/train.py` | YOLO training from a user-supplied dataset directory |
| `training/evaluate.py` and `training/metrics.py` | Validation/test evaluation, class-name checks, and metrics reports |
| `scripts/inspect_dataset.py` and `scripts/prepare_fasdd_uav.py` | Dataset inspection and FASDD UAV preparation utilities |
| `dashboard/src/App.jsx` | Dashboard data flow, navigation, pages, charts, and image-inference form |
| `dashboard/src/components/LiveMap*` | Map UI components |

### 3.3 Runtime data flow

The backend starts the FastAPI application, initializes the database, begins loading the configured model, and runs its live-source poller. The poller retrieves environmental and satellite-related information for the configured monitoring location, stores recent observations in memory, computes an available location-risk estimate, and broadcasts updates to WebSocket clients. For image inference, the API validates the uploaded file type, executes YOLO, and returns prediction metadata. The dashboard initially requests `/api/dashboard/summary`, listens on `/ws/live`, and refreshes the REST summary periodically as a fallback. [3]

## 4. Model and dataset

### 4.1 Configured detector

The default settings are model family `yolo26`, size `n`, device `cpu`, and an image confidence threshold of `0.35`. Unless a custom path is configured with `FIREGUARD_MODEL_WEIGHTS`, the weight name is resolved from family and size (default `yolo26n.pt`). The model loader uses the Ultralytics API. The project configuration also mentions a future `yolo27` switch, but this report does not treat that as a currently available trained model. [1][8]

**Wildfire-class caveat:** the repository documentation says the included/default weights contain COCO classes rather than fire/smoke classes. The training YAML lists class `0` as `fire` and class `1` as `smoke`, but the associated images are not included in the repository. The repository's runtime status distinguishes a model that loads from one whose class names include fire/smoke. Do not describe default COCO detections as validated wildfire detections. [2][9]

### 4.2 Dataset preparation and training

The repository documents an inspect → prepare → train → evaluate workflow. Dataset inspection is intended to count actual images and annotations and identify quality issues before training. FASDD UAV preparation expects source images and annotations; the prepared YOLO dataset should have training, validation, and test splits. A reproducible pilot may use a capped sample, but its scope must be clearly disclosed. [10]

The training script reads class names from a dataset YAML and calls Ultralytics training with configured hyperparameters. The supplied hyperparameter file includes automatic optimizer selection, initial learning rate `0.01`, cosine scheduling, early-stopping patience `20`, and image augmentation options. These are configuration values, not evidence that a training run has completed or achieved any particular score. [11]

### 4.3 Training objective (conceptual)

YOLO training learns model parameters [0m\(\theta\) from labeled images by minimizing a detection loss over the training set:

\[
\theta^* = \arg\min_{\theta} \frac{1}{N}\sum_{i=1}^{N} L_{\mathrm{det}}(f_{\theta}(I_i), y_i),
\]

where \(I_i\) is an input image, \(y_i\) its labeled boxes/classes, and \(f_{\theta}\) the detector. This is a general description of supervised object-detector training; this repository delegates the precise loss implementation to the installed Ultralytics package and does not define a separate YOLO loss formula in its own training script. [5][11]

## 5. Mathematical formulas used by the project

### 5.1 Feature-weighted event risk score

The event-level risk engine uses default weights for visual evidence, persistence, growth, wind, temperature, humidity, dryness, and smoke. For each feature \(x_i\), the implementation clips its supplied value to the range 0–100 and forms a weighted sum:

\[
R_{\mathrm{event}} = \operatorname{clip}\left(\sum_i w_i\operatorname{clip}(x_i,0,100),\,0,100\right).
\]

Default weights are:

\[
(w_{\mathrm{visual}},w_{\mathrm{persistence}},w_{\mathrm{growth}},w_{\mathrm{wind}},w_{\mathrm{temperature}},w_{\mathrm{humidity}},w_{\mathrm{dryness}},w_{\mathrm{smoke}})
=(0.25,0.15,0.15,0.10,0.10,0.10,0.10,0.05).
\]

The current image endpoint passes a subset of those features into this scorer. Risk labels are LOW for scores below 20, GUARDED below 40, MODERATE below 60, HIGH below 80, and CRITICAL otherwise. This is a heuristic index, not a probability calibrated against fire occurrence. [12]

### 5.2 Location weather-risk index

The location-based index is a separate calculation. For available, fresh weather observations, each variable is transformed and clipped to 0–100:

\[
\begin{aligned}
f_H &= \operatorname{clip}(100-H,0,100),\\
f_T &= \operatorname{clip}(3(T-15),0,100),\\
f_W &= \operatorname{clip}(2V,0,100),\\
f_P &= \operatorname{clip}(100-20P,0,100),
\end{aligned}
\]

where \(H\) is relative humidity (%), \(T\) temperature (°C), \(V\) wind speed (km/h), and \(P\) rainfall (mm), using the implementation's input conventions. For the available subset \(A\) of factors, the index is:

\[
R_{\mathrm{location}} = \frac{\sum_{i\in A} w_i f_i}{\sum_{i\in A}w_i},
\quad (w_H,w_T,w_W,w_P)=(0.40,0.25,0.20,0.15).
\]

At least two usable factors are required. The index is withheld for demo-only, stale, unavailable, or insufficient observations. It is explicitly a weather-only heuristic index—not a probability of a wildfire. [12]

### 5.3 Wind-advection baseline

The estimated drift distance for wind speed \(v\) over a horizon \(t\) is:

\[
d_{\mathrm{km}} = v_{\mathrm{km/h}}\frac{t_{\mathrm{minutes}}}{60}.
\]

The implementation converts the downwind bearing \(\beta\) to radians, with wind direction interpreted as the direction **from** which wind is blowing:

\[
\beta = \operatorname{radians}((\phi_{\mathrm{from}}+180)\bmod 360),
\qquad \delta = d/R_E,
\]

where \(R_E=6371.0088\) km is the Earth's mean radius. Given origin latitude/longitude \((\varphi_1,\lambda_1)\), the spherical destination equations are:

\[
\varphi_2=\arcsin(\sin\varphi_1\cos\delta+\cos\varphi_1\sin\delta\cos\beta),
\]

\[
\lambda_2=\lambda_1+\operatorname{atan2}(\sin\beta\sin\delta\cos\varphi_1,\cos\delta-\sin\varphi_1\sin\varphi_2).
\]

The resulting point is a current-wind advection estimate. It does not simulate fuel, terrain slope, changing weather, combustion, or an expanding fire perimeter. It requires a geolocated event and wind speed/direction; otherwise the API reports an unavailable/invalid request rather than inventing a forecast. [13]

### 5.4 Evaluation metric formulas

For an evaluation against labeled detections, precision, recall, and F1 are:

\[
\mathrm{Precision}=\frac{TP}{TP+FP},\qquad
\mathrm{Recall}=\frac{TP}{TP+FN},\qquad
F_1=2\frac{\mathrm{Precision}\cdot\mathrm{Recall}}{\mathrm{Precision}+\mathrm{Recall}}.
\]

The evaluator also reports mean Average Precision (mAP) at IoU 0.50 and across IoU thresholds 0.50–0.95 when the actual validation run provides those values. FPS is derived from measured evaluation timings. These are evaluation metrics, not results of this report; they should only be populated from a completed run on labeled data. [14]

## 6. Project evaluation and results

### 6.1 Evaluation status

The repository provides code for dataset inspection, training, evaluation, and measured inference-performance logging. However, the checked repository state contains no supplied labeled image dataset and no verified evaluation report establishing fire/smoke accuracy. Its configured default weights are COCO weights, not a fire/smoke checkpoint. Therefore, the responsible result for wildfire detector precision, recall, F1, mAP, and test FPS in this report is **not available / not established**. [2][9][14]

| Evaluation item | Result supported by repository evidence |
|---|---|
| Backend and dashboard implementation | Present in repository; must be run locally to verify a particular machine's behavior |
| Fire/smoke-labeled dataset included | No; dataset images are not included |
| Default weights trained for fire/smoke | No; documented as COCO-class weights |
| Fire/smoke validation precision / recall / F1 | Not established; no verified wildfire evaluation report supplied |
| Fire/smoke mAP@50 / mAP@50–95 | Not established; requires trained weights and labeled validation data |
| Live weather/satellite availability | Depends on network, monitoring coordinates, freshness, and provider configuration |
| NASA FIRMS hotspot availability | Requires configured NASA FIRMS MAP_KEY and may legitimately return zero records |
| Spread-estimate accuracy | Not validated; current implementation is a wind-only point-advection baseline |

### 6.2 Recommended reproducible test plan

1. Obtain a legally usable fire/smoke dataset with bounding-box annotations and documented class definitions.
2. Run the repository's dataset inspector on the root containing both images and annotations; preserve its generated report.
3. Prepare the dataset into train/validation/test splits. Record source, split procedure, class mapping, counts, exclusions, and any sample cap/seed.
4. Train with a unique run name and record the exact starting weights, command, package versions, hardware/device, epochs, batch size, image size, and date.
5. Evaluate the selected checkpoint on the validation split during iteration and on the held-out test split for final reporting. The evaluator checks that model classes match the dataset YAML.
6. Report confusion-related counts where available, per-class precision/recall/F1/AP, mAP@50, mAP@50–95, inference/evaluation speed, and the dataset/test scope. Do not call pilot metrics production performance.
7. Test external-source states separately: live/recent weather, missing weather, stale observations, missing/invalid FIRMS key, empty FIRMS response, and network failure. Confirm that unavailable statuses are visible.
8. Test the wind-advection output against known geometry/unit cases and label it as a baseline until compared with independent, time-stamped fire-perimeter observations.

### 6.3 Result interpretation

Successful API startup, successful model loading, or a displayed bounding box is not equivalent to demonstrated wildfire-detection accuracy. A model can load while lacking fire/smoke classes; the project exposes model readiness separately from loaded status. Likewise, a satellite hotspot is a source record, not ground-truth confirmation, and a computed risk score is not a calibrated incident probability. [2][3][12]

## 7. Step-by-step execution in Visual Studio Code

### 7.1 Prerequisites

- Git
- Python 3.10 or later (the repository Dockerfile uses Python 3.11)
- Node.js 20 and npm for the dashboard
- VS Code with the Python and JavaScript tooling/extensions of your choice
- Internet connectivity for initial package installation and optional external data/model downloads

### 7.2 Clone and open the project

In a terminal:

```bash
git clone https://github.com/MaddineniCharanKumar/wildfire-spread-detection.git
cd wildfire-spread-detection
code .
```

### 7.3 Install and start the backend

**Windows PowerShell:**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[ml,dev]"
uvicorn fireguard.api:app --reload --host 127.0.0.1 --port 8000
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[ml,dev]"
uvicorn fireguard.api:app --reload --host 127.0.0.1 --port 8000
```

Keep this terminal running. Verify `http://127.0.0.1:8000/api/health`; open the interactive API documentation at `http://127.0.0.1:8000/docs`.

### 7.4 Install and start the dashboard

Open a **second** VS Code terminal:

```bash
cd dashboard
npm ci
npm run dev
```

Open the local URL printed by Vite, normally `http://localhost:5173`. The dashboard uses `VITE_API_BASE_URL` when set; its default local configuration can use the same machine's API. If the browser reports a connection error, check that the backend terminal is still running and that the API URL/CORS configuration matches.

### 7.5 Configure monitoring inputs

The backend reads `FIREGUARD_`-prefixed settings from environment variables and a root `.env` file. Configure monitoring latitude/longitude to retrieve location-based observations. Set `FIREGUARD_FIRMS_MAP_KEY` on the **backend host** to request NASA FIRMS records. Weather provider options and defaults are described in `fireguard/config.py` and `README.md`. Do not put provider secrets in frontend code. If a key or provider is unavailable, the application should show its status instead of fabricated measurements. [1][15]

### 7.6 Try image inference and inspect outputs

1. Open **Analytics** in the dashboard.
2. Check the displayed model status and confirm whether fire/smoke classes are available.
3. Upload a JPEG, PNG, WebP, or BMP file using **Test detector with an image**.
4. Inspect the response's prediction count, class names, confidence scores, bounding boxes, and latency.
5. Treat predictions from default COCO weights as generic object predictions, not wildfire detections.
6. Check the dashboard's monitoring-site, source-status, alerts, and analytics views. Wind-based forecasts require an event with coordinates and usable wind; location-risk scores require sufficiently fresh weather observations.

### 7.7 Train and evaluate with a supplied dataset (optional)

After obtaining and verifying the dataset, use the actual dataset source path. The following commands illustrate the repository's documented workflow; paths must be adapted to the local machine:

```powershell
python scripts\inspect_dataset.py --dataset "C:\path\to\FASDD_UAV" --output reports
python scripts\prepare_fasdd_uav.py --source "C:\path\to\FASDD_UAV" --output data\processed --copy
python training\train.py --data data\processed --weights yolo26n.pt --epochs 50 --workers 2 --name fasdd-uav
python training\evaluate.py --weights runs\train\fasdd-uav\weights\best.pt --data data\processed\dataset.yaml --split test --output reports\final
```

Use a unique training name and verify the actual output checkpoint path under `runs/train/`. The command does not guarantee training success or any target accuracy. Report only metrics generated by the completed run and include its dataset scope. [10][14]

## 8. Expected final output and dashboard

When configured and running, the project aims to provide:

- **Command Center:** chosen monitoring location, available local weather-risk index, live conditions, observed risk history, and source freshness/status.
- **Live Map:** monitoring point/radius, available NASA FIRMS hotspot markers, wind vector, and NASA GIBS imagery.
- **Fire Events and Alerts:** events and alert acknowledgement workflow generated from available supported inputs.
- **Environment:** weather and environmental observations with source/status metadata.
- **Spread Prediction:** estimated downwind point for a geolocated event and selected horizon, when current wind inputs are usable.
- **Historical Data:** query of date-specific NASA FIRMS records where credentials and source availability permit.
- **Analytics:** measured inference runtime and prediction counts; labeled evaluation metrics only after a valid report exists; dataset/provider reports when available.
- **Data Sources and Monitoring Site:** provider status and editable area-of-interest coordinates/radius.

The UI is designed to show waiting, unavailable, stale, demo, or configuration-required states instead of manufacturing observations. Actual values and final screenshots depend on the machine, model weights, monitoring site, network, source configuration, and evaluation data. [3][16]

## 9. Conclusion

EcoSpread-YOLO provides a modular implementation that connects object-detection inference, live data providers, risk heuristics, a wind-advection baseline, and a React operations dashboard. Its strongest reportable contribution is the integrated and transparent prototype workflow, not a claimed wildfire accuracy score. To establish fire/smoke detector performance, the next essential step is to provide a labeled dataset, train a matching fire/smoke checkpoint, and publish reproducible held-out evaluation results. The risk and spread calculations should remain clearly identified as estimates until validated against appropriate ground truth.

## References

[1] MaddineniCharanKumar, *wildfire-spread-detection repository*, README and project configuration. https://github.com/MaddineniCharanKumar/wildfire-spread-detection (accessed 18 November 2026).

[2] MaddineniCharanKumar, *Model Configuration & Scientific Limitations*, `docs/model.md`; also `dashboard/README.md`. https://github.com/MaddineniCharanKumar/wildfire-spread-detection/blob/main/docs/model.md (accessed 18 November 2026).

[3] MaddineniCharanKumar, *Architecture*, `docs/architecture.md`, and FastAPI implementation, `fireguard/api.py`. https://github.com/MaddineniCharanKumar/wildfire-spread-detection/blob/main/docs/architecture.md (accessed 18 November 2026).

[4] MaddineniCharanKumar, *Dataset preparation and evaluation*, `docs/dataset.md`. https://github.com/MaddineniCharanKumar/wildfire-spread-detection/blob/main/docs/dataset.md (accessed 18 November 2026).

[5] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You Only Look Once: Unified, Real-Time Object Detection,” *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 779–788. https://doi.org/10.1109/CVPR.2016.91. This is background on the YOLO detector family, not a YOLO26 wildfire validation study.

[6] Ultralytics, *Ultralytics YOLO Documentation*. https://docs.ultralytics.com/ (accessed 18 November 2026).

[7] NASA, *Fire Information for Resource Management System (FIRMS)*. https://firms.modaps.eosdis.nasa.gov/ (accessed 18 November 2026).

[8] MaddineniCharanKumar, `fireguard/config.py` and `fireguard/model_loader.py`. https://github.com/MaddineniCharanKumar/wildfire-spread-detection/blob/main/fireguard/config.py and https://github.com/MaddineniCharanKumar/wildfire-spread-detection/blob/main/fireguard/model_loader.py (accessed 18 November 2026).

[9] MaddineniCharanKumar, `training/dataset.yaml` and `fireguard/runtime.py`. https://github.com/MaddineniCharanKumar/wildfire-spread-detection/blob/main/training/dataset.yaml and https://github.com/MaddineniCharanKumar/wildfire-spread-detection/blob/main/fireguard/runtime.py (accessed 18 November 2026).

[10] MaddineniCharanKumar, `scripts/inspect_dataset.py`, `scripts/prepare_fasdd_uav.py`, and `docs/dataset.md`. https://github.com/MaddineniCharanKumar/wildfire-spread-detection/tree/main/scripts (accessed 18 November 2026).

[11] MaddineniCharanKumar, `training/train.py` and `training/hyperparameters.yaml`. https://github.com/MaddineniCharanKumar/wildfire-spread-detection/blob/main/training/train.py and https://github.com/MaddineniCharanKumar/wildfire-spread-detection/blob/main/training/hyperparameters.yaml (accessed 18 November 2026).

[12] MaddineniCharanKumar, `fireguard/risk.py` and `docs/risk-engine.md`. https://github.com/MaddineniCharanKumar/wildfire-spread-detection/blob/main/fireguard/risk.py (accessed 18 November 2026).

[13] MaddineniCharanKumar, `fireguard/forecast.py`. https://github.com/MaddineniCharanKumar/wildfire-spread-detection/blob/main/fireguard/forecast.py (accessed 18 November 2026).

[14] MaddineniCharanKumar, `training/evaluate.py` and `training/metrics.py`. https://github.com/MaddineniCharanKumar/wildfire-spread-detection/blob/main/training/evaluate.py (accessed 18 November 2026).

[15] Open-Meteo, *Weather API Documentation*. https://open-meteo.com/en/docs (accessed 18 November 2026).

[16] MaddineniCharanKumar, `dashboard/src/App.jsx` and `dashboard/README.md`. https://github.com/MaddineniCharanKumar/wildfire-spread-detection/blob/main/dashboard/src/App.jsx and https://github.com/MaddineniCharanKumar/wildfire-spread-detection/blob/main/dashboard/README.md (accessed 18 November 2026).
