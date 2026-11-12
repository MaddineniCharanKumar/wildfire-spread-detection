# Dashboard

The dashboard should consume the live API and websocket outputs to visualize detection, environment, risk, alerts, and forecast data without inventing measurement values.

## Dashboard responsibilities

- use the selected monitoring coordinates for current weather, risk indexing, maps, and satellite queries;
- display environmental readings with source, observation time, and freshness;
- render a transparent weather-based index and observed history only when usable weather inputs exist;
- show real NASA FIRMS records with their actual date, confidence, radiative power, sensor, and source;
- clearly indicate unavailable or stale data states.

## Data policy

The frontend must never silently replace missing or stale live data with demo values. Any display of demo mode should be explicitly labeled in the UI.

## Live map (NASA GIBS map)

The Command Center and Live Map let users set a latitude, longitude, and search radius. The API persists the selection on the server and refreshes weather and nearby FIRMS detections for that location. The Live Map uses a keyless Leaflet map on NASA GIBS imagery (`src/components/LiveMap.jsx`), including:
- the selected monitoring point and its query-radius boundary;
- live NASA FIRMS satellite hotspot markers, when configured;
- a wind vector from the current weather reading;
- an imagery-date selector for NASA GIBS, including optional MODIS 16-day NDVI imagery;
- a Historical Data page that queries NASA FIRMS Standard Processing for the selected point/radius, a user-selected date, and a one-to-five-day window.

Spread Prediction uses `/api/forecast/{event_id}` to project the event location downwind from the latest observed wind. This baseline is explicitly an estimate, not a validated fire perimeter or operational forecast.

The location risk display is a weather-only heuristic index, not the probability of fire occurrence. It is withheld when readings are stale, demo-only, or insufficient. FIRMS summaries are source-observation descriptions, not verified incident statistics. Unavailable model-accuracy metrics are removed; measured inference timing is shown only when actual inference samples exist.

The available local `yolo26n.pt` weights contain COCO classes rather than fire/smoke classes. Model runtime and health panels are intentionally not presented as wildfire capability. Provider secrets belong in the API host `.env` file; the dashboard does not store them.
