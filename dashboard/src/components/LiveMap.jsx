import { useEffect } from 'react'
import { MapContainer, TileLayer, WMSTileLayer, LayersControl, LayerGroup, CircleMarker, Circle, Polyline, Popup, useMap } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'

// NASA GIBS (Global Imagery Browse Services): free, no API key. https://nasa-gibs.github.io/gibs-api-docs/
const GIBS_WMS = 'https://gibs.earthdata.nasa.gov/wms/epsg3857/best/wms.cgi'
const FALLBACK_LAT = Number(import.meta.env.VITE_MAP_LATITUDE) || 12.9716
const FALLBACK_LON = Number(import.meta.env.VITE_MAP_LONGITUDE) || 77.5946

const utcDate = (offsetDays = 0) => new Date(Date.now() + offsetDays * 86400000).toISOString().slice(0, 10)

// FIRMS confidence: VIIRS reports l / n / h, MODIS reports 0-100.
function tier(conf) {
  const s = String(conf ?? '').toLowerCase()
  if (s === 'h' || s === 'high') return 'high'
  if (s === 'l' || s === 'low') return 'low'
  if (s === 'n' || s === 'nominal') return 'nominal'
  const n = Number(s)
  if (!Number.isNaN(n) && s !== '') return n >= 80 ? 'high' : n >= 30 ? 'nominal' : 'low'
  return 'nominal'
}
const COLORS = { high: '#ef4444', nominal: '#f59e0b', low: '#eab308' }

// Point `km` away from (lat, lon) along a compass bearing.
function destination(lat, lon, bearingDeg, km) {
  const r = Math.PI / 180, d = km / 6371.0088, b = bearingDeg * r, la = lat * r, lo = lon * r
  const la2 = Math.asin(Math.sin(la) * Math.cos(d) + Math.cos(la) * Math.sin(d) * Math.cos(b))
  const lo2 = lo + Math.atan2(Math.sin(b) * Math.sin(d) * Math.cos(la), Math.cos(d) - Math.sin(la) * Math.sin(la2))
  return [la2 / r, lo2 / r]
}

function Recenter({ center, zoom }) {
  const map = useMap()
  useEffect(() => { map.setView(center, zoom) }, [center[0], center[1], zoom]) // eslint-disable-line react-hooks/exhaustive-deps
  return null
}

/**
 * Live operational map. Real geodata only:
 *  - NASA GIBS imagery + VIIRS thermal-anomaly tiles (keyless, straight from NASA)
 *  - NASA FIRMS hotspots returned by the backend (needs FIREGUARD_FIRMS_MAP_KEY on the server)
 *  - the detector's fire event, if any
 *  - a wind vector from the live weather reading
 * It never invents a fire, hotspot or forecast perimeter.
 */
export default function LiveMap({ location, radiusKm, fire, satelliteFires = [], environment, forecast, imageryDate }) {
  const hasSite = location?.latitude != null && location?.longitude != null
  const hasFire = fire?.latitude != null && fire?.longitude != null
  const first = satelliteFires.find(f => Number.isFinite(Number(f.latitude)))
  const center = hasSite ? [location.latitude, location.longitude]
    : hasFire ? [fire.latitude, fire.longitude]
    : first ? [Number(first.latitude), Number(first.longitude)]
    : [FALLBACK_LAT, FALLBACK_LON]
  const zoom = hasSite ? (radiusKm >= 150 ? 7 : radiusKm >= 60 ? 8 : 10) : hasFire ? 12 : first ? 8 : 5

  // Wind is reported as the bearing it blows FROM; the vector points where it blows TO.
  const windTo = environment?.wind_direction != null ? (Number(environment.wind_direction) + 180) % 360 : null
  const windKm = 2 + Math.min(Number(environment?.wind_speed) || 0, 90) / 15 // visual scale only
  const windOrigin = hasFire ? [fire.latitude, fire.longitude] : hasSite ? [location.latitude, location.longitude] : null
  const windLine = windOrigin && windTo != null ? [windOrigin, destination(windOrigin[0], windOrigin[1], windTo, windKm)] : null

  const forecastOrigin = forecast && Number.isFinite(Number(forecast.origin_latitude)) && Number.isFinite(Number(forecast.origin_longitude))
    ? [Number(forecast.origin_latitude), Number(forecast.origin_longitude)] : null
  const forecastPoint = forecast && Number.isFinite(Number(forecast.projected_latitude)) && Number.isFinite(Number(forecast.projected_longitude))
    ? [Number(forecast.projected_latitude), Number(forecast.projected_longitude)] : null

  return (
    <div className="livemap">
      <MapContainer center={center} zoom={zoom} scrollWheelZoom style={{ height: '100%', width: '100%' }}>
        <Recenter center={center} zoom={zoom} />
        <LayersControl position="topright">
          <LayersControl.BaseLayer name={`NASA VIIRS true colour (${imageryDate || 'yesterday'})`}>
            <WMSTileLayer url={GIBS_WMS} layers="VIIRS_SNPP_CorrectedReflectance_TrueColor" format="image/jpeg" transparent={false} time={imageryDate || utcDate(-1)} maxZoom={9}
              attribution='Imagery: <a href="https://earthdata.nasa.gov/gibs">NASA GIBS</a> / VIIRS' />
          </LayersControl.BaseLayer>
          <LayersControl.BaseLayer checked name="OpenStreetMap">
            <TileLayer url="https://tile.openstreetmap.org/{z}/{x}/{y}.png" maxZoom={19} attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors' />
          </LayersControl.BaseLayer>

          <LayersControl.Overlay name="NASA thermal anomalies (VIIRS NOAA-20, GIBS)">
            <WMSTileLayer url={GIBS_WMS} layers="VIIRS_NOAA20_Thermal_Anomalies_375m_All" format="image/png" transparent time={imageryDate || utcDate(-1)} maxZoom={12}
              attribution='Thermal anomalies: <a href="https://firms.modaps.eosdis.nasa.gov/">NASA FIRMS</a>' />
          </LayersControl.Overlay>

          <LayersControl.Overlay name="NASA MODIS NDVI (16-day composite)">
            <WMSTileLayer url={GIBS_WMS} layers="MODIS_Terra_NDVI_16Day" format="image/png" transparent time={imageryDate || utcDate(-1)} maxZoom={9}
              attribution='Vegetation index imagery: <a href="https://earthdata.nasa.gov/gibs">NASA GIBS</a> / MODIS' />
          </LayersControl.Overlay>

          <LayersControl.Overlay checked name="NASA FIRMS hotspots (API)">
            <LayerGroup>
              {satelliteFires.slice(0, 500).map((f, i) => {
                const la = Number(f.latitude), lo = Number(f.longitude)
                if (!Number.isFinite(la) || !Number.isFinite(lo)) return null
                const t = tier(f.confidence)
                return (
                  <CircleMarker key={f.event_id || `${la}-${lo}-${i}`} center={[la, lo]} radius={6}
                    pathOptions={{ color: '#3a0d05', weight: 1, fillColor: COLORS[t], fillOpacity: 0.9 }}>
                    <Popup>
                      <b>NASA FIRMS hotspot</b><br />
                      {f.satellite || 'Satellite'} · {f.instrument || ''} · {f.acq_date} {String(f.acq_time || '').padStart(4, '0')} UTC<br />
                      {f.age_minutes != null && <>Observed {Math.round(f.age_minutes)} min ago<br /></>}
                      Confidence: {f.confidence ?? '—'} ({t}) · FRP: {f.frp ?? '—'} MW<br />
                      {f.distance_km != null && <>Distance from site: {f.distance_km} km<br /></>}
                      <small>{f.firms_source}</small>
                    </Popup>
                  </CircleMarker>
                )
              })}
            </LayerGroup>
          </LayersControl.Overlay>
        </LayersControl>

        {hasSite && (
          <>
            <CircleMarker center={[location.latitude, location.longitude]} radius={7} pathOptions={{ color: '#ffffff', weight: 2.5, fillColor: '#0284c7', fillOpacity: 1 }}>
              <Popup><b>Monitoring site</b><br />{location.latitude.toFixed(4)}, {location.longitude.toFixed(4)}</Popup>
            </CircleMarker>
            {radiusKm > 0 && <Circle center={[location.latitude, location.longitude]} radius={radiusKm * 1000} interactive={false} pathOptions={{ color: '#0284c7', weight: 2, dashArray: '6 6', fillColor: '#0284c7', fillOpacity: 0.04, interactive: false }} />}
          </>
        )}

        {hasFire && (
          <>
            <CircleMarker center={[fire.latitude, fire.longitude]} radius={9} pathOptions={{ color: '#ffffff', weight: 2, fillColor: '#ef4444', fillOpacity: 1 }}>
              <Popup><b>{fire.event_id}</b><br />Detector confidence: {Math.round((fire.confidence || 0) * 100)}%<br />Risk: {Math.round(fire.risk_score || 0)}/100</Popup>
            </CircleMarker>
            <Circle center={[fire.latitude, fire.longitude]} radius={Math.max(120, Math.sqrt(fire.visual_area_proxy || 0) * 8)} pathOptions={{ color: '#ea580c', weight: 1.5, fillColor: '#ea580c', fillOpacity: 0.15, interactive: false }} />
          </>
        )}
        {forecastOrigin && forecastPoint && <Polyline positions={[forecastOrigin, forecastPoint]} pathOptions={{ color: '#f59e0b', weight: 3, dashArray: '5 5' }} />}
        {forecastPoint && <CircleMarker center={forecastPoint} radius={7} pathOptions={{ color: '#ffffff', weight: 2, fillColor: '#f59e0b', fillOpacity: 1 }}><Popup><b>Wind-advection estimate</b><br />{forecast.horizon_minutes} min · {forecast.drift_km} km downwind<br />Not a validated fire perimeter.</Popup></CircleMarker>}
        {windLine && <Polyline positions={windLine} pathOptions={{ color: '#64748b', weight: 2.5, dashArray: '3 6', interactive: false }} />}
        {windLine && <CircleMarker center={windLine[1]} radius={4} pathOptions={{ color: '#64748b', fillColor: '#64748b', fillOpacity: 1 }}><Popup>Wind blowing toward {Math.round(windTo)}° at {environment.wind_speed ?? '—'} km/h (live reading; line length is a visual scale)</Popup></CircleMarker>}
      </MapContainer>

      <div className="map-legend">
        <b>FIRMS hotspots: {satelliteFires.length}</b>
        {Object.entries(COLORS).map(([k, c]) => <span key={k}><i style={{ background: c }} />{k}</span>)}
        <span><i style={{ background: '#0284c7' }} />site / radius</span>
      </div>
      {!hasSite && !hasFire && !first && (
        <div className="map-note">Enter a latitude and longitude above to centre the map and query nearby NASA FIRMS detections.</div>
      )}
    </div>
  )
}
