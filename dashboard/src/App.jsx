import { createContext, useContext, useEffect, useMemo, useState } from 'react'
import { BrowserRouter, NavLink, Route, Routes, useLocation } from 'react-router-dom'
import { AreaChart, Area, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import LiveMap from './components/LiveMap'

const API = import.meta.env.VITE_API_BASE_URL || ''
const DashboardContext = createContext(null)
const nav = [
  ['/', '⌂', 'Command Center'],
  ['/map', '⌖', 'Live Map'],
  ['/fires', '🔥', 'Fire Events'],
  ['/environment', '☁', 'Environment'],
  ['/spread', '↗', 'Spread Prediction'],
  ['/history', '◷', 'Historical Data'],
  ['/alerts', '!', 'Alerts'],
  ['/analytics', '⌁', 'Analytics'],
  ['/system', '◉', 'Data Sources'],
  ['/monitoring', '◎', 'Monitoring Site'],
]

async function getJSON(path) {
  const res = await fetch(API + path)
  const body = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(body.detail || `${res.status} ${res.statusText}`)
  return body
}

function utcDayOffset(days = 0) {
  return new Date(Date.now() + days * 86400000).toISOString().slice(0, 10)
}

function useDashboardState() {
  const [data, setData] = useState(null)
  const [error, setError] = useState('')
  const [live, setLive] = useState(false)
  const load = async () => {
    try {
      const result = await getJSON('/api/dashboard/summary')
      setData(result)
      setError('')
      return result
    }
    catch (e) { setError(e.message) }
  }
  useEffect(() => {
    load()
    const wsBase = API ? API.replace(/^http/, 'ws') : `ws://${window.location.host}`
    const ws = new WebSocket(`${wsBase}/ws/live`)
    ws.onopen = () => { setLive(true); ws.send('subscribe') }
    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data)
        if (msg.type === 'live_update') {
          setData(prev => prev ? {
            ...prev,
            timestamp: msg.timestamp,
            environment: msg.environment || prev.environment,
            location_risk: msg.location_risk || prev.location_risk,
            risk_history: msg.risk_history || prev.risk_history,
            satellite_fires: msg.satellite?.fires || prev.satellite_fires || [],
            satellite_meta: msg.satellite ? Object.fromEntries(Object.entries(msg.satellite).filter(([k]) => k !== 'fires')) : prev.satellite_meta,
            live: { ...prev.live, timestamp: msg.timestamp, sources: {
              ...prev.live?.sources,
              weather: msg.environment?.status || prev.live?.sources?.weather,
              air_quality: msg.environment?.air_quality_status || prev.live?.sources?.air_quality,
              terrain: msg.terrain?.terrain_status || prev.live?.sources?.terrain,
              satellite: msg.satellite?.satellite_status || prev.live?.sources?.satellite,
            }},
            providers: {
              ...prev.providers,
              satellite: msg.satellite?.satellite_status || prev.providers?.satellite,
              terrain: msg.terrain?.terrain_status || prev.providers?.terrain,
            }
          } : prev)
        } else if (msg.type === 'detection_update') {
          load()
        }
      } catch (_) {}
    }
    ws.onerror = () => setLive(false)
    ws.onclose = () => setLive(false)
    const id = setInterval(load, 60000)
    return () => { clearInterval(id); ws.close() }
  }, [])
  return { data, error, live, refresh: load }
}

function useDashboard() {
  return useContext(DashboardContext)
}

function dashboardLive(data) { return data?.live?.timestamp && data?.health?.status === 'healthy' }

function Shell({ children, data, error }) {
  const location = useLocation()
  const status = data?.health?.status === 'healthy'
  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-mark">ES</div><div><b>FireGuard</b><span>WILDFIRE INTELLIGENCE</span></div></div>
      <div className="status-pill"><i className={dashboardLive(data) ? 'dot live' : 'dot'} />{data?.health?.demo_mode ? 'DEMO MODE' : dashboardLive(data) ? 'LIVE DATA' : status ? 'API ONLINE' : 'API OFFLINE'}</div>
      <nav>{nav.map(([to, icon, label]) => <NavLink key={to} to={to} className={({isActive}) => isActive ? 'nav-link active' : 'nav-link'}><span>{icon}</span>{label}</NavLink>)}</nav>
      <div className="sidebar-footer">
        <div className="mini-source"><span>Weather</span><b>{data?.environment?.status || '—'}</b></div>
        <div className="mini-source"><span>Satellite</span><b>{data?.providers?.satellite || '—'}</b></div>
        <small>Selected-area monitoring · sourced telemetry</small>
      </div>
    </aside>
    <main className="main">
      <header className="topbar">
        <div><span className="eyebrow">{data?.live?.location?.latitude != null ? `MONITORING SITE · ${Number(data.live.location.latitude).toFixed(4)}, ${Number(data.live.location.longitude).toFixed(4)}` : 'WILDFIRE MONITORING · NO SITE SELECTED'}</span><h1>{nav.find(x => x[0] === location.pathname)?.[2] || 'FireGuard'}</h1></div>
        <div className="top-actions"><span className="fresh">Updated {data?.timestamp ? new Date(data.timestamp).toLocaleTimeString() : '—'}</span><button onClick={() => window.location.reload()}>↻ Refresh</button></div>
      </header>
      {error && <div className="api-warning">API unavailable: {error}. The dashboard will not fabricate telemetry.</div>}
      {children}
    </main>
  </div>
}

function Stat({label, value, unit, tone=''}) {
  return <div className="stat-card"><span>{label}</span><strong className={tone}>{value ?? '—'}{value != null && unit ? <em>{unit}</em> : null}</strong></div>
}

function EnvironmentCards({env}) {
  const items = [
    ['Temperature', env?.temperature, '°C'], ['Humidity', env?.humidity, '%'], ['Wind speed', env?.wind_speed, 'km/h'],
    ['Wind direction', env?.wind_direction, '°'], ['Rainfall', env?.rainfall, 'mm'], ['Pressure', env?.pressure, 'hPa'],
    ['PM2.5', env?.pm25, 'µg/m³'], ['PM10', env?.pm10, 'µg/m³'], ['Visibility', env?.visibility, 'km']
  ]
  return <div className="metric-grid">{items.map(([l,v,u]) => <Stat key={l} label={l} value={v} unit={u}/>)}</div>
}

function RiskCard({risk}) {
  const available = risk?.status === 'ESTIMATE' && risk.score != null
  return <div className={`risk-card ${available ? `risk-${String(risk.level).toLowerCase()}` : ''}`}>
    <div className="risk-ring" style={{'--risk': `${available ? risk.score : 0}%`}}><div><strong>{available ? `${Math.round(risk.score)}%` : '—'}</strong>{available && <span>INDEX</span>}</div></div>
    <div><span className="eyebrow">LOCAL FIRE-WEATHER INDEX</span><h2>{available ? risk.level : risk?.status || 'WAITING FOR OBSERVATION'}</h2>
      <p>{available ? `${risk.source || 'Weather provider'} · ${risk.timestamp ? new Date(risk.timestamp).toLocaleTimeString() : 'time unavailable'}` : 'A current, usable weather observation is needed to estimate local conditions.'}</p>
      <small>{available && risk.inputs_used?.length ? `Inputs: ${risk.inputs_used.join(', ')}. ` : ''}Relative weather index only — not a probability that a fire will occur.</small>
    </div>
  </div>
}

function EmptyState({title, text}) { return <div className="empty"><div className="empty-icon">◎</div><h3>{title}</h3><p>{text}</p></div> }

function CommandCenter({data}) {
  const env = data?.environment || {}
  const location = data?.live?.location
  return <div className="page">
    <section className="hero-grid monitor-hero">
      <div className="panel monitor-panel"><span className="eyebrow">LOCATION-BASED MONITORING</span><h2>Choose an area to watch</h2>
        <p>Live weather and nearby NASA FIRMS satellite detections are fetched for the coordinates you set.</p>
        <LocationMonitor location={location} radiusKm={data?.satellite_meta?.radius_km} compact/>
      </div>
      <RiskCard risk={data?.location_risk}/>
    </section>
    <section className="section"><div className="section-head"><div><span className="eyebrow">ENVIRONMENTAL INTELLIGENCE</span><h2>Live conditions</h2></div><span className="source-chip">{env.source || 'UNAVAILABLE'} · {env.freshness || 'UNAVAILABLE'}</span></div><EnvironmentCards env={env}/></section>
    <section className="dashboard-grid">
      <div className="panel">
        <div className="panel-head"><div><span className="eyebrow">FIRE RISK TIMELINE</span><h2>Observed risk trend</h2></div></div>
        <RiskTimeline data={data?.risk_history || []}/>
      </div>
      <div className="panel">
        <div className="panel-head"><div><span className="eyebrow">DATA SOURCES</span><h2>Trust & freshness</h2></div></div>
        <SourceList data={data}/>
      </div>
    </section>
  </div>
}

function RiskTimeline({data}) {
  if (!data?.length) return <EmptyState title="No weather-risk observations yet" text="A trend appears after the system receives usable, timestamped weather observations for the monitored location."/>
  return <div className="chart"><ResponsiveContainer width="100%" height={260}><AreaChart data={data}><defs><linearGradient id="riskFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#ea580c" stopOpacity={0.22}/><stop offset="100%" stopColor="#ea580c" stopOpacity={0.01}/></linearGradient></defs><CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false}/><XAxis dataKey="time" tickFormatter={value => { const date = new Date(value); return Number.isNaN(date.getTime()) ? value : date.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'}) }} stroke="#94a3b8" tick={{fill: '#64748b', fontSize: 11}} axisLine={{stroke: '#e2e8f0'}} tickLine={false}/><YAxis domain={[0,100]} stroke="#94a3b8" tick={{fill: '#64748b', fontSize: 11}} axisLine={{stroke: '#e2e8f0'}} tickLine={false}/><Tooltip contentStyle={{background:'#ffffff', border:'1px solid #e2e8f0', borderRadius:'10px', boxShadow:'0 10px 25px -5px rgba(0,0,0,0.08)', color:'#0f172a', fontSize:'12px', fontWeight:500}} itemStyle={{color:'#ea580c', fontWeight:600}}/><Area type="monotone" dataKey="score" stroke="#ea580c" fill="url(#riskFill)" strokeWidth={2.5} activeDot={{r:5, fill: '#ea580c', stroke: '#ffffff', strokeWidth: 2}}/></AreaChart></ResponsiveContainer></div>
}

function SourceList({data}) {
  const rows = [
    ['Weather', data?.environment?.status, data?.environment?.source],
    ['Visibility', data?.environment?.visibility_status, data?.environment?.visibility_source],
    ['Satellite / FIRMS', data?.providers?.satellite, data?.live?.sources?.satellite || 'Provider'],
    ['Terrain / DEM', data?.providers?.terrain, 'Provider'],
    ['NDVI imagery', 'NASA GIBS', 'MODIS 16-day composite'],
  ]
  return <div className="source-list">{rows.map(([a,b,c]) => <div key={a}><span>{a}</span><b className={String(b).includes('UNAVAILABLE') ? 'muted' : ''}>{b || 'UNAVAILABLE'}</b><small>{c || '—'}</small></div>)}</div>
}

function SatelliteBanner({meta}) {
  if (!meta) return null
  const s = meta.satellite_status
  const msgs = {
    CONFIG_REQUIRED: 'NASA FIRMS hotspots are off: set FIREGUARD_FIRMS_MAP_KEY on the API server (free key). The NASA GIBS imagery layers above still work without a key.',
    INVALID_KEY: 'NASA FIRMS rejected the configured MAP_KEY. Check FIREGUARD_FIRMS_MAP_KEY.',
    RATE_LIMITED: 'NASA FIRMS rate limit reached; hotspots will refresh on the next successful poll.',
    UNAVAILABLE: `NASA FIRMS is unreachable${meta.satellite_message ? ': ' + meta.satellite_message : ''}.`,
    STALE: meta.satellite_message || 'Showing the last successful FIRMS refresh.',
    PARTIAL: meta.satellite_message || 'Some FIRMS satellite feeds failed.',
  }
  const text = msgs[s]
  const ok = ['LIVE','PARTIAL','STALE'].includes(s)
  return <div className={text ? 'api-warning' : 'notice'} style={{marginTop:12}}>
    {text || null}{text && ok ? ' ' : ''}
    {ok && <span>{meta.total_detections ?? 0} detections within {meta.radius_km ?? '—'} km · last {meta.days} day(s) · {(meta.sources || []).join(', ')} · fetched {meta.satellite_timestamp ? new Date(meta.satellite_timestamp).toLocaleTimeString() : '—'}</span>}
  </div>
}

function LocationMonitor({location, radiusKm, compact=false, onSaved}) {
  const dashboard = useDashboard()
  const [site, setSite] = useState({
    latitude: location?.latitude ?? '',
    longitude: location?.longitude ?? '',
    radius_km: radiusKm ?? 250,
  })
  const [saving, setSaving] = useState(false)
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')

  useEffect(() => {
    setSite(current => ({
      ...current,
      latitude: location?.latitude ?? current.latitude,
      longitude: location?.longitude ?? current.longitude,
      radius_km: radiusKm ?? current.radius_km,
    }))
  }, [location?.latitude, location?.longitude, radiusKm])

  async function save(event) {
    event.preventDefault()
    setSaving(true)
    setError('')
    setMessage('')
    try {
      const response = await fetch(`${API}/api/system/location`, {
        method: 'PUT',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          latitude: Number(site.latitude),
          longitude: Number(site.longitude),
          radius_km: Number(site.radius_km),
        }),
      })
      const body = await response.json().catch(() => ({}))
      if (!response.ok) throw new Error(body.detail || `${response.status} ${response.statusText}`)
      setMessage('Location saved. Refreshing live weather and nearby satellite detections…')
      if (onSaved) await onSaved()
      else await dashboard?.refresh()
    } catch (reason) {
      setError(reason.message || 'Could not update the monitoring location.')
    } finally {
      setSaving(false)
    }
  }

  return <form className={`site-form ${compact ? 'site-form-compact' : ''}`} onSubmit={save}>
    <label>Latitude<input aria-label="Monitoring latitude" required type="number" min="-90" max="90" step="any" value={site.latitude} onChange={event => setSite({...site, latitude: event.target.value})} placeholder="e.g. 12.9716"/></label>
    <label>Longitude<input aria-label="Monitoring longitude" required type="number" min="-180" max="180" step="any" value={site.longitude} onChange={event => setSite({...site, longitude: event.target.value})} placeholder="e.g. 77.5946"/></label>
    <label>Search radius (km)<input aria-label="FIRMS radius in kilometers" required type="number" min="1" max="5000" step="1" value={site.radius_km} onChange={event => setSite({...site, radius_km: event.target.value})}/></label>
    <button type="submit" disabled={saving}>{saving ? 'Updating location…' : 'Monitor this location'}</button>
    {message && <span className="inline-success site-form-message">{message}</span>}
    {error && <span className="inline-error site-form-message">{error}</span>}
  </form>
}

function MapPage() {
  const dashboard = useDashboard()
  const {data} = dashboard
  const [imageryDate, setImageryDate] = useState(utcDayOffset(-1))
  const minDate = utcDayOffset(-365)
  const maxDate = utcDayOffset(-1)
  return <div className="page"><section className="map-layout"><div className="panel map-panel"><div className="panel-head"><div><span className="eyebrow">GEOSPATIAL COMMAND</span><h2>Live operational map</h2></div><div className="layer-row"><span>🟠 FIRMS satellite hotspot</span><span>🛰 Thermal anomaly</span><span>🌿 NDVI composite</span></div></div><LocationMonitor location={data?.live?.location} radiusKm={data?.satellite_meta?.radius_km} compact onSaved={dashboard.refresh}/><div className="history-control"><label htmlFor="imagery-date">Imagery date</label><input id="imagery-date" type="date" min={minDate} max={maxDate} value={imageryDate} onChange={event => setImageryDate(event.target.value)}/><span>NASA imagery archive · separate from dated FIRMS observations</span></div><div className="map-canvas"><LiveMap location={data?.live?.location} radiusKm={data?.satellite_meta?.radius_km} satelliteFires={data?.satellite_fires || []} environment={data?.environment} imageryDate={imageryDate}/></div><SatelliteBanner meta={data?.satellite_meta}/><HotspotList fires={data?.satellite_fires || []} meta={data?.satellite_meta}/></div><div className="map-side"><RiskCard risk={data?.location_risk}/><div className="panel"><span className="eyebrow">MAP LEGEND</span><h2>Layers & coverage</h2><div className="layer-list">{[['Monitoring site',data?.live?.location?.latitude != null],['Nearby FIRMS detections',['LIVE','PARTIAL','STALE'].includes(data?.providers?.satellite)],['Terrain / elevation',data?.providers?.terrain==='LIVE'],['MODIS NDVI imagery',true],['NASA thermal imagery',true]].map(([x,on])=><div key={x}><i className={on?'dot live':'dot'}/><span>{x}</span><b>{on?'AVAILABLE':'UNAVAILABLE'}</b></div>)}</div><p className="disclaimer">NASA FIRMS detections are satellite hotspots, not verified fire incidents. NDVI is NASA MODIS 16-day imagery; no point-level NDVI reading is inferred. Map layers identify their provider and date.</p></div></div></section></div>
}

function HotspotList({fires, meta}) {
  const [filter, setFilter] = useState('all')
  const [search, setSearch] = useState('')
  const filtered = fires.filter(fire => {
    const confidence = String(fire.confidence ?? '').toLowerCase()
    const level = confidence === 'h' || confidence === 'high' || Number(confidence) >= 80 ? 'high'
      : confidence === 'l' || confidence === 'low' || (confidence !== '' && Number(confidence) < 30) ? 'low' : 'nominal'
    const query = search.trim().toLowerCase()
    return (filter === 'all' || filter === level)
      && (!query || [fire.satellite, fire.instrument, fire.acq_date, fire.firms_source].some(value => String(value || '').toLowerCase().includes(query)))
  })
  return <section className="hotspot-section">
    <div className="section-head"><div><span className="eyebrow">NASA FIRMS REGISTER</span><h2>Satellite hotspots</h2></div><span className="source-chip">{filtered.length} / {fires.length} shown · {meta?.satellite_status || 'UNAVAILABLE'}</span></div>
    <div className="hotspot-controls"><label>Confidence <select value={filter} onChange={event => setFilter(event.target.value)}><option value="all">All</option><option value="high">High</option><option value="nominal">Nominal</option><option value="low">Low</option></select></label><label>Search <input value={search} onChange={event => setSearch(event.target.value)} placeholder="Satellite, sensor, date, source"/></label></div>
    {filtered.length ? <div className="hotspot-table"><div className="hotspot-row hotspot-header"><span>Acquired (UTC)</span><span>Coordinates</span><span>Confidence</span><span>FRP</span><span>Distance</span><span>Satellite / source</span></div>{filtered.map((fire, index) => <div className="hotspot-row" key={fire.event_id || index}><span>{fire.acq_date || '—'} {String(fire.acq_time || '').padStart(4, '0')}</span><span>{Number(fire.latitude).toFixed(4)}, {Number(fire.longitude).toFixed(4)}</span><span>{fire.confidence ?? '—'}</span><span>{fire.frp != null ? `${fire.frp} MW` : '—'}</span><span>{fire.distance_km != null ? `${fire.distance_km} km` : '—'}</span><span>{fire.satellite || '—'} · {fire.firms_source || '—'}</span></div>)}</div> : <EmptyState title="No FIRMS hotspots to display" text={fires.length ? "No hotspots match these filters." : "NASA FIRMS detections appear here when a valid server MAP_KEY is configured and the feed is available."}/>}
  </section>
}

function FiresPage() {
  const {data} = useDashboard()
  const fires = data?.fires || []
  return <div className="page"><div className="section-head"><div><span className="eyebrow">EVENT REGISTER</span><h2>Fire digital twin</h2></div><span className="source-chip">{fires.length} active events</span></div>{fires.length ? <div className="fire-table">{fires.map(f => <div className="fire-row" key={f.event_id}><div><b>{f.event_id}</b><small>{f.status} · {f.camera_id}</small></div><strong>{Math.round((f.confidence||0)*100)}%</strong><span>{f.visual_area_proxy != null ? `${Math.round(f.visual_area_proxy)} px² proxy` : 'Area proxy unavailable'}</span><span>{f.growth_rate != null ? `${f.growth_rate}%/min` : 'Growth unavailable'}</span><span className="risk-text">{f.risk_level || 'LOW'} · {Math.round(f.risk_score||0)}</span></div>)}</div> : <EmptyState title="No active fire events" text="This is a real-data view. Events will appear after a detector creates them; no synthetic events are shown."/>}</div>
}

function HistoricalPage() {
  const {data} = useDashboard()
  const [startDate, setStartDate] = useState(utcDayOffset(-4))
  const [days, setDays] = useState(5)
  const [history, setHistory] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const latitude = data?.live?.location?.latitude
  const longitude = data?.live?.location?.longitude
  const radiusKm = data?.satellite_meta?.radius_km
  async function loadHistory(event) {
    event?.preventDefault()
    setLoading(true)
    setError('')
    try {
      const locationQuery = latitude != null && longitude != null
        ? `&latitude=${encodeURIComponent(latitude)}&longitude=${encodeURIComponent(longitude)}&radius_km=${encodeURIComponent(radiusKm || 250)}`
        : ''
      const result = await getJSON(`/api/satellite/history?start_date=${encodeURIComponent(startDate)}&days=${days}&limit=50${locationQuery}`)
      setHistory(result)
    } catch (reason) {
      setError(reason.message || 'Could not load historical observations.')
    } finally {
      setLoading(false)
    }
  }
  useEffect(() => { loadHistory() }, [latitude, longitude, radiusKm])
  const metrics = history?.metrics || {}
  const rows = history?.records || []
  return <div className="page">
    <div className="section-head"><div><span className="eyebrow">NASA FIRMS · STANDARD PROCESSING</span><h2>Archived satellite detections</h2></div><span className="source-chip">{history?.status || 'WAITING'}</span></div>
    <div className="panel history-panel">
      <form className="history-query" onSubmit={loadHistory}>
        <label>Window start (UTC)<input type="date" value={startDate} max={utcDayOffset(0)} onChange={event => setStartDate(event.target.value)}/></label>
        <label>Window length<select value={days} onChange={event => setDays(Number(event.target.value))}>{[1,2,3,4,5].map(value => <option key={value} value={value}>{value} day{value > 1 ? 's' : ''}</option>)}</select></label>
        <button type="submit" disabled={loading}>{loading ? 'Querying NASA archive…' : 'Query verified archive'}</button>
      </form>
      <p className="archive-scope">{latitude != null ? `Monitoring ${Number(latitude).toFixed(4)}, ${Number(longitude).toFixed(4)} · within ${radiusKm || 250} km.` : 'Using the server-configured monitoring site.'} NASA FIRMS Standard Processing archive · {history?.source || 'NASA FIRMS'}.</p>
      {error && <div className="api-warning">{error}</div>}
      {!history && !error && <div className="notice">Loading archived observations…</div>}
      {history?.status === 'CONFIG_REQUIRED' && <div className="api-warning">Historical data is disabled until FIREGUARD_FIRMS_MAP_KEY is set in the API server’s .env file. NASA FIRMS provides a free key; the dashboard never stores or returns provider secrets.</div>}
      {history?.status === 'UNAVAILABLE' && <div className="api-warning">NASA FIRMS archive is unavailable for this query. Source status: {Object.entries(history.sources || {}).map(([source, status]) => `${source}: ${status}`).join(' · ') || 'No source response'}.</div>}
      {history?.status === 'AVAILABLE' && <>
        <p className="disclaimer">These are source records returned by NASA FIRMS, not independently verified wildfire incidents. Search uses your monitoring coordinates and exact radius; maximum archive window is five days.</p>
        <div className="metric-grid history-metrics"><Stat label="Returned observations" value={metrics.returned_records}/><Stat label="High-confidence detections" value={metrics.high_confidence_records}/><Stat label="Mean FRP" value={metrics.mean_frp_mw} unit={metrics.mean_frp_mw != null ? 'MW' : ''}/><Stat label="FRP observations measured" value={metrics.frp_sample_count}/></div>
        <div className="hotspot-table history-table"><div className="hotspot-row hotspot-header"><span>Observed (UTC)</span><span>Latitude / longitude</span><span>Confidence</span><span>FRP</span><span>Source</span><span>Satellite / instrument</span></div>{rows.map((record, index) => <div className="hotspot-row" key={record.event_id || index}><span>{record.acq_date || '—'} {String(record.acq_time || '').padStart(4, '0')}</span><span>{Number(record.latitude).toFixed(4)}, {Number(record.longitude).toFixed(4)}</span><span>{record.confidence ?? '—'}</span><span>{record.frp != null ? `${record.frp} MW` : '—'}</span><span>{record.firms_source || 'NASA FIRMS'}</span><span>{record.satellite || '—'} · {record.instrument || '—'}</span></div>)}</div>
        {!rows.length && <EmptyState title="No detections in this archive window" text="Try another date or window. A zero-result search is not evidence that no fire occurred."/>}
        <p className="disclaimer">{metrics.metric_scope}</p>
      </>}
    </div>
  </div>
}

function MonitoringPage() {
  const dashboard = useDashboard()
  const [site, setSite] = useState({latitude: '', longitude: '', radius_km: 250})
  const [providers, setProviders] = useState(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')
  useEffect(() => {
    getJSON('/api/system/location').then(result => {
      setSite({latitude: result.latitude ?? '', longitude: result.longitude ?? '', radius_km: result.radius_km ?? 250})
      setProviders(result.providers)
    }).catch(reason => setError(reason.message || 'Could not load monitoring configuration.'))
      .finally(() => setLoading(false))
  }, [])
  async function saveLocation(event) {
    event.preventDefault()
    setSaving(true)
    setError('')
    setMessage('')
    try {
      const response = await fetch(`${API}/api/system/location`, {
        method: 'PUT',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({latitude: Number(site.latitude), longitude: Number(site.longitude), radius_km: Number(site.radius_km)}),
      })
      const body = await response.json().catch(() => ({}))
      if (!response.ok) throw new Error(body.detail || `${response.status} ${response.statusText}`)
      setMessage(`Monitoring site saved at ${body.latitude}, ${body.longitude}. Live sources are refreshing.`)
      dashboard?.refresh()
      const updated = await getJSON('/api/system/location')
      setProviders(updated.providers)
    } catch (reason) {
      setError(reason.message || 'Could not save monitoring location.')
    } finally {
      setSaving(false)
    }
  }
  const providerRows = [
    ['Weather', providers?.weather, providers?.weather_status, 'Global provider; no API key required'],
    ['Visibility', 'Open-Meteo', dashboard?.data?.environment?.visibility_status, 'Current visibility when available; timestamped'],
    ['Air quality', 'Open-Meteo', providers?.air_quality, 'PM2.5 and PM10 when available'],
    ['Terrain elevation', 'Open-Meteo Elevation', providers?.terrain, 'No API key required'],
    ['NASA FIRMS', 'VIIRS / MODIS', providers?.firms, providers?.firms_key_configured ? 'Server key configured' : 'Server key required'],
    ['NASA GIBS / NDVI', 'MODIS archive', 'AVAILABLE', 'Keyless satellite imagery'],
  ]
  return <div className="page">
    <div className="section-head"><div><span className="eyebrow">SITE & PROVIDER CONFIGURATION</span><h2>Monitoring site</h2></div><span className="source-chip">{loading ? 'LOADING' : providers?.firms_key_configured ? 'PROVIDERS CONFIGURED' : 'ACTION REQUIRED'}</span></div>
    <div className="dashboard-grid monitoring-grid">
      <div className="panel"><span className="eyebrow">GEOGRAPHIC SCOPE</span><h2>Choose the point to monitor</h2>
        <form className="site-form" onSubmit={saveLocation}>
          <label>Latitude<input required type="number" min="-90" max="90" step="any" value={site.latitude} onChange={event => setSite({...site, latitude: event.target.value})} placeholder="e.g. 12.9716"/></label>
          <label>Longitude<input required type="number" min="-180" max="180" step="any" value={site.longitude} onChange={event => setSite({...site, longitude: event.target.value})} placeholder="e.g. 77.5946"/></label>
          <label>FIRMS search radius (km)<input required type="number" min="1" max="5000" step="1" value={site.radius_km} onChange={event => setSite({...site, radius_km: event.target.value})}/></label>
          <button type="submit" disabled={saving || loading}>{saving ? 'Saving site…' : 'Save monitoring site'}</button>
        </form>
        {message && <p className="inline-success">{message}</p>}{error && <p className="inline-error">{error}</p>}
        <p className="disclaimer">Saved on the API host for subsequent restarts. This site coordinate is an area-of-interest centre, not a UAV GPS fix. No camera stream is implied.</p>
      </div>
      <div className="panel"><span className="eyebrow">INTEGRATION STATUS</span><h2>Weather & satellite providers</h2>
        <div className="provider-cards">{providerRows.map(([name, source, status, note]) => <div className="provider-card" key={name}><div><b>{name}</b><small>{source}</small></div><span className={`provider-status ${status === 'LIVE' || status === 'AVAILABLE' ? 'is-live' : ''}`}>{status || 'WAITING'}</span><p>{note}</p></div>)}</div>
      </div>
    </div>
    <div className="panel credentials-note"><span className="eyebrow">CREDENTIAL SAFETY</span><h2>Provider keys</h2><p>For safety, API keys are not entered into or returned by the dashboard. Add them to the API server’s local <code>.env</code> file, then restart the API. NASA FIRMS archive access needs <code>FIREGUARD_FIRMS_MAP_KEY</code>. MET Norway, Open-Meteo weather, elevation, and NASA GIBS imagery do not need a provider key.</p><a href="https://firms.modaps.eosdis.nasa.gov/api/map_key" target="_blank" rel="noreferrer">Request a free NASA FIRMS MAP_KEY ↗</a></div>
  </div>
}

function EnvironmentPage() {
  const {data} = useDashboard()
  const environment = data?.environment || {}
  const observations = (data?.environment_history || []).filter(item => item.temperature != null).map(item => ({
    time: new Date(item.timestamp).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'}),
    score: item.temperature,
  }))
  return <div className="page"><div className="section-head"><div><span className="eyebrow">WEATHER + SENSOR FUSION</span><h2>Environmental intelligence</h2></div><span className="source-chip">{environment.status || 'UNAVAILABLE'} · {environment.source || 'NO PROVIDER'}</span></div><div className="panel observation-summary"><div><span className="eyebrow">CURRENT OBSERVATION</span><b>{environment.timestamp ? new Date(environment.timestamp).toLocaleString() : 'No observation timestamp'}</b></div><div><span>Freshness</span><b>{environment.freshness || 'UNAVAILABLE'}</b></div><div><span>Air quality</span><b>{environment.air_quality_status || 'UNAVAILABLE'}</b></div><div><span>Visibility source</span><b>{environment.visibility_source || 'UNAVAILABLE'} · {environment.visibility_status || 'UNAVAILABLE'}</b></div></div><EnvironmentCards env={environment}/><div className="panel"><div className="panel-head"><div><span className="eyebrow">OBSERVED HISTORY</span><h2>Measured temperature timeline</h2></div></div>{observations.length ? <RiskTimeline data={observations}/> : <EmptyState title="No historical weather measurements" text="This process has not collected any timestamped temperature observations yet. Historical observations are not synthesized."/>}</div></div>
}

function SpreadPage() {
  const {data} = useDashboard()
  const [horizon,setHorizon]=useState(30)
  const [forecast,setForecast]=useState(null)
  const [forecastError,setForecastError]=useState('')
  const fire=data?.fires?.[0]
  useEffect(() => {
    let cancelled = false
    setForecast(null)
    setForecastError('')
    if (!fire || horizon === 0) return () => { cancelled = true }
    getJSON(`/api/forecast/${encodeURIComponent(fire.event_id)}?horizon_minutes=${horizon}`)
      .then(result => { if (!cancelled) setForecast(result) })
      .catch(error => { if (!cancelled) setForecastError(error.message) })
    return () => { cancelled = true }
  }, [fire?.event_id, horizon])
  const environment = data?.environment || {}
  return <div className="page"><div className="section-head"><div><span className="eyebrow">WIND-ADVECTION BASELINE</span><h2>Spread estimate</h2></div><div className="segmented">{[0,30,60,120].map(v=><button key={v} className={(horizon===v?'selected ':'')} onClick={()=>setHorizon(v)}>{v===0?'Current':`+${v} min`}</button>)}</div></div><div className="forecast-grid"><div className="panel forecast-map"><div className="map-canvas">{fire || horizon===0 ? <LiveMap location={data?.live?.location} radiusKm={data?.satellite_meta?.radius_km} fire={fire} satelliteFires={data?.satellite_fires || []} environment={environment} forecast={forecast}/> : <div className="map-empty"><b>Forecast unavailable</b><span>A confirmed, geolocated fire and current wind readings are required.</span></div>}</div></div><div className="panel"><span className="eyebrow">{horizon===0?'CURRENT OBSERVATION':'FORECAST STATUS'}</span><h2>{horizon===0?'Observed conditions':`+${horizon} minute horizon`}</h2><div className="forecast-facts">{horizon===0 ? <><div><span>Observation status</span><b>{environment.status || 'UNAVAILABLE'} · {environment.freshness || 'NO FRESHNESS'}</b></div><div><span>Observed at</span><b>{environment.timestamp ? new Date(environment.timestamp).toLocaleString() : 'NOT REPORTED'}</b></div><div><span>Temperature / humidity</span><b>{environment.temperature != null ? `${environment.temperature} °C` : '—'} / {environment.humidity != null ? `${environment.humidity}%` : '—'}</b></div><div><span>Wind</span><b>{environment.wind_speed != null ? `${environment.wind_speed} km/h` : '—'} · {environment.wind_direction != null ? `${environment.wind_direction}°` : 'direction unavailable'}</b></div><div><span>Visibility</span><b>{environment.visibility != null ? `${environment.visibility} km` : 'NOT AVAILABLE'}</b></div><div><span>Location</span><b>{data?.live?.location?.latitude != null ? `${data.live.location.latitude}, ${data.live.location.longitude}` : 'NOT CONFIGURED'}</b></div></> : <><div><span>Event</span><b>{fire?.event_id || 'UNAVAILABLE'}</b></div><div><span>Wind input</span><b>{environment.wind_speed != null && environment.wind_direction != null ? `${environment.wind_speed} km/h · from ${environment.wind_direction}°` : 'UNAVAILABLE'}</b></div><div><span>Model</span><b>{forecast?.model || (forecastError ? 'UNAVAILABLE' : 'WAITING')}</b></div><div><span>Projected drift</span><b>{forecast ? `${forecast.drift_km} km` : forecastError || '—'}</b></div><div><span>Projected point</span><b>{forecast ? `${forecast.projected_latitude.toFixed(4)}, ${forecast.projected_longitude.toFixed(4)}` : '—'}</b></div></>}</div><div className="notice">{horizon===0 ? 'Live values above are sourced from the displayed provider and observation time. Missing fields are not filled with estimates.' : 'This baseline advects the fire’s point location using the latest observed wind speed and direction. It does not predict a fire perimeter and is not operationally validated; future weather, fuel, terrain, and measured spread are not modeled.'}</div></div></div></div>
}

function AlertsPage() {
  const {data, refresh} = useDashboard()
  const alerts = data?.alerts || []
  const [error, setError] = useState('')
  async function acknowledge(alertId) {
    setError('')
    try {
      const response = await fetch(`${API}/api/alerts/${encodeURIComponent(alertId)}/acknowledge`, {method: 'POST'})
      const body = await response.json().catch(() => ({}))
      if (!response.ok) throw new Error(body.detail || `${response.status} ${response.statusText}`)
      await refresh()
    } catch (reason) {
      setError(reason.message || 'Could not acknowledge alert.')
    }
  }
  return <div className="page"><div className="section-head"><div><span className="eyebrow">RESPONSE WORKFLOW</span><h2>Alert center</h2></div><span className="source-chip">{alerts.length} alerts · {alerts.filter(alert => !alert.acknowledged).length} open</span></div>{error && <div className="api-warning">{error}</div>}{alerts.length ? <div className="alerts">{alerts.map(a=><div className="alert-card" key={a.alert_id}><div className="alert-level">{a.level}</div><div className="alert-content"><b>{a.title}</b><p>{a.message}</p><small>{new Date(a.created_at).toLocaleString()} · {a.acknowledged ? `Acknowledged ${a.acknowledged_at ? new Date(a.acknowledged_at).toLocaleString() : ''}` : 'Open'}</small></div>{!a.acknowledged && <button className="ack-button" onClick={() => acknowledge(a.alert_id)}>Acknowledge</button>}</div>)}</div> : <EmptyState title="No alerts" text="Alerts are created from qualifying fire/smoke detections. Live satellite hotspots remain separate observations and do not automatically create incident alerts."/>}</div>
}

function DetectorInferencePanel({model, refresh}) {
  const [file, setFile] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  async function runInference(event) {
    event.preventDefault()
    if (!file) return

    setBusy(true)
    setError('')
    setResult(null)
    try {
      const body = new FormData()
      body.append('file', file)
      const response = await fetch(`${API}/api/detection/image`, {method: 'POST', body})
      const payload = await response.json().catch(() => ({}))
      if (!response.ok) {
        const detail = payload.detail
        throw new Error(typeof detail === 'string' ? detail : detail?.message || `${response.status} ${response.statusText}`)
      }
      setResult(payload)
      await refresh()
    } catch (reason) {
      setError(reason.message || 'Image inference failed.')
    } finally {
      setBusy(false)
    }
  }

  return <div className="panel">
    <div className="panel-head"><div><span className="eyebrow">RUN A REAL INFERENCE</span><h2>Test detector with an image</h2></div><span className="source-chip">{model?.status || 'MODEL STATUS UNKNOWN'}</span></div>
    <form onSubmit={runInference} className="inference-form">
      <label htmlFor="detector-image">Image file (JPEG, PNG, WebP or BMP)</label>
      <input id="detector-image" type="file" accept=".jpg,.jpeg,.png,.webp,.bmp" onChange={event => setFile(event.target.files?.[0] || null)} required/>
      <button type="submit" disabled={!file || busy || model?.status !== 'LOADED'}>{busy ? 'Running inference…' : 'Run inference'}</button>
    </form>
    {model?.status !== 'LOADED' && <p className="api-warning">The detector model is not loaded. Check the API model status before running inference.</p>}
    {error && <p className="api-warning">{error}</p>}
    {result && <div className="facts inference-result">
      <div><span>Inference completed</span><b>{result.latency_ms} ms · {result.count} predicted box(es)</b></div>
      {result.detections.map(detection => <div key={detection.detection_id}><span>{detection.class_name}</span><b>{(Number(detection.confidence) * 100).toFixed(1)}% confidence</b></div>)}
      {!result.detections.length && <div><span>Predictions</span><b>No boxes above the configured confidence threshold</b></div>}
    </div>}
    <p className="disclaimer">Only prediction metadata and bounding boxes are saved to the local database; the uploaded image itself is not retained. Model confidence and prediction counts are not precision/recall. The currently loaded model must be trained on fire/smoke classes to make wildfire predictions.</p>
  </div>
}

function AnalyticsPage() {
  const {data, refresh} = useDashboard()
  const observations = data?.satellite_fires || []
  const runtime = data?.analytics?.model_metrics?.runtime || {}
  const validation = data?.analytics?.model_metrics?.validation
  const datasetReport = data?.analytics?.dataset_report
  const providerCheck = data?.analytics?.provider_check
  const frpValues = observations.map(item => Number(item.frp)).filter(value => Number.isFinite(value) && value >= 0)
  const highConfidence = observations.filter(item => ['h', 'high'].includes(String(item.confidence).toLowerCase()) || Number(item.confidence) >= 80).length
  const meanFrp = frpValues.length ? (frpValues.reduce((sum, value) => sum + value, 0) / frpValues.length).toFixed(2) : null
  const risk = data?.location_risk
  const percent = value => value != null && Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(1)}%` : null
  return <div className="page">
    <div className="section-head"><div><span className="eyebrow">MEASURED OBSERVATIONS</span><h2>Location analytics</h2></div><span className="source-chip">{data?.environment?.source || 'NO WEATHER SOURCE'}</span></div>
    <div className="metric-grid">
      <Stat label="Current FIRMS observations" value={observations.length}/>
      <Stat label="High confidence detections" value={highConfidence}/>
      <Stat label="Mean FRP in returned sample" value={meanFrp} unit={meanFrp != null ? 'MW' : ''}/>
      <Stat label="Local weather-risk index" value={risk?.status === 'ESTIMATE' ? risk.score : risk?.status || 'UNAVAILABLE'} unit={risk?.status === 'ESTIMATE' ? '/100' : ''}/>
    </div>
    <div className="dashboard-grid analytics-grid">
      <div className="panel"><div className="panel-head"><div><span className="eyebrow">MEASURED SYSTEM PERFORMANCE</span><h2>Persisted detector inference</h2></div><span className="source-chip">{runtime.sample_count > 0 ? 'MEASURED' : 'WAITING FOR INFERENCE'}</span></div>
        <div className="metric-grid">
          <Stat label="Completed image inferences" value={runtime.sample_count ?? 0}/>
          <Stat label="Stored predicted boxes" value={runtime.total_detections ?? 0}/>
          <Stat label="Mean inference latency" value={runtime.mean_latency_ms} unit="ms"/>
          <Stat label="Observed single-image throughput" value={runtime.single_image_fps_estimate} unit="FPS"/>
        </div>
        {Object.keys(runtime.detections_by_class || {}).length > 0 && <div className="facts">{Object.entries(runtime.detections_by_class).map(([name, count]) => <div key={name}><span>{name} predictions</span><b>{count}</b></div>)}</div>}
        <p className="disclaimer">{runtime.scope || 'Throughput and prediction counts are measured from completed inference calls.'} This is not precision, recall, or validated fire-detection accuracy. Last inference: {runtime.last_inference_at ? new Date(runtime.last_inference_at).toLocaleString() : 'not yet recorded'}.</p>
      </div>
      <DetectorInferencePanel model={data?.model} refresh={refresh}/>
      <div className="panel"><div className="panel-head"><div><span className="eyebrow">LABELED EVALUATION RUN</span><h2>Detector accuracy metrics</h2></div><span className="source-chip">{validation?.split ? `${validation.split.toUpperCase()} SPLIT` : 'NOT EVALUATED'}</span></div>
        {validation?.status === 'INVALID_REPORT' ? <div className="api-warning">{validation.error}</div> : validation ? <>
          {validation.evaluation_label && <p className="disclaimer">{validation.evaluation_label}</p>}
          <div className="metric-grid">
            <Stat label="Precision" value={percent(validation.precision)}/>
            <Stat label="Recall" value={percent(validation.recall)}/>
            <Stat label="F1" value={percent(validation.f1)}/>
            <Stat label="mAP@50" value={percent(validation.mAP50)}/>
            <Stat label="mAP@50–95" value={percent(validation.mAP50_95)}/>
            <Stat label="Evaluation FPS" value={validation.fps}/>
            <Stat label="Inference FPS" value={validation.inference_fps}/>
          </div>
          <h3>Per-class evaluation</h3>
          {Object.keys(validation.per_class || {}).length ? <div className="facts">
            {Object.entries(validation.per_class).map(([id, metrics]) => <div key={id}><span>{metrics.name || `class_${id}`} · Precision {percent(metrics.precision) ?? 'N/A'} · Recall {percent(metrics.recall) ?? 'N/A'} · F1 {percent(metrics.f1) ?? 'N/A'} · AP@0.5 {percent(metrics.ap50) ?? 'N/A'} · AP@0.5:0.95 {percent(metrics.ap50_95) ?? 'N/A'}</span><b>Per-class validation result</b></div>)}
          </div> : <p className="disclaimer">This evaluation report contains no per-class results.</p>}
          <div className="facts"><div><span>Weights</span><b>{validation.weights || '—'}</b></div><div><span>Evaluated</span><b>{validation.evaluated_at ? new Date(validation.evaluated_at).toLocaleString() : '—'}</b></div></div>
        </> : <>
          <div className="metric-grid">
            <Stat label="Precision" value="N/A"/>
            <Stat label="Recall" value="N/A"/>
            <Stat label="F1" value="N/A"/>
            <Stat label="mAP@0.5" value="N/A"/>
            <Stat label="mAP@0.5:0.95" value="N/A"/>
            <Stat label="Validation FPS" value="N/A"/>
          </div>
          <h3>Per-class evaluation</h3>
          <div className="facts">
            {['fire', 'smoke'].map(name => <div key={name}><span>{name} · Precision N/A · Recall N/A · F1 N/A · AP@0.5 N/A · AP@0.5:0.95 N/A</span><b>Not evaluated</b></div>)}
          </div>
          <p className="disclaimer">These metrics require a fire/smoke-trained model and labeled validation images. Inference predictions alone cannot determine true positives or false positives, so accuracy metrics are not guessed. The overall row is the validation summary across dataset classes.</p>
        </>}
      </div>
      <div className="panel"><div className="panel-head"><div><span className="eyebrow">DATASET INSPECTION</span><h2>Images and annotations</h2></div><span className="source-chip">{datasetReport?.quality?.status || datasetReport?.status || 'NOT INSPECTED'}</span></div>
        {datasetReport?.status === 'SOURCE_MISSING' ? <p className="disclaimer">{datasetReport.message}</p> : datasetReport?.counts ? <div className="facts">
          <div><span>Images</span><b>{datasetReport.counts.images ?? '—'}</b></div><div><span>Objects</span><b>{datasetReport.counts.annotations ?? '—'}</b></div>
          <div><span>Classes</span><b>{datasetReport.counts.classes_detected ?? '—'}</b></div><div><span>Quality issues</span><b>{datasetReport.quality?.issue_count ?? '—'}</b></div>
          <div><span>Dataset</span><b>{datasetReport.dataset_name || '—'}</b></div><div><span>Inspected</span><b>{datasetReport.inspected_at ? new Date(datasetReport.inspected_at).toLocaleString() : '—'}</b></div>
        </div> : <EmptyState title="No inspection report" text="Run scripts/inspect_dataset.py on the dataset root that contains the source images and annotations."/>}
      </div>
      <div className="panel"><div className="panel-head"><div><span className="eyebrow">LIVE PROVIDER VERIFICATION</span><h2>Source check</h2></div><span className="source-chip">{providerCheck?.status || 'NOT RUN'}</span></div>
        {providerCheck?.status === 'INVALID_REPORT' ? <div className="api-warning">{providerCheck.error}</div> : providerCheck?.checks?.length ? <div className="facts">
          {providerCheck.checks.map((check, index) => <div key={`${check.name}-${index}`}><span>{check.name} · {check.status}</span><b>{check.details}</b></div>)}
          <div><span>Checked</span><b>{providerCheck.checked_at ? new Date(providerCheck.checked_at).toLocaleString() : '—'}</b></div>
        </div> : <EmptyState title="No provider run log" text="Run scripts/verify_live_sources.py to test configured upstream providers and save a report."/>}
      </div>
      <div className="panel"><span className="eyebrow">WEATHER-BASED RISK</span><h2>Observed index trend</h2><RiskTimeline data={data?.risk_history || []}/><p className="disclaimer">A relative weather index based on available temperature, humidity, wind and precipitation readings. It is not the chance or probability of a fire; no validated fire-occurrence model is configured.</p></div>
      <div className="panel"><span className="eyebrow">DATA QUALITY</span><h2>Live observation coverage</h2><div className="facts">
        <div><span>Monitoring site</span><b>{data?.live?.location?.latitude != null ? `${Number(data.live.location.latitude).toFixed(4)}, ${Number(data.live.location.longitude).toFixed(4)}` : 'NOT CONFIGURED'}</b></div>
        <div><span>Weather</span><b>{data?.environment?.status || 'UNAVAILABLE'} · {data?.environment?.source || '—'}</b></div>
        <div><span>Observation time</span><b>{data?.environment?.timestamp ? new Date(data.environment.timestamp).toLocaleString() : 'NOT AVAILABLE'}</b></div>
        <div><span>FIRMS feed</span><b>{data?.satellite_meta?.satellite_status || 'UNAVAILABLE'}</b></div>
        <div><span>FIRMS search area</span><b>{data?.satellite_meta?.radius_km != null ? `${data.satellite_meta.radius_km} km radius` : 'NOT AVAILABLE'}</b></div>
        <div><span>Terrain</span><b>{data?.providers?.terrain || 'UNAVAILABLE'}</b></div>
      </div><p className="disclaimer">Satellite counts and fire-radiative-power summaries describe the returned FIRMS sample only. Satellite detections are not independently verified incidents.</p></div>
    </div>
  </div>
}

function SystemPage() {
  const {data}=useDashboard()
  return <div className="page"><div className="section-head"><div><span className="eyebrow">SOURCE STATUS</span><h2>Data availability</h2></div><span className="source-chip">{data?.health?.status === 'healthy' ? 'API CONNECTED' : 'API UNAVAILABLE'}</span></div>
    <div className="dashboard-grid"><div className="panel"><span className="eyebrow">LIVE INPUTS</span><h2>Provider health</h2><SourceList data={data}/><p className="disclaimer">Statuses reflect the latest provider response. Missing credentials or coverage are shown as unavailable; sensor readings and satellite alerts are not fabricated.</p></div>
      <div className="panel"><span className="eyebrow">CURRENT MONITOR</span><h2>Area status</h2><div className="facts"><div><span>Coordinates</span><b>{data?.live?.location?.latitude != null ? `${Number(data.live.location.latitude).toFixed(5)}, ${Number(data.live.location.longitude).toFixed(5)}` : 'NOT CONFIGURED'}</b></div><div><span>Weather source / age</span><b>{data?.environment?.source || '—'} · {data?.environment?.freshness || 'UNAVAILABLE'}</b></div><div><span>Weather observed</span><b>{data?.environment?.timestamp ? new Date(data.environment.timestamp).toLocaleString() : '—'}</b></div><div><span>FIRMS radius</span><b>{data?.satellite_meta?.radius_km != null ? `${data.satellite_meta.radius_km} km` : '—'}</b></div></div></div></div>
  </div>
}

function App() {
  const dashboard = useDashboardState()
  return <DashboardContext.Provider value={dashboard}><BrowserRouter><Shell data={dashboard.data} error={dashboard.error}><Routes>
    <Route path="/" element={<CommandCenter data={dashboard.data}/>}/>
    <Route path="/map" element={<MapPage/>}/><Route path="/fires" element={<FiresPage/>}/>
    <Route path="/environment" element={<EnvironmentPage/>}/><Route path="/spread" element={<SpreadPage/>}/>
    <Route path="/history" element={<HistoricalPage/>}/><Route path="/alerts" element={<AlertsPage/>}/>
    <Route path="/analytics" element={<AnalyticsPage/>}/><Route path="/system" element={<SystemPage/>}/>
    <Route path="/monitoring" element={<MonitoringPage/>}/>
  </Routes></Shell></BrowserRouter></DashboardContext.Provider>
}
export default App
