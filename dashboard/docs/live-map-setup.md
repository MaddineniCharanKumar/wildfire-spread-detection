# Live map (NASA GIBS + FIRMS): no Google, no map key

- Imagery: NASA GIBS WMS (VIIRS true colour + thermal anomalies), keyless.
- Hotspots: NASA FIRMS via the API server. Set FIREGUARD_FIRMS_MAP_KEY, FIREGUARD_LATITUDE, FIREGUARD_LONGITUDE in the root .env.
- Map centre comes from the API, so coordinates are configured in one place.
- If FIRMS is unconfigured/rejected/unreachable the Live Map page says so; nothing is simulated.
