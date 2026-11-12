"""Verify configured live providers and persist a sanitized run report."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fireguard.config import settings
from fireguard.environment import MetNoProvider, NWSProvider, OpenMeteoProvider
from fireguard.live_data import fetch_air_quality, fetch_elevation, fetch_firms


async def main() -> int:
    parser = argparse.ArgumentParser(description="Check configured live data providers and save results.")
    parser.add_argument("--output", type=Path, default=Path("reports/live_provider_report.json"))
    parser.add_argument(
        "--skip-firms",
        action="store_true",
        help="Skip NASA FIRMS so its configured API key is not used for this run.",
    )
    args = parser.parse_args()

    checks: list[dict[str, Any]] = []
    choice = settings.environment_provider.lower()
    required_failures = 0
    skipped_checks = 0

    def show(
        name: str,
        ok: bool,
        detail: str,
        *,
        required: bool = True,
        skipped: bool = False,
        measurements: dict[str, Any] | None = None,
    ) -> None:
        nonlocal required_failures, skipped_checks
        status = "SKIP" if skipped else ("PASS" if ok else "FAIL")
        print(f"{status:<4}  {name:<20} {detail}")
        checks.append({
            "name": name,
            "status": status,
            "required": required,
            "details": detail,
        })
        if measurements:
            checks[-1]["measurements"] = measurements
        if not ok and required and not skipped:
            required_failures += 1
        if skipped:
            skipped_checks += 1

    async def run_provider(
        name: str,
        operation: Awaitable[dict[str, Any]],
        is_ok: Callable[[dict[str, Any]], bool],
        describe: Callable[[dict[str, Any]], str],
        *,
        required: bool = True,
    ) -> dict[str, Any] | None:
        try:
            payload = await operation
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            show(name, False, f"Provider request failed: {exc.__class__.__name__}", required=required)
            return None
        measurements = {
            key: payload[key]
            for key in (
                "status",
                "source",
                "timestamp",
                "temperature",
                "humidity",
                "wind_speed",
                "wind_direction",
                "air_quality_status",
                "pm25",
                "pm10",
                "terrain_status",
                "elevation_m",
                "satellite_status",
                "total_detections",
            )
            if key in payload
        }
        show(
            name,
            is_ok(payload),
            describe(payload),
            required=required,
            measurements=measurements,
        )
        return payload

    report: dict[str, Any] = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "location_configured": settings.latitude is not None and settings.longitude is not None,
        "environment_provider": choice,
        "checks": checks,
    }

    if settings.latitude is None or settings.longitude is None:
        show("Monitoring location", False, "Set FIREGUARD_LATITUDE and FIREGUARD_LONGITUDE.")
    else:
        lat, lon = settings.latitude, settings.longitude
        met_selected = choice in {"met_no", "met-no", "metno", "norway", "live"}

        if met_selected:
            await run_provider(
                "MET Norway weather",
                MetNoProvider(
                    settings.environment_timeout_seconds,
                    settings.environment_max_age_seconds,
                    settings.met_no_user_agent,
                ).current(lat, lon),
                lambda value: value.get("status") in {"LIVE", "RECENT"},
                lambda value: (
                    f"{value.get('status')} T={value.get('temperature')}C "
                    f"wind={value.get('wind_speed')}km/h obs={value.get('timestamp')}"
                ),
            )
        else:
            await run_provider(
                "Open-Meteo weather",
                OpenMeteoProvider(
                    settings.weather_api_url,
                    settings.environment_timeout_seconds,
                    settings.environment_max_age_seconds,
                ).current(lat, lon),
                lambda value: value.get("status") in {"LIVE", "RECENT"},
                lambda value: (
                    f"{value.get('status')} T={value.get('temperature')}C "
                    f"wind={value.get('wind_speed')}km/h obs={value.get('timestamp')}"
                ),
            )

        if not met_selected:
            await run_provider(
                "MET Norway weather",
                MetNoProvider(
                    settings.environment_timeout_seconds,
                    settings.environment_max_age_seconds,
                    settings.met_no_user_agent,
                ).current(lat, lon),
                lambda value: value.get("status") in {"LIVE", "RECENT"},
                lambda value: (
                    f"{value.get('status')} T={value.get('temperature')}C "
                    f"wind={value.get('wind_speed')}km/h obs={value.get('timestamp')}"
                ),
                required=False,
            )
        if choice not in {"open_meteo", "open-meteo"}:
            await run_provider(
                "Open-Meteo weather",
                OpenMeteoProvider(
                    settings.weather_api_url,
                    settings.environment_timeout_seconds,
                    settings.environment_max_age_seconds,
                ).current(lat, lon),
                lambda value: value.get("status") in {"LIVE", "RECENT"},
                lambda value: (
                    f"{value.get('status')} T={value.get('temperature')}C "
                    f"wind={value.get('wind_speed')}km/h obs={value.get('timestamp')}"
                ),
                required=False,
            )

        await run_provider(
            "NWS (U.S. only)",
            NWSProvider(
                settings.environment_timeout_seconds,
                settings.nws_user_agent,
            ).current(lat, lon),
            lambda value: value.get("status") in {"LIVE", "RECENT"},
            lambda value: f"{value.get('status')}",
            required=choice.startswith("nws"),
        )
        await run_provider(
            "Air quality",
            fetch_air_quality(lat, lon),
            lambda value: value.get("air_quality_status") == "LIVE",
            lambda value: (
                f"{value.get('air_quality_status')} PM2.5={value.get('pm25')} "
                f"PM10={value.get('pm10')}"
            ),
            required=False,
        )
        await run_provider(
            "Elevation",
            fetch_elevation(lat, lon),
            lambda value: value.get("terrain_status") == "LIVE",
            lambda value: f"{value.get('terrain_status')} {value.get('elevation_m')} m",
            required=False,
        )
        if args.skip_firms:
            show(
                "NASA FIRMS",
                True,
                "Skipped; configured API key was not used.",
                required=True,
                skipped=True,
            )
        else:
            await run_provider(
                "NASA FIRMS",
                fetch_firms(lat, lon, use_cache=False),
                lambda value: value.get("satellite_status") in {"LIVE", "PARTIAL"},
                lambda value: (
                    f"{value.get('satellite_status')} detections={value.get('total_detections')}"
                ).strip(),
            )
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.get(
                    "https://gibs.earthdata.nasa.gov/wms/epsg3857/best/wms.cgi"
                    "?service=WMS&version=1.3.0&request=GetCapabilities"
                )
            show("NASA GIBS imagery", response.status_code == 200, f"HTTP {response.status_code}")
        except httpx.HTTPError as exc:
            show("NASA GIBS imagery", False, f"Provider request failed: {exc.__class__.__name__}")

    report["status"] = (
        "FAIL" if required_failures else ("PARTIAL" if skipped_checks else "PASS")
    )
    report["required_failures"] = required_failures
    report["skipped_checks"] = skipped_checks
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nProvider report saved to {args.output}")
    if required_failures:
        print(f"{required_failures} required source(s) failing.")
    elif skipped_checks:
        print(f"Provider check completed with {skipped_checks} skipped source(s).")
    else:
        print("All required sources live.")
    return 0 if required_failures == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
