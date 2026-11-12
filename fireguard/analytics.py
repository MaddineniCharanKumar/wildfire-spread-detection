from __future__ import annotations

from typing import Any


class AnalysisWindow:
    def __init__(self, history: list[float] | None = None):
        self.history = history or []

    def append(self, value: float) -> None:
        self.history.append(value)

    def rolling_average(self, window_size: int = 5) -> float:
        if not self.history:
            return 0.0
        segment = self.history[-window_size:]
        return sum(segment) / len(segment)

    def state(self) -> str:
        if len(self.history) < 2:
            return "STABLE"
        delta = self.history[-1] - self.history[-2]
        if delta < -10:
            return "DECREASING"
        if delta > 10:
            return "GROWING"
        return "STABLE"


class AnalyticsEngine:
    def build_overview(self) -> dict[str, Any]:
        return {
            "total_events": 1,
            "active_events": 1,
            "alert_count": 2,
            "risk_level": "HIGH",
            "gpu_status": "READY",
            "source_status": "DEMO",
        }

    def build_historical(self) -> dict[str, Any]:
        return {
            "series": [
                {"time": "2026-09-19T00:00:00Z", "risk": 30, "confidence": 0.65},
                {"time": "2026-09-19T00:10:00Z", "risk": 45, "confidence": 0.72},
                {"time": "2026-09-19T00:20:00Z", "risk": 56, "confidence": 0.79},
            ]
        }


def build_overview():
    return AnalyticsEngine().build_overview()


def build_historical():
    return AnalyticsEngine().build_historical()
