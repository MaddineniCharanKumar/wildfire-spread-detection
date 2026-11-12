from __future__ import annotations

from typing import Any


class RiskService:
    def __init__(self):
        pass

    def score_risk(self, confidence: float, wind_speed: float | None, humidity: float | None, growth_rate: float) -> dict[str, Any]:
        score = confidence * 60
        score += (wind_speed or 0) * 0.8
        score += max(0, (50 - (humidity or 0)) * 0.6)
        score += max(0, growth_rate * 10)

        if score >= 85:
            level = "CRITICAL"
        elif score >= 70:
            level = "HIGH"
        elif score >= 50:
            level = "MEDIUM"
        else:
            level = "LOW"

        return {
            "risk_score": round(score, 2),
            "level": level,
            "explanation": {
                "fire_confidence": confidence,
                "wind_speed": wind_speed,
                "humidity": humidity,
                "growth_rate": growth_rate,
                "score": round(score, 2),
            },
        }
