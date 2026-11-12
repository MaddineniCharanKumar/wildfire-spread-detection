from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


def risk_level(score: float) -> str:
    if score < 20:
        return "LOW"
    if score < 40:
        return "GUARDED"
    if score < 60:
        return "MODERATE"
    if score < 80:
        return "HIGH"
    return "CRITICAL"


@dataclass
class RiskWeights:
    visual: float = 0.25
    persistence: float = 0.15
    growth: float = 0.15
    wind: float = 0.10
    temperature: float = 0.10
    humidity: float = 0.10
    dryness: float = 0.10
    smoke: float = 0.05


def calculate_risk(features: dict[str, float], weights: RiskWeights | None = None) -> dict[str, float | str | dict]:
    weights = weights or RiskWeights()
    score = 0.0
    for key in weights.__annotations__:
        value = float(features.get(key, 0.0))
        score += min(max(value, 0.0), 100.0) * getattr(weights, key)
    score = min(100.0, max(0.0, score))
    return {"score": round(score, 2), "level": risk_level(score), "features": features}


def calculate_location_risk(environment: dict) -> dict:
    """Build a transparent weather-risk index; this is not a fire-probability model."""
    freshness = environment.get("freshness") or environment.get("status")
    base = {
        "status": "UNAVAILABLE",
        "score": None,
        "level": "UNAVAILABLE",
        "source": environment.get("source", "UNAVAILABLE"),
        "timestamp": environment.get("timestamp"),
        "basis": "Weather-only heuristic index; not a probability or validated forecast.",
        "factors": {},
    }
    if environment.get("demo") or freshness == "DEMO_MODE":
        return {**base, "status": "DEMO_MODE"}
    if freshness not in {"LIVE", "RECENT"}:
        return base

    weighted_factors = (
        ("humidity", 0.40, lambda value: 100 - value),
        ("temperature", 0.25, lambda value: (value - 15) * 3),
        ("wind_speed", 0.20, lambda value: value * 2),
        ("rainfall", 0.15, lambda value: 100 - value * 20),
    )
    factors: dict[str, float] = {}
    weighted_score = 0.0
    available_weight = 0.0
    for key, weight, transform in weighted_factors:
        value = environment.get(key)
        if value is None:
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if not isfinite(numeric_value):
            continue
        if key == "humidity":
            numeric_value = min(max(numeric_value, 0), 100)
        elif key == "rainfall":
            numeric_value = max(numeric_value, 0)
        transformed = min(max(transform(numeric_value), 0), 100)
        factors[key] = round(transformed, 1)
        weighted_score += transformed * weight
        available_weight += weight

    if len(factors) < 2 or available_weight == 0:
        return {**base, "source": environment.get("source", "UNAVAILABLE")}

    score = round(weighted_score / available_weight, 1)
    return {
        **base,
        "status": "ESTIMATE",
        "score": score,
        "level": risk_level(score),
        "factors": factors,
        "inputs_used": list(factors),
    }
