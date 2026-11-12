from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AlertRule:
    level: int
    title: str
    message: str
    threshold: float


@dataclass
class EarlyWarningEngine:
    rules: list[AlertRule] = field(default_factory=lambda: [
        AlertRule(1, "Persistent weak detection", "Detection persisted but confidence remains low.", 0.55),
        AlertRule(2, "Growth trend detected", "The bounding-box proxy is increasing over time.", 0.70),
        AlertRule(3, "Environmental risk rising", "Conditions indicate increasing risk exposure.", 0.80),
    ])

    def evaluate(self, confidence: float, growth: float, environmental_risk: float) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        if confidence >= self.rules[0].threshold:
            matches.append({"level": 1, "title": self.rules[0].title, "message": self.rules[0].message})
        if growth >= self.rules[1].threshold:
            matches.append({"level": 2, "title": self.rules[1].title, "message": self.rules[1].message})
        if environmental_risk >= self.rules[2].threshold:
            matches.append({"level": 3, "title": self.rules[2].title, "message": self.rules[2].message})
        return matches
