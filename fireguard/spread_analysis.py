from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List


@dataclass
class TemporalDetection:
    confidence: float
    bbox: tuple[float, float, float, float]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class TemporalVerifier:
    """Persistence logic for weak detections and confirmation after a window."""

    def __init__(
        self,
        confidence_threshold: float = 0.35,
        temporal_window: int = 12,
        minimum_persistence: int = 3,
        ema_alpha: float = 0.35,
    ):
        self.confidence_threshold = confidence_threshold
        self.temporal_window = temporal_window
        self.minimum_persistence = minimum_persistence
        self.ema_alpha = ema_alpha
        self.history: List[TemporalDetection] = []

    def push(self, confidence: float, bbox: tuple[float, float, float, float]) -> bool:
        now = datetime.now(timezone.utc)
        self.history.append(TemporalDetection(confidence=confidence, bbox=bbox, timestamp=now))
        window = self.history[-self.temporal_window:]
        if len(window) < self.minimum_persistence:
            return False

        avg_confidence = sum(d.confidence for d in window) / len(window)
        if avg_confidence >= self.confidence_threshold:
            return True
        return False

    def ema_smooth(self, previous: float, current: float) -> float:
        return self.ema_alpha * current + (1.0 - self.ema_alpha) * previous

    def reset(self) -> None:
        self.history.clear()
