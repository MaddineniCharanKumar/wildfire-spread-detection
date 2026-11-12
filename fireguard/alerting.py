from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, List


@dataclass
class FireEvent:
    event_id: str
    camera_id: str
    source_type: str
    status: str = "DETECTED"
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    duration: float = 0.0
    risk_score: float = 0.0
    alert_level: int = 0
    confidence: float = 0.0
    growth_rate: float = 0.0
    spread_direction: str = "UNKNOWN"
    latitude: float | None = None
    longitude: float | None = None
    model_version: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def touch(self, confidence: float, growth_rate: float, direction: str) -> None:
        self.last_seen = datetime.now(timezone.utc)
        self.confidence = confidence
        self.growth_rate = growth_rate
        self.spread_direction = direction
        self.duration = (self.last_seen - self.start_time).total_seconds()


class EventManager:
    def __init__(self):
        self.events: List[FireEvent] = []

    def create(self, camera_id: str, source_type: str, *, model_version: str = "unknown") -> FireEvent:
        now = datetime.now(timezone.utc)
        event = FireEvent(
            event_id=f"event-{now.strftime('%Y%m%d%H%M%S%f')}",
            camera_id=camera_id,
            source_type=source_type,
            model_version=model_version,
            start_time=now,
            last_seen=now,
        )
        self.events.append(event)
        return event

    def update(self, event: FireEvent, confidence: float, growth_rate: float, direction: str) -> FireEvent:
        event.touch(confidence, growth_rate, direction)
        return event
