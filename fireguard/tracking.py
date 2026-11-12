from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class TrackingState:
    track_id: str
    camera_id: str
    bbox: tuple[float, float, float, float]
    centroid: tuple[float, float]
    velocity: float = 0.0
    area: float = 0.0
    growth_rate: float = 0.0
    trajectory: list[tuple[float, float]] = field(default_factory=list)
    duration: float = 0.0
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class Tracker:
    def __init__(self):
        self.tracks: dict[str, TrackingState] = {}

    def update(self, track_id: str, camera_id: str, bbox: tuple[float, float, float, float], centroid: tuple[float, float], timestamp: datetime | None = None):
        now = timestamp or datetime.now(timezone.utc)
        track = self.tracks.get(track_id)
        if track is None:
            track = TrackingState(track_id=track_id, camera_id=camera_id, bbox=bbox, centroid=centroid)
            self.tracks[track_id] = track
        else:
            dx = centroid[0] - track.centroid[0]
            dy = centroid[1] - track.centroid[1]
            track.velocity = (dx ** 2 + dy ** 2) ** 0.5
            track.bbox = bbox
            track.centroid = centroid
            track.trajectory.append(centroid)
            track.area = max(0.0, bbox[2] * bbox[3])
            track.duration = (now - track.last_seen).total_seconds()
            track.last_seen = now
        return track


tracker = Tracker()
