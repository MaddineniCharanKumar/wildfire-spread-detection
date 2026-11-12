from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class SpreadEstimate:
    centroid_motion: float
    bbox_expansion: float
    trajectory: List[Tuple[float, float]]
    direction: str
    confidence: float
    status: str


class SpreadAnalyzer:
    """Estimate visual spread direction and expansion using centroid and bbox changes."""

    def estimate(self, previous_centroid: tuple[float, float] | None, current_centroid: tuple[float, float] | None, previous_bbox: tuple[float, float, float, float] | None, current_bbox: tuple[float, float, float, float] | None) -> SpreadEstimate:
        if previous_centroid is None or current_centroid is None:
            return SpreadEstimate(0.0, 0.0, [], "UNKNOWN", 0.0, "STABLE")

        dx = current_centroid[0] - previous_centroid[0]
        dy = current_centroid[1] - previous_centroid[1]
        centroid_motion = (dx ** 2 + dy ** 2) ** 0.5

        prev_area = 0.0 if previous_bbox is None else previous_bbox[2] * previous_bbox[3]
        curr_area = 0.0 if current_bbox is None else current_bbox[2] * current_bbox[3]
        bbox_expansion = 0.0 if prev_area == 0 else ((curr_area - prev_area) / prev_area) * 100.0

        direction = "UNKNOWN"
        if abs(dx) > abs(dy):
            direction = "EAST" if dx >= 0 else "WEST"
        elif abs(dy) > 0:
            direction = "SOUTH" if dy >= 0 else "NORTH"

        confidence = min(1.0, max(0.0, (abs(centroid_motion) + max(0.0, bbox_expansion) / 100.0) / 2.0))
        if bbox_expansion > 30:
            status = "RAPIDLY_EXPANDING"
        elif bbox_expansion > 10:
            status = "EXPANDING"
        else:
            status = "STABLE"

        return SpreadEstimate(
            centroid_motion=centroid_motion,
            bbox_expansion=bbox_expansion,
            trajectory=[previous_centroid, current_centroid],
            direction=direction,
            confidence=confidence,
            status=status,
        )
