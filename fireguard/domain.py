from __future__ import annotations

from dataclasses import dataclass


def area_proxy(bbox: tuple[float, float, float, float]) -> float:
    """Return a visual-area proxy only. This is not a calibrated physical area estimate."""
    x1, y1, x2, y2 = bbox
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    return width * height


def growth_rate(previous: float, current: float) -> float:
    if previous == 0:
        return 0.0
    return ((current - previous) / previous) * 100.0
