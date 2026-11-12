from __future__ import annotations

import os
from pathlib import Path

for path in [
    Path("data/raw"),
    Path("data/processed"),
    Path("data/train"),
    Path("data/val"),
    Path("data/test"),
    Path("reports/false_positive_gallery"),
    Path("reports/false_negative_gallery"),
    Path("dashboard/src"),
    Path("services"),
    Path("api"),
    Path("websocket"),
]:
    path.mkdir(parents=True, exist_ok=True)
    (path / ".gitkeep").touch(exist_ok=True)

print("Created required directories.")
