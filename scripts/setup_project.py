from __future__ import annotations

import json
from pathlib import Path


def ensure_project_layout() -> None:
    for directory in [
        Path("data/raw"),
        Path("data/processed"),
        Path("data/train"),
        Path("data/val"),
        Path("data/test"),
        Path("reports"),
        Path("training"),
        Path("dashboard"),
        Path("fireguard"),
    ]:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ".gitkeep").touch(exist_ok=True)


ensure_project_layout()
print("Project layout ensured.")
