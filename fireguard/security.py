from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class GPUStatus:
    name: str | None = None
    vram: str | None = None
    utilization: float | None = None
    temperature: float | None = None
    cuda_version: str | None = None
    pytorch_version: str | None = None
    fps: float | None = None
    latency_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu_name": self.name,
            "vram": self.vram,
            "gpu_utilization": self.utilization,
            "temperature_c": self.temperature,
            "cuda_version": self.cuda_version,
            "pytorch_version": self.pytorch_version,
            "fps": self.fps,
            "latency_ms": self.latency_ms,
        }
