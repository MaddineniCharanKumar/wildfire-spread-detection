from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def detect_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def get_gpu_info() -> dict[str, Any]:
    try:
        import torch
        if not torch.cuda.is_available():
            return {"available": False, "reason": "CUDA unavailable"}
        props = torch.cuda.get_device_properties(0)
        return {
            "available": True,
            "device_count": torch.cuda.device_count(),
            "name": props.name,
            "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
            "cuda_version": torch.version.cuda,
            "pytorch_version": torch.__version__,
        }
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


def get_gpu_memory() -> dict[str, Any]:
    try:
        import torch
        if not torch.cuda.is_available():
            return {"available": False, "reason": "CUDA unavailable"}
        return {
            "available": True,
            "allocated_mb": round(torch.cuda.memory_allocated() / (1024 ** 2), 2),
            "reserved_mb": round(torch.cuda.memory_reserved() / (1024 ** 2), 2),
            "total_mb": round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 2), 2),
        }
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


def get_gpu_utilization() -> dict[str, Any]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=3,
        )
        parts = output.strip().split(",")
        if len(parts) < 4:
            return {"available": False, "reason": "Unexpected nvidia-smi output"}
        return {
            "available": True,
            "utilization_percent": float(parts[0]),
            "temperature_c": float(parts[1]),
            "memory_used_mb": float(parts[2]),
            "memory_total_mb": float(parts[3]),
        }
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


def configure_precision(device: str | None = None) -> dict[str, Any]:
    device_name = device or detect_device()
    return {
        "device": device_name,
        "cuda_available": device_name == "cuda",
        "fp16": device_name == "cuda",
        "mixed_precision": device_name == "cuda",
        "batch_inference_enabled": device_name == "cuda",
        "cpu_fallback": True,
        "multi_gpu": False,
    }


if __name__ == "__main__":
    print(json.dumps({
        "device": detect_device(),
        "gpu_info": get_gpu_info(),
        "gpu_memory": get_gpu_memory(),
        "gpu_utilization": get_gpu_utilization(),
        "precision": configure_precision(),
    }, indent=2))
