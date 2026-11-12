from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from fireguard.config import settings
from fireguard.models import _build_weight_name, load_yolo_model
from training.metrics import (
    save_validation_metrics,
    summarize_validation_metrics,
    verify_model_matches_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a YOLO model on a dataset.")
    parser.add_argument("--weights", type=str, default=None, help="Model weights (default: family+size from config)")
    parser.add_argument("--family", type=str, default=None, help="Model family (default: FIREGUARD_MODEL_FAMILY)")
    parser.add_argument("--size", type=str, default=None, help="Model size (default: FIREGUARD_MODEL_SIZE)")
    parser.add_argument("--data", type=str, required=True, help="Dataset YAML path")
    parser.add_argument("--split", type=str, default="val", help="Split to validate on: val, test, train")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0" if __import__("utils.device", fromlist=["detect_device"]).detect_device() == "cuda" else "cpu")
    parser.add_argument("--output", type=Path, default=Path("runs/validate"))
    args = parser.parse_args()

    family = args.family or settings.model_family
    size = args.size or settings.model_size
    custom_weights = args.weights or settings.model_weights

    weight_name = _build_weight_name(family, size)
    weights_path = custom_weights or weight_name

    args.output.mkdir(parents=True, exist_ok=True)

    model, meta = load_yolo_model(weights_path, args.device)
    if model is None:
        print(f"[ERROR] Failed to load model: {meta.get('error')}", file=sys.stderr)
        return 1
    try:
        class_names = verify_model_matches_dataset(getattr(model, "names", None), args.data)
    except (OSError, TypeError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print(f"[INFO] Validating with model: {weights_path}")
    metrics = model.val(data=args.data, split=args.split, imgsz=args.imgsz, device=args.device)

    payload = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "weights": weights_path,
        "family": family,
        "size": size,
        "data": args.data,
        "split": args.split,
        "imgsz": args.imgsz,
        "device": args.device,
        **summarize_validation_metrics(metrics, class_names),
    }
    report_path = save_validation_metrics(payload, args.output)
    print(f"Validation metrics saved to {report_path}")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())