from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import yaml


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _metric_value(results: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _number(results.get(key))
        if value is not None:
            return value
    return None


def _items(value: Any) -> list[Any]:
    return [] if value is None else list(value)


def _normalize_class_names(value: Any, source: str) -> dict[int, str]:
    if isinstance(value, list):
        names = {index: str(name).strip() for index, name in enumerate(value)}
    elif isinstance(value, dict):
        try:
            names = {int(key): str(name).strip() for key, name in value.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{source} class IDs must be integers.") from exc
    else:
        raise TypeError(f"{source} must define class names as a list or mapping.")
    if not names or sorted(names) != list(range(len(names))) or any(not name for name in names.values()):
        raise ValueError(f"{source} class names must be non-empty and use contiguous IDs starting at 0.")
    return names


def verify_model_matches_dataset(model_names: Any, data_path: str | Path) -> dict[int, str]:
    """Reject evaluation when checkpoint classes do not match the dataset labels."""
    data_yaml = Path(data_path)
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")
    try:
        dataset = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Could not parse dataset YAML: {exc}") from exc
    if not isinstance(dataset, dict):
        raise TypeError("Dataset YAML root must be a mapping.")

    dataset_names = _normalize_class_names(dataset.get("names"), "Dataset YAML")
    checkpoint_names = _normalize_class_names(model_names, "Model checkpoint")
    normalized_checkpoint_names = {key: value.casefold() for key, value in checkpoint_names.items()}
    normalized_dataset_names = {key: value.casefold() for key, value in dataset_names.items()}
    if normalized_checkpoint_names != normalized_dataset_names:
        raise ValueError(
            "Model class names do not match dataset.yaml. Evaluate with the fire/smoke-trained "
            "checkpoint produced by training/train.py, not generic COCO weights."
        )
    return dataset_names


def summarize_validation_metrics(
    metrics: Any,
    class_names: dict[int | str, str] | None = None,
) -> dict[str, Any]:
    """Normalize Ultralytics validation metrics without inventing missing values."""
    results = getattr(metrics, "results_dict", {}) or {}
    box = getattr(metrics, "box", None)
    precision = _metric_value(results, "metrics/precision(B)", "precision")
    recall = _metric_value(results, "metrics/recall(B)", "recall")
    map50 = _metric_value(results, "metrics/mAP50(B)", "mAP50")
    map50_95 = _metric_value(results, "metrics/mAP50-95(B)", "mAP50-95")

    if box is not None:
        precision = precision if precision is not None else _number(getattr(box, "mp", None))
        recall = recall if recall is not None else _number(getattr(box, "mr", None))
        map50 = map50 if map50 is not None else _number(getattr(box, "map50", None))
        map50_95 = map50_95 if map50_95 is not None else _number(getattr(box, "map", None))

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
    speed = {
        key: value
        for key in ("preprocess", "inference", "loss", "postprocess")
        if (value := _number((getattr(metrics, "speed", {}) or {}).get(key))) is not None
    }
    measured_stages = [
        speed[key] for key in ("preprocess", "inference", "postprocess") if key in speed
    ]
    fps = 1000 / sum(measured_stages) if measured_stages and sum(measured_stages) > 0 else None
    inference_fps = (
        1000 / speed["inference"]
        if speed.get("inference", 0) > 0
        else None
    )

    per_class: dict[str, dict[str, Any]] = {}
    if class_names:
        for raw_class_id, name in class_names.items():
            class_id = int(raw_class_id)
            per_class[str(class_id)] = {
                "name": name,
                "precision": None,
                "recall": None,
                "f1": None,
                "ap50": None,
                "ap50_95": None,
            }

    if box is not None:
        class_ids = _items(getattr(box, "ap_class_index", None))
        precisions = _items(getattr(box, "p", None))
        recalls = _items(getattr(box, "r", None))
        ap50_values = _items(getattr(box, "ap50", None))
        ap_values = _items(getattr(box, "ap", None))
        for position, raw_class_id in enumerate(class_ids):
            class_id = int(raw_class_id)
            name = None
            if class_names:
                name = class_names.get(class_id, class_names.get(str(class_id)))
            ap50 = _number(ap50_values[position]) if position < len(ap50_values) else None
            ap = ap_values[position] if position < len(ap_values) else None
            ap_mean = None
            if ap is not None:
                try:
                    ap_items = [float(value) for value in ap]
                    ap_mean = sum(ap_items) / len(ap_items) if ap_items else None
                except TypeError:
                    ap_mean = _number(ap)
            class_precision = (
                _number(precisions[position]) if position < len(precisions) else None
            )
            class_recall = (
                _number(recalls[position]) if position < len(recalls) else None
            )
            class_f1 = (
                2 * class_precision * class_recall / (class_precision + class_recall)
                if class_precision is not None
                and class_recall is not None
                and class_precision + class_recall > 0
                else None
            )
            per_class[str(class_id)] = {
                "name": name or f"class_{class_id}",
                "precision": class_precision,
                "recall": class_recall,
                "f1": class_f1,
                "ap50": ap50,
                "ap50_95": ap_mean,
            }

    if not per_class:
        per_class = results.get("per_class", {}) or {}

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mAP50": map50,
        "mAP50_95": map50_95,
        "fps": fps,
        "inference_fps": inference_fps,
        "speed_ms_per_image": speed,
        "per_class": per_class,
    }


def save_validation_metrics(payload: dict[str, Any], output_dir: Path) -> Path:
    """Write a run-local report and the latest report consumed by the dashboard."""
    output_dir.mkdir(parents=True, exist_ok=True)
    run_report = output_dir / "metrics.json"
    latest_report = Path(__file__).resolve().parents[1] / "reports" / "latest_evaluation.json"
    serialized = json.dumps(payload, indent=2, allow_nan=False)
    run_report.write_text(serialized, encoding="utf-8")
    if latest_report.resolve() != run_report.resolve():
        latest_report.parent.mkdir(parents=True, exist_ok=True)
        latest_report.write_text(serialized, encoding="utf-8")
    return run_report
