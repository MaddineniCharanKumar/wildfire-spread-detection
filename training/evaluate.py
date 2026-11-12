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
    parser = argparse.ArgumentParser(description="Evaluate a YOLO model on a dataset and output actual metrics only.")
    parser.add_argument("--weights", type=str, default=None, help="Model weights (default: family+size from config)")
    parser.add_argument("--family", type=str, default=None, help="Model family (default: FIREGUARD_MODEL_FAMILY)")
    parser.add_argument("--size", type=str, default=None, help="Model size (default: FIREGUARD_MODEL_SIZE)")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0" if __import__("utils.device", fromlist=["detect_device"]).detect_device() == "cuda" else "cpu")
    parser.add_argument("--output", type=Path, default=Path("reports/eval"), help="Output directory for metrics and model card")
    parser.add_argument("--hard-negatives-dir", type=Path, default=None, help="Directory with hard negative test images for FP breakdown")
    parser.add_argument("--label", type=str, default=None, help="Optional note describing a subset or pilot evaluation")
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

    print(f"[INFO] Evaluating with model: {weights_path} on {args.split} split")
    metrics = model.val(data=args.data, split=args.split, imgsz=args.imgsz, device=args.device)
    payload = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "weights": weights_path,
        "family": family,
        "size": size,
        "data": args.data,
        "split": args.split,
        "evaluation_label": args.label,
        "imgsz": args.imgsz,
        "device": args.device,
        **summarize_validation_metrics(metrics, class_names),
    }

    report_path = save_validation_metrics(payload, args.output)
    print(f"[INFO] Metrics saved to {report_path}")

    # Hard negative false positive breakdown (if provided)
    fp_breakdown = {}
    if args.hard_negatives_dir and args.hard_negatives_dir.exists():
        print("[INFO] Running hard negative false positive analysis...")
        fp_breakdown = run_hard_negative_fp_analysis(model, args.hard_negatives_dir, args.device, args.imgsz)
        payload["hard_negative_fp_breakdown"] = fp_breakdown
        (args.output / "hard_negative_fp.json").write_text(json.dumps(fp_breakdown, indent=2))
        print(f"[INFO] FP breakdown saved to {args.output / 'hard_negative_fp.json'}")

    # Generate model card
    generate_model_card(payload, args.output / "model_card.md", weights_path, family, size)

    print(json.dumps(payload, indent=2))
    return 0


def run_hard_negative_fp_analysis(model, hard_negatives_dir: Path, device: str, imgsz: int) -> dict:
    """
    Run inference on hard negative images (which should have NO fire/smoke).
    Any detection is a false positive. Break down by hard negative category folder.
    """
    fp_breakdown = {}
    total_images = 0
    total_fps = 0

    for split in ["train", "val", "test"]:
        split_dir = hard_negatives_dir / split / "images"
        if not split_dir.exists():
            continue

        for category_dir in split_dir.iterdir():
            if not category_dir.is_dir():
                continue

            category = category_dir.name
            images = list(category_dir.glob("*.jpg")) + list(category_dir.glob("*.png"))
            if not images:
                continue

            cat_fps = 0
            cat_images = 0
            detections_by_class = {0: 0, 1: 0}  # fire, smoke

            for img_path in images:
                cat_images += 1
                total_images += 1
                results = model(str(img_path), imgsz=imgsz, device=device, verbose=False)
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            cls = int(box.cls[0].item())
                            if cls in detections_by_class:
                                detections_by_class[cls] += 1
                                cat_fps += 1
                                total_fps += 1

            if cat_images > 0:
                fp_breakdown[f"{split}/{category}"] = {
                    "images_tested": cat_images,
                    "false_positives": cat_fps,
                    "fp_rate": cat_fps / cat_images if cat_images > 0 else 0,
                    "false_positives_by_class": {
                        "fire": detections_by_class.get(0, 0),
                        "smoke": detections_by_class.get(1, 0),
                    }
                }

    fp_breakdown["summary"] = {
        "total_images_tested": total_images,
        "total_false_positives": total_fps,
        "overall_fp_rate": total_fps / total_images if total_images > 0 else 0,
    }

    return fp_breakdown


def generate_model_card(payload: dict, output_path: Path, weights_path: str, family: str, size: str) -> None:
    """Generate a model card markdown from evaluation results."""
    from datetime import datetime, timezone

    pc = payload.get("per_class", {})
    md = f"""# Model Card: {family}{size} Fire/Smoke Detector

**Generated**: {datetime.now(timezone.utc).isoformat()}
**Weights**: `{weights_path}`
**Family**: {family}
**Size**: {size}
**Data**: {payload.get('data')}
**Split**: {payload.get('split')}
**Evaluation note**: {payload.get('evaluation_label') or 'Full configured split'}
**Image Size**: {payload.get('imgsz')}
**Device**: {payload.get('device')}

## Overall Metrics

| Metric | Value |
|--------|-------|
| Precision | {_format_metric(payload.get('precision'))} |
| Recall | {_format_metric(payload.get('recall'))} |
| F1 | {_format_metric(payload.get('f1'))} |
| mAP@50 | {_format_metric(payload.get('mAP50'))} |
| mAP@50-95 | {_format_metric(payload.get('mAP50_95'))} |
| FPS (evaluation pipeline) | {_format_metric(payload.get('fps'))} |
| FPS (model inference only) | {_format_metric(payload.get('inference_fps'))} |

## Per-Class Metrics

| Class | Precision | Recall | F1 | AP@50 | AP@50-95 |
|-------|-----------|--------|----|-------|----------|
"""

    for idx in sorted(pc.keys(), key=int):
        cls_name = pc[idx].get("name") or f"class_{idx}"
        m = pc[idx]
        md += (
            f"| {cls_name} | {_format_metric(m.get('precision'))} | "
            f"{_format_metric(m.get('recall'))} | {_format_metric(m.get('f1'))} | "
            f"{_format_metric(m.get('ap50'))} | "
            f"{_format_metric(m.get('ap50_95'))} |\n"
        )

    # Hard negative breakdown
    fp = payload.get("hard_negative_fp_breakdown", {})
    if fp:
        md += "\n## Hard Negative False Positive Analysis\n\n"
        md += "| Category | Images Tested | False Positives | FP Rate | FP (fire) | FP (smoke) |\n"
        md += "|----------|---------------|-----------------|---------|-----------|------------|\n"

        for key, val in fp.items():
            if key == "summary":
                continue
            md += f"| {key} | {val.get('images_tested', 0)} | {val.get('false_positives', 0)} | {val.get('fp_rate', 0):.4f} | {val.get('false_positives_by_class', {}).get('fire', 0)} | {val.get('false_positives_by_class', {}).get('smoke', 0)} |\n"

        summary = fp.get("summary", {})
        md += f"\n**Overall**: {summary.get('total_false_positives', 0)} FPs / {summary.get('total_images_tested', 0)} images = {summary.get('overall_fp_rate', 0):.4f} FP rate\n"

    md += """

## Limitations

- Bounding-box area is an **area proxy**, not physical fire area.
- Centroid and box motion are estimated visual spread direction, not true propagation speed.
- Risk score is an environmental fusion estimate, not future-fire certainty.
- Metrics computed on FASDD UAV dataset; domain shift may affect real-world performance.
- Hard negative analysis only covers categories present in test set.

## License

**Ultralytics YOLO and weights are AGPL-3.0**. See https://github.com/ultralytics/ultralytics/blob/main/LICENSE

## Intended Use

Research prototype for UAV-based wildfire detection. Not certified for operational firefighting decisions.
"""

    output_path.write_text(md, encoding="utf-8")
    print(f"[INFO] Model card saved to {output_path}")


def _format_metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())