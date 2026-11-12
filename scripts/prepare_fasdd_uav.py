#!/usr/bin/env python3
"""
Prepare FASDD_UAV dataset for YOLO training.

This script converts the FASDD_UAV dataset (which comes with COCO, VOC, YOLO, and TDML annotations)
into a standard YOLO layout using the OFFICIAL train/val/test splits provided in the COCO JSON files.

Usage:
    python scripts/prepare_fasdd_uav.py \
        --source /path/to/FASDD_UAV \
        --output data/processed \
        --classes both \
        --symlink
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

# Class mapping for FASDD_UAV
FASDD_CLASSES_BOTH = {0: "fire", 1: "smoke"}
FASDD_CLASSES_FIRE_ONLY = {0: "fire"}
FASDD_CLASSES_SMOKE_ONLY = {0: "smoke"}


def validate_bbox(x: float, y: float, w: float, h: float, img_w: int, img_h: int) -> tuple[bool, str]:
    """Validate YOLO-format bbox (normalized center-x, center-y, width, height)."""
    if w <= 0 or h <= 0:
        return False, "width or height <= 0"
    if x < 0 or x > 1 or y < 0 or y > 1:
        return False, "center coordinates out of [0,1] range"
    if w > 1 or h > 1:
        return False, "width or height > 1"
    # Check if bbox extends beyond image
    x_min = x - w / 2
    y_min = y - h / 2
    x_max = x + w / 2
    y_max = y + h / 2
    if x_min < 0 or y_min < 0 or x_max > 1 or y_max > 1:
        return False, "bbox extends beyond image boundaries"
    return True, ""


def convert_coco_to_yolo_bbox(bbox: list[float], img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """Convert COCO bbox [x, y, w, h] (top-left) to YOLO [x_c, y_c, w, h] (normalized center)."""
    x, y, w, h = bbox
    x_c = (x + w / 2) / img_w
    y_c = (y + h / 2) / img_h
    w_n = w / img_w
    h_n = h / img_h
    return x_c, y_c, w_n, h_n


def process_coco_split(
    coco_json: Path,
    images_root: Path,
    output_dir: Path,
    class_map: dict[int, int],
    use_symlink: bool,
    split_name: str,
    split_stats: dict[str, Any],
    max_images: int | None = None,
    seed: int = 42,
) -> None:
    """Process a single COCO split (train/val/test)."""
    with coco_json.open() as f:
        coco = json.load(f)

    # Build category mapping: COCO category_id -> YOLO class_id
    coco_cats = {cat["id"]: cat["name"] for cat in coco["categories"]}
    print(f"  COCO categories: {coco_cats}")

    # Build image_id -> image info mapping
    images = {img["id"]: img for img in coco["images"]}

    # Output directories
    labels_dir = output_dir / split_name / "labels"
    images_dir = output_dir / split_name / "images"
    labels_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # Process annotations
    anns_by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in images:
            split_stats["orphan_annotations"] += 1
            continue
        cat_id = ann["category_id"]
        if cat_id not in class_map:
            split_stats["skipped_annotations"] += 1
            continue
        yolo_cls = class_map[cat_id]
        x_c, y_c, w_n, h_n = convert_coco_to_yolo_bbox(ann["bbox"], images[img_id]["width"], images[img_id]["height"])
        valid, err = validate_bbox(x_c, y_c, w_n, h_n, images[img_id]["width"], images[img_id]["height"])
        if not valid:
            split_stats["invalid_boxes"] += 1
            continue
        anns_by_image.setdefault(img_id, []).append({
            "class": yolo_cls,
            "x": x_c, "y": y_c, "w": w_n, "h": h_n,
        })

    if max_images is not None and len(images) > max_images:
        class_ids = sorted(set(class_map.values()))
        if max_images < len(class_ids):
            raise ValueError(f"--max-images-per-split must be at least {len(class_ids)}.")

        image_classes = {
            image_id: {ann["class"] for ann in annotations}
            for image_id, annotations in anns_by_image.items()
        }
        selected_ids: set[int] = set()
        selected_class_counts = {class_id: 0 for class_id in class_ids}
        target_per_class = max(1, max_images // len(class_ids))
        rng = random.Random(seed)
        for class_id in class_ids:
            candidates = [
                image_id for image_id, classes in image_classes.items()
                if class_id in classes
            ]
            rng.shuffle(candidates)
            for image_id in candidates:
                if selected_class_counts[class_id] >= target_per_class:
                    break
                if image_id not in selected_ids:
                    selected_ids.add(image_id)
                    for selected_class in image_classes[image_id]:
                        selected_class_counts[selected_class] += 1

        remaining_ids = [image_id for image_id in images if image_id not in selected_ids]
        rng.shuffle(remaining_ids)
        selected_ids.update(remaining_ids[: max_images - len(selected_ids)])
        images = {image_id: images[image_id] for image_id in sorted(selected_ids)}

    # Write label files and link/copy images
    for img_id, img_info in images.items():
        img_filename = img_info["file_name"]
        src_img = images_root / img_filename
        dst_img = images_dir / img_filename
        label_filename = Path(img_filename).with_suffix(".txt")
        label_path = labels_dir / label_filename

        # Link or copy image
        if use_symlink:
            try:
                if dst_img.exists() or dst_img.is_symlink():
                    dst_img.unlink()
                os.symlink(src_img.resolve(), dst_img)
                split_stats["images_linked"] += 1
            except (OSError, FileNotFoundError):
                try:
                    os.link(src_img, dst_img)
                    split_stats["images_hardlinked"] += 1
                except (OSError, FileNotFoundError):
                    try:
                        shutil.copy2(src_img, dst_img)
                        split_stats["images_copied"] += 1
                        split_stats["images_fallback_copy"] = split_stats.get("images_fallback_copy", 0) + 1
                    except (OSError, FileNotFoundError) as e2:
                        split_stats["images_missing"] += 1
                        print(f"    [WARN] Could not copy {src_img}: {e2}", file=sys.stderr)
        else:
            try:
                shutil.copy2(src_img, dst_img)
                split_stats["images_copied"] += 1
            except (OSError, FileNotFoundError) as e:
                split_stats["images_missing"] += 1
                print(f"    [WARN] Could not copy {src_img}: {e}", file=sys.stderr)

        # Write label file (empty if no annotations = negative sample)
        anns = anns_by_image.get(img_id, [])
        split_stats["annotations_processed"] += len(anns)
        with label_path.open("w") as f:
            for ann in anns:
                f.write(f"{ann['class']} {ann['x']:.6f} {ann['y']:.6f} {ann['w']:.6f} {ann['h']:.6f}\n")
        split_stats["label_files_written"] += 1
        split_stats["images_with_annotations"] += 1 if anns else 0

    split_stats["images_in_split"] += len(images)
    print(f"  {split_name}: {len(images)} images, {split_stats['annotations_processed']} annotations processed, "
          f"{split_stats['images_with_annotations']} with annotations")


def write_yaml_config(output_dir: Path, class_names: dict[int, str], use_symlink: bool) -> None:
    """Write dataset.yaml with relative paths."""
    # Use relative paths for portability
    yaml_content = {
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": class_names,
    }
    if use_symlink:
        # Add note about symlinks
        yaml_content["# Note"] = "Images are linked from original dataset"

    with (output_dir / "dataset.yaml").open("w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
    print(f"  Wrote dataset.yaml to {output_dir}")


def write_dataset_report(output_dir: Path, split_stats: dict[str, Any], class_names: dict[int, str],
                         total_images: int, total_annotations: int, total_images_with_ann: int,
                         total_negative_samples: int, max_images_per_split: int | None, seed: int) -> None:
    """Write dataset_report.json with comprehensive statistics."""
    report = {
        "dataset": "FASDD_UAV",
        "source_format": "COCO",
        "classes": class_names,
        "sampling": {
            "max_images_per_split": max_images_per_split,
            "seed": seed,
            "description": "Deterministic class-balanced pilot sample" if max_images_per_split else "Full official split",
        },
        "splits": {
            "train": {
                "images": split_stats.get("train", {}).get("images_in_split", 0),
                "annotations": split_stats.get("train", {}).get("annotations_processed", 0),
                "images_with_annotations": split_stats.get("train", {}).get("images_with_annotations", 0),
            },
            "val": {
                "images": split_stats.get("val", {}).get("images_in_split", 0),
                "annotations": split_stats.get("val", {}).get("annotations_processed", 0),
                "images_with_annotations": split_stats.get("val", {}).get("images_with_annotations", 0),
            },
            "test": {
                "images": split_stats.get("test", {}).get("images_in_split", 0),
                "annotations": split_stats.get("test", {}).get("annotations_processed", 0),
                "images_with_annotations": split_stats.get("test", {}).get("images_with_annotations", 0),
            },
        },
        "total_images": total_images,
        "total_annotations": sum(split_stats.get(s, {}).get("annotations_processed", 0) for s in ["train", "val", "test"]),
        "total_images_with_annotations": sum(split_stats.get(s, {}).get("images_with_annotations", 0) for s in ["train", "val", "test"]),
        "negative_samples": total_negative_samples,
        "images_linked": sum(split_stats.get(s, {}).get("images_linked", 0) for s in ["train", "val", "test"]),
        "images_hardlinked": sum(split_stats.get(s, {}).get("images_hardlinked", 0) for s in ["train", "val", "test"]),
        "images_copied": sum(split_stats.get(s, {}).get("images_copied", 0) for s in ["train", "val", "test"]),
        "images_fallback_copy": sum(split_stats.get(s, {}).get("images_fallback_copy", 0) for s in ["train", "val", "test"]),
        "images_missing": sum(split_stats.get(s, {}).get("images_missing", 0) for s in ["train", "val", "test"]),
        "invalid_boxes": sum(split_stats.get(s, {}).get("invalid_boxes", 0) for s in ["train", "val", "test"]),
        "skipped_annotations": sum(split_stats.get(s, {}).get("skipped_annotations", 0) for s in ["train", "val", "test"]),
        "orphan_annotations": sum(split_stats.get(s, {}).get("orphan_annotations", 0) for s in ["train", "val", "test"]),
    }
    (output_dir / "dataset_report.json").write_text(json.dumps(report, indent=2))
    print(f"  Wrote dataset_report.json to {output_dir}")


def print_summary_table(split_stats: dict[str, Any], class_names: dict[int, str], use_symlink: bool) -> None:
    """Print a summary table to console."""
    print("\n" + "=" * 70)
    print("FASDD_UAV DATASET PREPARATION SUMMARY")
    print("=" * 70)
    print(f"Classes: {class_names}")
    print(f"Mode: {'symlink' if use_symlink else 'copy'}")
    print()
    print(f"{'Split':<10} {'Images':>10} {'Annotations':>12} {'With Annot.':>12} {'Negatives':>10}")
    print("-" * 70)
    for split in ["train", "val", "test"]:
        s = split_stats.get(split, {})
        imgs = s.get("images_in_split", 0)
        anns = s.get("annotations_processed", 0)
        with_ann = s.get("images_with_annotations", 0)
        neg = imgs - with_ann
        print(f"{split:<10} {imgs:>10} {anns:>12} {with_ann:>12} {neg:>10}")
    print("-" * 70)
    total_imgs = sum(s.get("images_in_split", 0) for s in split_stats.values())
    total_ann = sum(s.get("annotations_processed", 0) for s in split_stats.values())
    total_with_ann = sum(s.get("images_with_annotations", 0) for s in split_stats.values())
    total_neg = total_imgs - total_with_ann
    print(f"{'TOTAL':<10} {total_imgs:>10} {total_ann:>12} {total_with_ann:>12} {total_neg:>10}")
    print()
    total_linked = sum(s.get("images_linked", 0) for s in split_stats.values())
    total_hardlinked = sum(s.get("images_hardlinked", 0) for s in split_stats.values())
    total_copied = sum(s.get("images_copied", 0) for s in split_stats.values())
    total_fallback = sum(s.get("images_fallback_copy", 0) for s in split_stats.values())
    total_missing = sum(s.get("images_missing", 0) for s in split_stats.values())
    print(f"Images linked: {total_linked}")
    print(f"Images hardlinked: {total_hardlinked}")
    print(f"Images copied: {total_copied}")
    print(f"Images fallback copy: {total_fallback}")
    print(f"Images missing: {total_missing}")
    print(f"Invalid boxes: {sum(s.get('invalid_boxes', 0) for s in split_stats.values())}")
    print(f"Skipped annotations (class filter): {sum(s.get('skipped_annotations', 0) for s in split_stats.values())}")
    print(f"Orphan annotations: {sum(s.get('orphan_annotations', 0) for s in split_stats.values())}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare FASDD_UAV dataset for YOLO training")
    parser.add_argument("--source", type=Path, required=True, help="FASDD_UAV root directory (contains images/ and annotations/)")
    parser.add_argument("--annotations-format", choices=["yolo", "coco", "voc", "tdml"], default="coco",
                        help="Annotation format to use (default: coco - uses official splits)")
    parser.add_argument("--output", type=Path, default=Path("data/processed"), help="Output directory")
    parser.add_argument("--classes", choices=["both", "fire", "smoke"], default="both",
                        help="Which classes to include (default: both)")
    parser.add_argument("--max-images-per-split", type=int, default=None,
                        help="Optional deterministic, class-balanced image cap per split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for limited splits")
    parser.add_argument("--symlink", action="store_true", default=True, help="Use symlinks instead of copying images")
    parser.add_argument("--copy", action="store_false", dest="symlink", help="Copy images instead of symlinking")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output directory")
    args = parser.parse_args()

    if args.max_images_per_split is not None and args.max_images_per_split < 1:
        parser.error("--max-images-per-split must be a positive integer.")

    # Validate source
    source = args.source.expanduser().resolve()
    if not source.exists():
        sys.exit(f"ERROR: Source directory not found: {source}")
    if not (source / "images").exists():
        sys.exit(f"ERROR: images/ directory not found in {source}")
    if not (source / "annotations").exists():
        sys.exit(f"ERROR: annotations/ directory not found in {source}")

    # Determine class mapping
    if args.classes == "both":
        class_map = {0: 0, 1: 1}  # COCO fire=0->YOLO 0, COCO smoke=1->YOLO 1
        class_names = FASDD_CLASSES_BOTH
    elif args.classes == "fire":
        class_map = {0: 0}  # Only fire
        class_names = FASDD_CLASSES_FIRE_ONLY
    else:  # smoke
        class_map = {1: 0}  # COCO smoke=1->YOLO 0
        class_names = FASDD_CLASSES_SMOKE_ONLY

    print(f"[INFO] Source: {source}")
    print(f"[INFO] Output: {args.output}")
    print(f"[INFO] Classes: {args.classes} -> {class_names}")
    print(f"[INFO] Mode: {'symlink' if args.symlink else 'copy'}")

    # Handle output directory
    output_dir = args.output.resolve()
    if output_dir.exists():
        if args.force:
            print(f"[WARN] Output directory exists, removing: {output_dir}")
            shutil.rmtree(output_dir)
        else:
            sys.exit(f"ERROR: Output directory exists. Use --force to overwrite: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Statistics tracking per split
    split_stats = {s: {
        "images_in_split": 0,
        "annotations_processed": 0,
        "images_with_annotations": 0,
        "images_linked": 0,
        "images_hardlinked": 0,
        "images_copied": 0,
        "images_fallback_copy": 0,
        "images_missing": 0,
        "invalid_boxes": 0,
        "skipped_annotations": 0,
        "orphan_annotations": 0,
        "label_files_written": 0,
    } for s in ["train", "val", "test"]}

    # Process splits using COCO annotations (official splits)
    annotations_root = source / "annotations" / "COCO_UAV" / "Annotations"
    if not annotations_root.exists():
        sys.exit(f"ERROR: COCO annotations not found at {annotations_root}")

    images_root = source / "images"
    if not images_root.exists():
        sys.exit(f"ERROR: Images directory not found at {images_root}")

    # Process each split
    for split_name in ["train", "val", "test"]:
        coco_json = annotations_root / f"{split_name}.json"
        if not coco_json.exists():
            print(f"[WARN] {split_name}.json not found, skipping", file=sys.stderr)
            continue

        print(f"[INFO] Processing {split_name} split...")
        process_coco_split(
            coco_json=coco_json,
            images_root=images_root,
            output_dir=output_dir,
            class_map=class_map,
            use_symlink=args.symlink,
            split_name=split_name,
            split_stats=split_stats[split_name],
            max_images=args.max_images_per_split,
            seed=args.seed + ["train", "val", "test"].index(split_name),
        )

    # Aggregate totals
    total_images = sum(split_stats[s]["images_in_split"] for s in ["train", "val", "test"])
    total_annotations = sum(split_stats[s]["annotations_processed"] for s in ["train", "val", "test"])
    total_images_with_ann = sum(split_stats[s]["images_with_annotations"] for s in ["train", "val", "test"])
    total_negative_samples = total_images - total_images_with_ann

    # Write outputs
    write_yaml_config(output_dir, class_names, args.symlink)
    write_dataset_report(output_dir, split_stats, class_names, total_images,
                         sum(split_stats[s]["annotations_processed"] for s in ["train", "val", "test"]),
                         total_images_with_ann, total_negative_samples, args.max_images_per_split, args.seed)
    print_summary_table(split_stats, class_names, args.symlink)
    print(f"\n[INFO] Dataset prepared at: {output_dir}")


if __name__ == "__main__":
    main()