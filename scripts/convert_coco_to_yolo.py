from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List


def convert_coco_to_yolo(
    coco_json: Path,
    output_dir: Path,
    images_root: Path,
    copy_images: bool = False,
) -> Dict:
    """
    Convert COCO format annotations to YOLO format.

    Args:
        coco_json: Path to COCO JSON file
        output_dir: Output directory for YOLO labels (will create labels/ subdir)
        images_root: Root directory where images are located (for copying if needed)
        copy_images: Whether to copy images to output_dir/images/

    Returns:
        Dict with conversion stats
    """
    with coco_json.open() as f:
        coco = json.load(f)

    # Build category mapping: COCO category_id -> YOLO class_id (0-indexed)
    categories = {cat["id"]: idx for idx, cat in enumerate(sorted(coco["categories"], key=lambda x: x["id"]))}
    print(f"Category mapping: {categories}")

    # Build image_id -> image info mapping
    images = {img["id"]: img for img in coco["images"]}

    # Output directories
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    images_out_dir = output_dir / "images"
    if copy_images:
        images_out_dir.mkdir(parents=True, exist_ok=True)

    # Process annotations
    label_files_created = 0
    annotations_processed = 0
    missing_images = 0

    for ann in coco["annotations"]:
        image_id = ann["image_id"]
        if image_id not in images:
            continue

        img_info = images[image_id]
        img_w = img_info["width"]
        img_h = img_info["height"]
        img_filename = img_info["file_name"]

        # COCO bbox: [x, y, width, height] (top-left corner)
        # YOLO bbox: [x_center, y_center, width, height] normalized
        x, y, w, h = ann["bbox"]
        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        w_norm = w / img_w
        h_norm = h / img_h

        # Clamp to valid range
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        w_norm = max(0.0, min(1.0, w_norm))
        h_norm = max(0.0, min(1.0, h_norm))

        # Category mapping
        cat_id = ann["category_id"]
        if cat_id not in categories:
            continue
        class_id = categories[cat_id]

        # Write label file (one line per annotation)
        label_filename = Path(img_filename).with_suffix(".txt")
        label_path = labels_dir / label_filename

        with label_path.open("a") as f:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        annotations_processed += 1

        # Track label files created
        if label_path.stat().st_size > 0:
            label_files_created += 1

    # Copy images if requested
    images_copied = 0
    if copy_images and images_root.exists():
        for img_info in coco["images"]:
            src = images_root / img_info["file_name"]
            dst = images_out_dir / img_info["file_name"]
            if src.exists():
                shutil.copy2(src, dst)
                images_copied += 1
            else:
                missing_images += 1

    return {
        "images_in_json": len(coco["images"]),
        "annotations_processed": annotations_processed,
        "label_files_created": label_files_created,
        "images_copied": images_copied,
        "missing_images": missing_images,
        "categories": {v: k for k, v in categories.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="Convert FASDD COCO annotations to YOLO format")
    parser.add_argument("--annotations-root", type=Path, default=Path(r"C:\Users\dines\Downloads\FASDD_UAV\annotations"), help="Root of annotations folder")
    parser.add_argument("--images-root", type=Path, default=None, help="Root directory where images are located (optional)")
    parser.add_argument("--output", type=Path, default=Path("data/processed"), help="Output directory for YOLO format dataset")
    parser.add_argument("--copy-images", action="store_true", help="Copy images to output if images-root provided")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # Process each split
    splits = ["train", "val", "test"]
    all_stats = {}

    for split in splits:
        coco_json = args.annotations_root / "COCO_UAV" / "Annotations" / f"{split}.json"
        if not coco_json.exists():
            print(f"[WARN] {coco_json} not found, skipping")
            continue

        print(f"\n[*] Converting {split} split...")
        split_output = args.output / split
        stats = convert_coco_to_yolo(
            coco_json=coco_json,
            output_dir=split_output,
            images_root=args.images_root or Path("."),
            copy_images=args.copy_images,
        )
        all_stats[split] = stats
        print(f"  Images in JSON: {stats['images_in_json']}")
        print(f"  Annotations processed: {stats['annotations_processed']}")
        print(f"  Label files created: {stats['label_files_created']}")
        if args.copy_images:
            print(f"  Images copied: {stats['images_copied']}")
            print(f"  Missing images: {stats['missing_images']}")

    # Create dataset.yaml
    categories = {0: "fire", 1: "smoke"}
    dataset_yaml = {
        "path": str(args.output.resolve()),
        "train": "train/images" if args.copy_images else "train",
        "val": "val/images" if args.copy_images else "val",
        "test": "test/images" if args.copy_images else "test",
        "names": categories,
    }

    yaml_content = "\n".join([
        f"path: {dataset_yaml['path']}",
        f"train: {dataset_yaml['train']}",
        f"val: {dataset_yaml['val']}",
        f"test: {dataset_yaml['test']}",
        "names:",
        *[f"  {idx}: '{name}'" for idx, name in categories.items()],
    ]) + "\n"

    (args.output / "dataset.yaml").write_text(yaml_content, encoding="utf-8")
    print(f"\n[*] Created {args.output / 'dataset.yaml'}")
    print(yaml_content)

    # Save conversion report
    import datetime
    report = {
        "converted_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_annotations": str(args.annotations_root),
        "output_dir": str(args.output),
        "categories": categories,
        "splits": all_stats,
    }
    (args.output / "conversion_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[*] Saved conversion report to {args.output / 'conversion_report.json'}")


if __name__ == "__main__":
    main()