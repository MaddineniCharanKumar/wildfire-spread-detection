from __future__ import annotations

"""
Hard Negatives Preparation Script for FASDD Fire/Smoke Detection

This script creates a scaffold for adding hard negative samples (images without fire/smoke
that look similar to positive samples) to reduce false positives.

Hard negative categories for wildfire detection:
1. Fog/mist - looks like smoke
2. Sunrise/sunset glow - looks like fire
3. Brake lights / vehicle lights at night - looks like fire
4. Chimney steam / industrial steam - looks like smoke
5. Dust clouds - looks like smoke
6. Cloud shadows - can look like smoke
7. Water reflections - can look like fire glow
8. Lens flare - can look like fire

Usage:
    python scripts/prepare_hard_negatives.py --output data/hard_negatives --count-per-class 50

This creates empty label files (no annotations) for each hard negative image.
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import List, Tuple

HARD_NEGATIVE_CATEGORIES = [
    ("fog", "Fog/mist that resembles smoke"),
    ("sunrise_sunset", "Sunrise/sunset glow resembling fire"),
    ("brake_lights", "Vehicle brake lights at night resembling fire"),
    ("chimney_steam", "Industrial/chimney steam resembling smoke"),
    ("dust", "Dust clouds resembling smoke"),
    ("cloud_shadows", "Cloud shadows resembling smoke"),
    ("water_reflection", "Water reflections resembling fire glow"),
    ("lens_flare", "Camera lens flare resembling fire"),
]


def create_hard_negative_structure(output_dir: Path) -> None:
    """Create directory structure for hard negatives."""
    for split in ["train", "val", "test"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # Create a README with instructions
    readme = output_dir / "README.md"
    readme.write_text("""# Hard Negatives for Fire/Smoke Detection

## Purpose
Hard negatives are images that look similar to fire/smoke but contain neither.
Adding them with empty label files teaches the model to NOT detect fire/smoke in these cases.

## Categories
""" + "\n".join([f"- **{cat}**: {desc}" for cat, desc in HARD_NEGATIVE_CATEGORIES]) + """

## Structure
```
hard_negatives/
├── train/
│   ├── images/     # Place hard negative images here
│   └── labels/     # Empty .txt files (one per image)
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## How to Add Hard Negatives

1. **Source images** from:
   - Public datasets: Foggy driving datasets (e.g., Foggy Zurich, Foggy Cityscapes)
   - Sunrise/sunset: Landscape photography datasets
   - Night driving: BDD100K, nuScenes (for brake lights)
   - Industrial: Custom collection or steam/smoke datasets
   - Dust: Construction/mining site footage
   - Your own UAV footage without fire/smoke

2. **Place images** in the appropriate `split/images/` folder

3. **Create empty label files** (run this script with `--create-labels` or manually):
   ```bash
   # For each image.jpg, create an empty image.txt in labels/
   touch labels/image.txt
   ```

3. **Recommended counts** (per category, per split):
   - Train: 50-100 per category
   - Val: 10-20 per category
   - Test: 10-20 per category

4. **Update dataset.yaml** to include hard negatives in training:
   ```yaml
   train: ../hard_negatives/train/images
   val: ../hard_negatives/val/images
   ```

## Label Format
Empty `.txt` files = no objects (background class). The model learns these are negative examples.

## Evaluation
After training with hard negatives:
1. Run evaluation on hard negative test set
2. Check false positive rate per category
3. Report in model card
""", encoding="utf-8")


def create_empty_labels(images_dir: Path, labels_dir: Path) -> int:
    """Create empty label files for all images in images_dir."""
    labels_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for img in images_dir.glob("*.jpg"):
        label_path = labels_dir / img.with_suffix(".txt").name
        if not label_path.exists():
            label_path.write_text("")
            count += 1
    for img in images_dir.glob("*.png"):
        label_path = labels_dir / img.with_suffix(".txt").name
        if not label_path.exists():
            label_path.write_text("")
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Prepare hard negatives structure for fire/smoke detection")
    parser.add_argument("--output", type=Path, default=Path("data/hard_negatives"), help="Output directory")
    parser.add_argument("--create-labels", action="store_true", help="Create empty labels for existing images")
    parser.add_argument("--count-per-class", type=int, default=50, help="Target count per category (for reference)")
    args = parser.parse_args()

    print("[*] Creating hard negatives structure...")
    create_hard_negative_structure(args.output)

    if args.create_labels:
        print("[*] Creating empty labels for existing images...")
        for split in ["train", "val", "test"]:
            images_dir = args.output / split / "images"
            labels_dir = args.output / split / "labels"
            if images_dir.exists():
                count = create_empty_labels(images_dir, labels_dir)
                print(f"  {split}: created {count} empty label files")

    print(f"\n[*] Done! Structure created at {args.output}")
    print("[*] Next steps:")
    print("  1. Add hard negative images to each split/images/ folder")
    print("  2. Run with --create-labels to generate empty label files")
    print("  3. Update training/dataset.yaml to include hard negatives")
    print("  4. Retrain model")


if __name__ == "__main__":
    main()