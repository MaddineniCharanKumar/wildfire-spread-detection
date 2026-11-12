from __future__ import annotations

import json
from pathlib import Path


def discover_annotations(dataset_root: str | Path) -> dict:
    dataset_root = Path(dataset_root)
    candidates = []
    for path in dataset_root.rglob('*'):
        if path.is_file() and path.suffix.lower() in {'.txt', '.json', '.xml'}:
            candidates.append(str(path.relative_to(dataset_root)))

    formats = []
    if any(p.endswith('.txt') for p in candidates):
        formats.append('YOLO')
    if any(p.endswith('.xml') for p in candidates):
        formats.append('VOC/XML')
    if any(p.endswith('.json') for p in candidates):
        formats.append('COCO/JSON')

    return {
        "formats": formats,
        "annotation_files": candidates,
    }


def build_manifest(dataset_root: str | Path, output_path: str | Path | None = None) -> dict:
    dataset_root = Path(dataset_root)
    manifest = []
    for split_name in ["train", "val", "test"]:
        split_dir = dataset_root / split_name
        if split_dir.exists():
            for image_path in sorted(split_dir.rglob('*')):
                if image_path.is_file() and image_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}:
                    label_path = image_path.with_suffix('.txt')
                    manifest.append({
                        "split": split_name,
                        "image": str(image_path.relative_to(dataset_root)),
                        "label": str(label_path.relative_to(dataset_root)) if label_path.exists() else None,
                    })
    output = Path(output_path) if output_path else dataset_root / "manifest.json"
    output.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return {"manifest_path": str(output), "count": len(manifest)}
