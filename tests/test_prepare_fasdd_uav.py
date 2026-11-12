from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.prepare_fasdd_uav import process_coco_split, write_yaml_config


def test_limited_split_is_class_balanced_and_reports_sample_counts(tmp_path: Path) -> None:
    images_root = tmp_path / "source"
    images_root.mkdir()
    images = [
        {"id": image_id, "file_name": f"{image_id}.jpg", "width": 10, "height": 10}
        for image_id in range(1, 5)
    ]
    for image in images:
        (images_root / image["file_name"]).write_bytes(b"image")

    annotations = [
        {"id": 1, "image_id": 1, "category_id": 0, "bbox": [1, 1, 2, 2]},
        {"id": 2, "image_id": 2, "category_id": 1, "bbox": [1, 1, 2, 2]},
        {"id": 3, "image_id": 3, "category_id": 0, "bbox": [1, 1, 2, 2]},
        {"id": 4, "image_id": 3, "category_id": 1, "bbox": [4, 4, 2, 2]},
    ]
    coco_path = tmp_path / "annotations.json"
    coco_path.write_text(json.dumps({
        "categories": [{"id": 0, "name": "fire"}, {"id": 1, "name": "smoke"}],
        "images": images,
        "annotations": annotations,
    }), encoding="utf-8")
    stats = {
        "orphan_annotations": 0,
        "skipped_annotations": 0,
        "invalid_boxes": 0,
        "annotations_processed": 0,
        "images_in_split": 0,
        "images_with_annotations": 0,
        "images_copied": 0,
        "images_missing": 0,
        "label_files_written": 0,
    }

    process_coco_split(
        coco_json=coco_path,
        images_root=images_root,
        output_dir=tmp_path / "prepared",
        class_map={0: 0, 1: 1},
        use_symlink=False,
        split_name="train",
        split_stats=stats,
        max_images=2,
        seed=42,
    )

    labels = list((tmp_path / "prepared" / "train" / "labels").glob("*.txt"))
    class_ids = {
        int(line.split()[0])
        for label in labels
        for line in label.read_text(encoding="utf-8").splitlines()
    }
    assert len(labels) == 2
    assert class_ids == {0, 1}
    assert stats["images_in_split"] == 2
    assert stats["annotations_processed"] == sum(
        len(label.read_text(encoding="utf-8").splitlines()) for label in labels
    )


def test_dataset_yaml_points_to_absolute_prepared_root(tmp_path: Path) -> None:
    write_yaml_config(tmp_path, {0: "fire", 1: "smoke"}, use_symlink=False)

    config = yaml.safe_load((tmp_path / "dataset.yaml").read_text(encoding="utf-8"))

    assert Path(config["path"]).resolve() == tmp_path.resolve()
