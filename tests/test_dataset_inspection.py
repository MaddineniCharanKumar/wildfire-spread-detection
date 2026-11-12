from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from scripts.inspect_dataset import inspect_dataset


def test_inspection_counts_coco_objects_and_matches_voc_by_filename(tmp_path: Path) -> None:
    image_path = tmp_path / "images" / "forest.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (100, 80)).save(image_path)

    annotations = tmp_path / "annotations"
    annotations.mkdir()
    (annotations / "forest.xml").write_text(
        "<annotation><size><width>100</width><height>80</height></size>"
        "<object><name>fire</name><bndbox><xmin>10</xmin><ymin>12</ymin>"
        "<xmax>40</xmax><ymax>50</ymax></bndbox></object></annotation>",
        encoding="utf-8",
    )
    (annotations / "train.json").write_text(
        json.dumps({
            "images": [{"id": 1, "file_name": "images/forest.jpg", "width": 100, "height": 80}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 2, "bbox": [10, 12, 30, 38]}
            ],
            "categories": [{"id": 2, "name": "fire"}],
        }),
        encoding="utf-8",
    )

    report = inspect_dataset(tmp_path)

    assert report["counts"]["images"] == 1
    assert report["counts"]["annotations"] == 1
    assert report["counts"]["classes_detected"] == 1
    assert report["annotations_by_format"] == {"YOLO": 0, "VOC/XML": 1, "COCO/JSON": 1}
    assert report["classes"] == {"fire": 1}
    assert report["orphan_labels"] == []
    assert report["missing_coco_images"] == []


def test_inspection_counts_voc_objects_without_an_image(tmp_path: Path) -> None:
    annotations = tmp_path / "Annotations"
    annotations.mkdir()
    (annotations / "missing.xml").write_text(
        "<annotation><size><width>20</width><height>10</height></size>"
        "<object><name>smoke</name><bndbox><xmin>1</xmin><ymin>1</ymin>"
        "<xmax>8</xmax><ymax>7</ymax></bndbox></object>"
        "<object><name>fire</name><bndbox><xmin>9</xmin><ymin>2</ymin>"
        "<xmax>18</xmax><ymax>9</ymax></bndbox></object></annotation>",
        encoding="utf-8",
    )

    report = inspect_dataset(tmp_path)

    assert report["counts"]["images"] == 0
    assert report["counts"]["annotations"] == 2
    assert report["classes"] == {"fire": 1, "smoke": 1}
    assert report["orphan_labels"] == [str(Path("Annotations") / "missing.xml")]
