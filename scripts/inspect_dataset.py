from __future__ import annotations

import argparse
import hashlib
import html
import json
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LABEL_EXTENSIONS = {".txt", ".xml", ".json"}


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def is_label(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in LABEL_EXTENSIONS


def detect_annotation_format(paths: list[Path]) -> list[str]:
    suffixes = {path.suffix.lower() for path in paths}
    formats: list[str] = []
    if ".txt" in suffixes:
        formats.append("YOLO")
    if ".xml" in suffixes:
        formats.append("VOC/XML")
    if ".json" in suffixes:
        formats.append("COCO/JSON")
    return formats or ["UNKNOWN"]


def parse_yolo_label(path: Path) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
    ):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"line {line_number}: expected 5 YOLO fields")
        try:
            class_id = int(parts[0])
            values = [float(value) for value in parts[1:]]
        except ValueError as exc:
            raise ValueError(f"line {line_number}: non-numeric YOLO value") from exc
        boxes.append(
            {
                "class_id": class_id,
                "x_center": values[0],
                "y_center": values[1],
                "width": values[2],
                "height": values[3],
            }
        )
    return boxes


def box_is_valid(box: dict[str, Any]) -> bool:
    x, y, width, height = (
        float(box[key]) for key in ("x_center", "y_center", "width", "height")
    )
    return (
        width > 0
        and height > 0
        and 0 <= x <= 1
        and 0 <= y <= 1
        and width <= 1
        and height <= 1
        and 0 <= x - width / 2
        and x + width / 2 <= 1
        and 0 <= y - height / 2
        and y + height / 2 <= 1
    )


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_label_candidates(image: Path) -> list[Path]:
    return [image.with_suffix(extension) for extension in LABEL_EXTENSIONS]


def parse_voc_label(path: Path) -> list[dict[str, Any]]:
    root = ET.parse(path).getroot()
    objects: list[dict[str, Any]] = []
    for item in root.findall("object"):
        name = (item.findtext("name") or "unknown").strip()
        box = item.find("bndbox")
        if box is None:
            objects.append({"class_name": name, "box": None})
            continue
        try:
            bounds = {
                key: float(box.findtext(xml_key, "nan"))
                for key, xml_key in (
                    ("xmin", "xmin"),
                    ("ymin", "ymin"),
                    ("xmax", "xmax"),
                    ("ymax", "ymax"),
                )
            }
        except ValueError as exc:
            raise ValueError(f"invalid VOC bounding box: {exc}") from exc
        objects.append({"class_name": name, "box": bounds})
    return objects


def parse_coco_label(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not all(isinstance(payload.get(key), list) for key in ("images", "annotations", "categories")):
        raise ValueError("JSON file is not a COCO annotations document")
    return payload


def coco_box_is_valid(annotation: dict[str, Any], image: dict[str, Any]) -> bool:
    bbox = annotation.get("bbox")
    try:
        x, y, width, height = (float(value) for value in bbox)
        image_width = float(image["width"])
        image_height = float(image["height"])
    except (TypeError, ValueError, KeyError):
        return False
    return (
        image_width > 0
        and image_height > 0
        and width > 0
        and height > 0
        and x >= 0
        and y >= 0
        and x + width <= image_width
        and y + height <= image_height
    )


def voc_box_is_valid(box: dict[str, float] | None, size: tuple[int, int] | None) -> bool:
    if box is None or size is None:
        return False
    width, height = size
    return (
        width > 0
        and height > 0
        and 0 <= box["xmin"] < box["xmax"] <= width
        and 0 <= box["ymin"] < box["ymax"] <= height
    )


def inspect_dataset(dataset_root: Path) -> dict[str, Any]:
    dataset_root = dataset_root.expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_root}")
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {dataset_root}")

    images = sorted(path for path in dataset_root.rglob("*") if is_image(path))
    labels = sorted(path for path in dataset_root.rglob("*") if is_label(path))
    image_keys = {path.stem.casefold() for path in images}

    corrupted_files: list[dict[str, str]] = []
    invalid_boxes: list[dict[str, Any]] = []
    yolo_class_counts: Counter[str] = Counter()
    voc_class_counts: Counter[str] = Counter()
    coco_class_counts: Counter[str] = Counter()
    annotations_by_format = {"YOLO": 0, "VOC/XML": 0, "COCO/JSON": 0}
    coco_documents: list[dict[str, Any]] = []
    coco_referenced_images: set[str] = set()
    image_dimensions: Counter[str] = Counter()
    subset_counts: Counter[str] = Counter()

    for image in images:
        relative = image.relative_to(dataset_root)
        subset = relative.parts[0] if len(relative.parts) > 1 else "root"
        subset_counts[subset] += 1
        try:
            with Image.open(image) as picture:
                image_dimensions[f"{picture.width}x{picture.height}"] += 1
                picture.verify()
        except (
            OSError,
            ValueError,
            SyntaxError,
            Image.DecompressionBombError,
        ) as exc:
            corrupted_files.append(
                {"file": str(relative), "error": str(exc)}
            )

    for label in labels:
        relative = str(label.relative_to(dataset_root))
        try:
            if label.suffix.lower() == ".txt":
                boxes = parse_yolo_label(label)
                annotations_by_format["YOLO"] += len(boxes)
                for box in boxes:
                    yolo_class_counts[str(box["class_id"])] += 1
                    if not box_is_valid(box):
                        invalid_boxes.append({"file": relative, "box": box})
            elif label.suffix.lower() == ".xml":
                objects = parse_voc_label(label)
                annotations_by_format["VOC/XML"] += len(objects)
                root = ET.parse(label).getroot()
                try:
                    size = (
                        int(root.findtext("size/width", "0")),
                        int(root.findtext("size/height", "0")),
                    )
                except ValueError:
                    size = None
                for item in objects:
                    voc_class_counts[item["class_name"]] += 1
                    if not voc_box_is_valid(item["box"], size):
                        invalid_boxes.append({"file": relative, "box": item["box"]})
            elif label.suffix.lower() == ".json":
                document = parse_coco_label(label)
                coco_documents.append(document)
                categories = {
                    item.get("id"): str(item.get("name", item.get("id", "unknown")))
                    for item in document["categories"]
                }
                images_by_id = {item.get("id"): item for item in document["images"]}
                annotations_by_format["COCO/JSON"] += len(document["annotations"])
                for image_info in document["images"]:
                    if image_info.get("file_name"):
                        coco_referenced_images.add(Path(image_info["file_name"]).stem.casefold())
                for annotation in document["annotations"]:
                    class_name = categories.get(annotation.get("category_id"), "unknown")
                    coco_class_counts[class_name] += 1
                    image_info = images_by_id.get(annotation.get("image_id"))
                    if image_info is None or not coco_box_is_valid(annotation, image_info):
                        invalid_boxes.append({"file": relative, "annotation_id": annotation.get("id")})
        except (OSError, ValueError, ET.ParseError, json.JSONDecodeError) as exc:
            corrupted_files.append({"file": relative, "error": str(exc)})

    image_label_keys = {
        label.stem.casefold()
        for label in labels
        if label.suffix.lower() in {".txt", ".xml"}
    }
    known_label_keys = image_label_keys | coco_referenced_images
    missing_labels = [
        str(image.relative_to(dataset_root))
        for image in images
        if image.stem.casefold() not in known_label_keys
    ]
    orphan_labels = [
        str(label.relative_to(dataset_root))
        for label in labels
        if label.suffix.lower() in {".txt", ".xml"}
        and label.stem.casefold() not in image_keys
    ]
    missing_coco_images = sorted(coco_referenced_images - image_keys)

    # Prefer one canonical format to avoid double-counting datasets that ship
    # the same annotations in both COCO and VOC formats.
    if annotations_by_format["YOLO"]:
        class_counts = yolo_class_counts
        annotation_count = annotations_by_format["YOLO"]
    elif coco_documents:
        class_counts = coco_class_counts
        annotation_count = annotations_by_format["COCO/JSON"]
    else:
        class_counts = voc_class_counts
        annotation_count = annotations_by_format["VOC/XML"]

    duplicates: list[dict[str, str]] = []
    seen_hashes: dict[str, str] = {}
    for path in images + labels:
        try:
            digest = compute_sha256(path)
        except OSError as exc:
            corrupted_files.append(
                {"file": str(path.relative_to(dataset_root)), "error": str(exc)}
            )
            continue
        relative = str(path.relative_to(dataset_root))
        if digest in seen_hashes:
            duplicates.append({"file": relative, "same_as": seen_hashes[digest]})
        else:
            seen_hashes[digest] = relative

    issues: list[dict[str, Any]] = []
    for name, values in (
        ("missing_labels", missing_labels),
        ("orphan_labels", orphan_labels),
        ("missing_coco_images", missing_coco_images),
        ("corrupted_files", corrupted_files),
        ("invalid_boxes", invalid_boxes),
        ("duplicates", duplicates),
    ):
        if values:
            issues.append({"issue": name, "count": len(values)})

    return {
        "dataset": str(dataset_root),
        "inspected_at": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "images": len(images),
            "annotations": annotation_count,
            "classes_detected": len(class_counts),
            "label_files": len(labels),
            "annotation_files": len(labels),
            "txt_files": sum(path.suffix.lower() == ".txt" for path in labels),
            "xml_files": sum(path.suffix.lower() == ".xml" for path in labels),
            "json_files": sum(path.suffix.lower() == ".json" for path in labels),
        },
        "annotations_by_format": annotations_by_format,
        "annotation_formats": detect_annotation_format(labels),
        "classes": {str(key): value for key, value in sorted(class_counts.items())},
        "image_dimensions": dict(image_dimensions),
        "subset_distribution": dict(subset_counts),
        "missing_labels": missing_labels,
        "orphan_labels": orphan_labels,
        "missing_coco_images": missing_coco_images,
        "corrupted_files": corrupted_files,
        "invalid_boxes": invalid_boxes,
        "duplicates": duplicates,
        "quality": {
            "status": "PASS" if not issues else "WARN",
            "issues": issues,
            "issue_count": sum(item["count"] for item in issues),
        },
    }


def write_html_report(report: dict[str, Any], output_path: Path) -> None:
    rows = "".join(
        f"<tr><td>{html.escape(str(key))}</td>"
        f"<td>{html.escape(str(value))}</td></tr>"
        for key, value in report["counts"].items()
    )
    document = (
        "<html><head><meta charset='utf-8'><title>FIREGUARD Dataset Report</title>"
        "<style>body{font-family:Arial;margin:2rem}table{border-collapse:collapse}"
        "td,th{border:1px solid #bbb;padding:.5rem}</style></head><body>"
        f"<h1>FIREGUARD Dataset Report</h1>"
        f"<p>Status: {html.escape(report['quality']['status'])}</p>"
        f"<table><tr><th>Metric</th><th>Value</th></tr>{rows}</table>"
        f"<h2>Details</h2><pre>{html.escape(json.dumps(report, indent=2))}</pre>"
        "</body></html>"
    )
    output_path.write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a real FIREGUARD dataset without copying it."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Dataset root; defaults to FIREGUARD_DATASET_PATH or configured path",
    )
    parser.add_argument("--output", type=Path, default=Path("reports"))
    args = parser.parse_args()

    if args.dataset is None:
        from fireguard.config import settings

        if not settings.dataset_path:
            parser.error("Pass --dataset or set FIREGUARD_DATASET_PATH to a real dataset root.")
        args.dataset = Path(settings.dataset_path)

    args.output.mkdir(parents=True, exist_ok=True)
    try:
        report = inspect_dataset(args.dataset)
    except (FileNotFoundError, NotADirectoryError) as exc:
        parser.error(str(exc))

    (args.output / "dataset_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    write_html_report(report, args.output / "dataset_report.html")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
