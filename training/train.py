from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from fireguard.config import settings
from fireguard.models import _build_weight_name, load_yolo_model


def build_dataset_yaml(data_dir: str | Path, output_path: str | Path | None = None) -> dict:
    data_dir = Path(data_dir)
    output = Path(output_path) if output_path else data_dir / "dataset.yaml"
    if output.exists():
        existing = yaml.safe_load(output.read_text(encoding="utf-8")) or {}
        if not isinstance(existing, dict) or not existing.get("names"):
            raise ValueError(f"Dataset YAML must define class names: {output}")
        return existing

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    names: dict[int, str] = {}
    for label_file in list(train_dir.rglob("*.txt")) + list(val_dir.rglob("*.txt")) + list(test_dir.rglob("*.txt")):
        for line in label_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            parts = line.split()
            if not parts:
                continue
            try:
                class_id = int(parts[0])
            except ValueError:
                continue
            if class_id not in names:
                names[class_id] = f"class_{class_id}"

    if not names:
        raise ValueError(
            f"No YOLO classes found in {data_dir}; provide dataset.yaml with a names mapping."
        )

    dataset = {
        "path": str(data_dir),
        "train": "train/images" if (train_dir / "images").is_dir() else "train",
        "val": "val/images" if (val_dir / "images").is_dir() else "val",
        "test": "test/images" if (test_dir / "images").is_dir() else "test",
        "names": names,
    }

    output.write_text("\n".join([
        f"path: {dataset['path']}",
        f"train: {dataset['train']}",
        f"val: {dataset['val']}",
        f"test: {dataset['test']}",
        "names:",
        *[f"  {idx}: '{name}'" for idx, name in sorted(names.items())],
    ]) + "\n", encoding="utf-8")
    return dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a YOLO model with dynamically discovered classes.")
    parser.add_argument("--data", type=str, required=True, help="Dataset root directory containing train/val/test")
    parser.add_argument("--weights", type=str, default=None, help="Model weights (default: family+size from config, e.g., yolo26n.pt)")
    parser.add_argument("--family", type=str, default=None, help="Model family (default: FIREGUARD_MODEL_FAMILY)")
    parser.add_argument("--size", type=str, default=None, help="Model size: n, s, m, l, x (default: FIREGUARD_MODEL_SIZE)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0" if __import__("utils.device", fromlist=["detect_device"]).detect_device() == "cuda" else "cpu")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--name", type=str, default="train", help="Output run name under runs/train")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # Resolve model weights
    family = args.family or settings.model_family
    size = args.size or settings.model_size
    custom_weights = args.weights or settings.model_weights

    weight_name = _build_weight_name(family, size)
    weights_path = custom_weights or weight_name

    data_dir = Path(args.data)
    try:
        dataset_cfg = build_dataset_yaml(data_dir, data_dir / "dataset.yaml")
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))

    model, meta = load_yolo_model(weights_path, args.device)
    if model is None:
        print(f"[ERROR] Failed to load model: {meta.get('error')}")
        return 1

    # Load hyperparameters
    hp_path = Path("training/hyperparameters.yaml")
    hyperparams = {}
    if hp_path.exists():
        with hp_path.open() as f:
            hyperparams = yaml.safe_load(f) or {}

    print(f"[INFO] Training with model: {weights_path} (family={family}, size={size})")
    model.train(
        data=str(data_dir / "dataset.yaml"),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        resume=args.resume,
        project=str((Path.cwd() / "runs" / "train").resolve()),
        name=args.name,
        exist_ok=True,
        workers=args.workers,
        **hyperparams,
    )
    print(f"Training completed using dataset config: {dataset_cfg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())