from __future__ import annotations

from pathlib import Path

import yaml

from training.train import build_dataset_yaml


def test_training_preserves_explicit_dataset_class_names(tmp_path: Path) -> None:
    config = {
        "path": ".",
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {0: "fire", 1: "smoke"},
    }
    dataset_yaml = tmp_path / "dataset.yaml"
    dataset_yaml.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = build_dataset_yaml(tmp_path)

    assert result["names"] == config["names"]
    assert yaml.safe_load(dataset_yaml.read_text(encoding="utf-8")) == config
