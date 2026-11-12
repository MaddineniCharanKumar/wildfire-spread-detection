from __future__ import annotations

from types import SimpleNamespace

import pytest
import yaml

from training.metrics import summarize_validation_metrics, verify_model_matches_dataset


def test_summarize_validation_metrics_uses_ultralytics_box_keys_and_speed() -> None:
    metrics = SimpleNamespace(
        results_dict={
            "metrics/precision(B)": 0.8,
            "metrics/recall(B)": 0.6,
            "metrics/mAP50(B)": 0.7,
            "metrics/mAP50-95(B)": 0.4,
        },
        speed={"preprocess": 2, "inference": 8, "loss": 1, "postprocess": 2},
        box=SimpleNamespace(
            ap_class_index=[1],
            p=[0.8],
            r=[0.6],
            ap50=[0.7],
            ap=[[0.5, 0.3]],
        ),
    )

    report = summarize_validation_metrics(metrics, {0: "fire", 1: "smoke"})

    assert report["precision"] == 0.8
    assert report["recall"] == 0.6
    assert report["f1"] == 0.6857142857142857
    assert report["mAP50"] == 0.7
    assert report["mAP50_95"] == 0.4
    assert report["fps"] == 1000 / 12
    assert report["inference_fps"] == 125
    assert report["per_class"]["1"] == {
        "name": "smoke",
        "precision": 0.8,
        "recall": 0.6,
        "f1": 0.6857142857142857,
        "ap50": 0.7,
        "ap50_95": 0.4,
    }
    assert report["per_class"]["0"] == {
        "name": "fire",
        "precision": None,
        "recall": None,
        "f1": None,
        "ap50": None,
        "ap50_95": None,
    }


def test_summarize_validation_metrics_reports_each_dataset_class() -> None:
    report = summarize_validation_metrics(
        SimpleNamespace(
            results_dict={},
            speed={},
            box=SimpleNamespace(
                ap_class_index=[0, 1],
                p=[0.5, 0.75],
                r=[0.25, 0.5],
                ap50=[0.4, 0.7],
                ap=[[0.3, 0.2], [0.6, 0.4]],
            ),
        ),
        {0: "fire", 1: "smoke"},
    )

    assert report["per_class"]["0"]["f1"] == 1 / 3
    assert report["per_class"]["1"]["f1"] == 0.6


def test_summarize_validation_metrics_does_not_replace_missing_values_with_zero() -> None:
    report = summarize_validation_metrics(SimpleNamespace(results_dict={}, speed={}))

    assert report["precision"] is None
    assert report["recall"] is None
    assert report["f1"] is None
    assert report["mAP50"] is None
    assert report["mAP50_95"] is None
    assert report["fps"] is None


def test_evaluation_requires_checkpoint_classes_to_match_dataset(tmp_path) -> None:
    data_yaml = tmp_path / "dataset.yaml"
    data_yaml.write_text(
        yaml.safe_dump({"names": {0: "fire", 1: "smoke"}}),
        encoding="utf-8",
    )

    assert verify_model_matches_dataset({0: "Fire", 1: "Smoke"}, data_yaml) == {
        0: "fire",
        1: "smoke",
    }
    with pytest.raises(ValueError, match="class names do not match"):
        verify_model_matches_dataset({0: "person", 1: "bicycle"}, data_yaml)
