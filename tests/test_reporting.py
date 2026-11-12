from __future__ import annotations

import json
from pathlib import Path

from fireguard.reporting import extract_dataset_report, extract_evaluation_report


def test_dataset_report_is_marked_unavailable_when_its_source_disappears(tmp_path: Path) -> None:
    report_path = tmp_path / "dataset_report.json"
    report_path.write_text(
        json.dumps({
            "dataset": str(tmp_path / "missing_dataset"),
            "inspected_at": "2026-10-13T07:08:49+00:00",
            "counts": {"images": 0, "annotations": 2390},
            "quality": {"status": "WARN"},
        }),
        encoding="utf-8",
    )

    report = extract_dataset_report(report_path)

    assert report is not None
    assert report["status"] == "SOURCE_MISSING"
    assert report["counts"] is None
    assert "dataset" not in report


def test_evaluation_report_hides_local_dataset_and_weight_paths(tmp_path: Path) -> None:
    report_path = tmp_path / "latest_evaluation.json"
    report_path.write_text(
        json.dumps({
            "weights": str(tmp_path / "best.pt"),
            "data": str(tmp_path / "dataset.yaml"),
            "precision": 0.9,
        }),
        encoding="utf-8",
    )

    report = extract_evaluation_report(report_path)

    assert report is not None
    assert report["weights"] == "best.pt"
    assert report["data"] == "dataset.yaml"
