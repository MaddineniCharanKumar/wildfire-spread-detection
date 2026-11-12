from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"


def _read_report(path: str | Path) -> dict[str, Any] | None:
    report_path = Path(path)
    if not report_path.is_absolute():
        report_path = (
            REPORTS_DIR.parent / report_path
            if report_path.parts and report_path.parts[0] == "reports"
            else REPORTS_DIR / report_path
        )
    try:
        with report_path.open(encoding="utf-8") as stream:
            report = json.load(stream)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        return {"status": "INVALID_REPORT", "error": str(exc)}
    if not isinstance(report, dict):
        return {"status": "INVALID_REPORT", "error": "Report root must be a JSON object."}
    return report


def extract_dataset_report(path: str | Path = "dataset_report.json") -> dict[str, Any] | None:
    report = _read_report(path)
    if report is None or report.get("status") == "INVALID_REPORT":
        return report

    source = report.get("dataset")
    if not source or source == "NOT_RUN":
        return report
    source_path = Path(str(source)).expanduser()
    if not source_path.is_absolute():
        source_path = REPORTS_DIR.parent / source_path
    if not source_path.exists():
        return {
            "status": "SOURCE_MISSING",
            "inspected_at": report.get("inspected_at"),
            "message": "The inspected dataset is no longer available. Run dataset inspection again.",
            "counts": None,
            "quality": {"status": "UNAVAILABLE"},
        }

    report = dict(report)
    report.pop("dataset", None)
    report["dataset_name"] = source_path.name
    return report


def extract_evaluation_report(path: str | Path = "latest_evaluation.json") -> dict[str, Any] | None:
    report = _read_report(path)
    if report is None or report.get("status") == "INVALID_REPORT":
        return report
    report = dict(report)
    for key in ("weights", "data"):
        value = report.get(key)
        if isinstance(value, str):
            report[key] = Path(value).name
    return report


def extract_live_provider_report(
    path: str | Path = "live_provider_report.json",
) -> dict[str, Any] | None:
    return _read_report(path)
