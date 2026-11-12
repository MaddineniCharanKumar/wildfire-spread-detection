from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Generate a placeholder dataset report when inspection has not yet run.")
    parser.add_argument("--output", type=Path, default=Path("reports"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "NOT_RUN",
        "status": "NOT_INSPECTED",
        "message": "The dataset has not been inspected yet. Run: python scripts/inspect_dataset.py --dataset <path> --output reports",
        "counts": {"images": 0, "label_files": 0, "xml_files": 0, "json_files": 0},
        "annotation_formats": [],
        "classes": {},
        "missing_labels": [],
        "orphan_labels": [],
        "corrupted_files": [],
        "invalid_boxes": [],
        "duplicates": [],
        "quality": {"status": "NOT_RUN", "issue_count": 0, "issues": []},
    }
    (args.output / "dataset_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (args.output / "dataset_report.html").write_text(
        "<html><body><h1>FIREGUARD Dataset Report</h1><p>Dataset has not been inspected yet.</p><pre>{}</pre></body></html>".format(json.dumps(payload, indent=2)),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
