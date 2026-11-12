# Dataset preparation and evaluation

The project reports only statistics read from the supplied data. Start by inspecting the **dataset root that contains both images and annotations**:

```bash
python scripts/inspect_dataset.py --dataset /path/to/FASDD_UAV --output reports
```

Inspection counts actual object annotations in YOLO, Pascal VOC, or COCO data. When the same annotations are present in multiple formats, it reports per-format totals and selects one canonical format to avoid double-counting. COCO image references are checked against local images; VOC/YOLO labels are matched by filename even when annotations and images are in separate subdirectories. Reports are written to `reports/dataset_report.json` and `reports/dataset_report.html`.

FASDD_UAV preparation uses the official COCO train/validation/test splits and expects the source to contain both `images/` and `annotations/`:

```bash
python scripts/prepare_fasdd_uav.py --source C:\path\to\FASDD_UAV --output data\processed --copy
```

For a reproducible CPU pilot, add `--max-images-per-split 1000 --seed 42`; the cap is sampled with fire/smoke class balance, and the report records the sample scope. Without the cap, the full official splits are prepared. The generated dataset YAML uses the resolved output root so Ultralytics can locate the images regardless of the current working directory. The preparation report includes missing source images and invalid boxes. Do not train if the prepared split has missing images or zero labeled objects. The included `yolo26n.pt` is a general COCO model, not a fire/smoke detector.

Train using the generated class names and split paths:

```bash
python training/train.py --data data\processed --weights yolo26n.pt --epochs 50 --workers 2 --name fasdd-uav
```

Run evaluation against the validation split while iterating, then use the held-out test split once for final reporting:

```bash
python training/evaluate.py --weights runs\train\train\weights\best.pt --data data\processed\dataset.yaml --split val --output reports\eval
python training/evaluate.py --weights runs\train\train\weights\best.pt --data data\processed\dataset.yaml --split test --output reports\final
```

The evaluator rejects checkpoints whose class names do not exactly match the dataset YAML, preventing generic COCO weights from producing misleading fire/smoke scores. The metrics JSON and model card contain overall precision, recall, F1, mAP@50, mAP@50-95, and FPS measured from the selected evaluation split, plus per-class precision, recall, F1, AP@50, and AP@50-95 for every dataset class. FPS is reported both as the complete evaluation pipeline and model inference-only throughput. Metrics and FPS are `null`/`N/A` when the framework does not provide a measured value or a class has no evaluable examples; they are never substituted with zero. The latest run is also written to `reports/latest_evaluation.json` and exposed by the analytics API/dashboard.

For a quick CPU-only exploratory run on Windows PowerShell, use a small deterministic sample and label the limited scope in the evaluation report:

```powershell
python scripts\prepare_fasdd_uav.py --source "C:\path\to\FASDD_UAV" --output data\processed --classes both --max-images-per-split 100 --seed 42
python -m training.train --data data\processed --weights yolo26n.pt --epochs 5 --batch 16 --imgsz 160 --device cpu --workers 2 --name fasdd-uav-pilot
python -m training.evaluate --weights runs\train\fasdd-uav-pilot\weights\best.pt --data data\processed\dataset.yaml --split test --imgsz 160 --device cpu --output reports\pilot-test --label "Exploratory CPU pilot: 100 deterministic class-balanced images per official split (seed 42), 5 epochs, 160px; limited sample, not a production benchmark."
```

Use a unique `--name` for each training run. These pilot scores are real measurements on the sampled held-out test data, but the small sample and short training run are not a production benchmark.

Verify live providers and save the run log:

```bash
python scripts/verify_live_sources.py --output reports\live_provider_report.json
```

Set `FIREGUARD_LATITUDE` and `FIREGUARD_LONGITUDE` in the local environment configuration before running. The report omits coordinates and credentials. Use `--skip-firms` to verify public providers without using the configured NASA FIRMS API key; the report will be marked `PARTIAL`. A missing dataset or unavailable provider is reported as unavailable/failing; no synthetic statistics are generated.
