from __future__ import annotations

import argparse
from pathlib import Path

from fireguard.config import settings
from fireguard.models import _build_weight_name, load_yolo_model


def main():
    parser = argparse.ArgumentParser(description="Export a YOLO model to PyTorch, ONNX, TensorRT or TorchScript formats")
    parser.add_argument("--weights", type=str, default=None, help="Model weights (default: family+size from config)")
    parser.add_argument("--family", type=str, default=None, help="Model family (default: FIREGUARD_MODEL_FAMILY)")
    parser.add_argument("--size", type=str, default=None, help="Model size (default: FIREGUARD_MODEL_SIZE)")
    parser.add_argument("--format", choices=["torchscript", "onnx", "engine", "pytorch"], default="onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=Path, default=Path("models/exported"))
    parser.add_argument("--verify", action="store_true", help="Verify exported model loads and runs one prediction")
    args = parser.parse_args()

    family = args.family or settings.model_family
    size = args.size or settings.model_size
    custom_weights = args.weights or settings.model_weights

    weight_name = _build_weight_name(family, size)
    weights_path = custom_weights or weight_name

    args.output.mkdir(parents=True, exist_ok=True)

    model, meta = load_yolo_model(weights_path, args.device)
    if model is None:
        print(f"[ERROR] Failed to load model: {meta.get('error')}")
        return

    print(f"[INFO] Exporting model: {weights_path} to {args.format}")
    exported_path = model.export(format=args.format, imgsz=args.imgsz, device=args.device)

    print(f"[SUCCESS] Export completed: {exported_path}")

    # Verify exported model
    if args.verify:
        print("[INFO] Verifying exported model...")
        try:
            from ultralytics import YOLO
            test_model = YOLO(exported_path)

            # Create a dummy image for verification
            import numpy as np
            from PIL import Image
            dummy = Image.fromarray(np.zeros((640, 640, 3), dtype=np.uint8))
            results = test_model(dummy, imgsz=args.imgsz, device=args.device, verbose=False)
            print(f"[SUCCESS] Exported model verified - ran prediction on dummy image")
            print(f"  Output: {len(results)} results, {sum(len(r.boxes) if r.boxes else 0 for r in results)} detections")

        except Exception as e:
            print(f"[WARNING] Export completed but verification failed: {e}")

    print(f"[INFO] Exported model saved to: {exported_path}")


if __name__ == "__main__":
    main()