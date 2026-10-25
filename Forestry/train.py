"""Train or fine-tune a YOLO detection model for Forestry."""
import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Train a YOLO model.")
    p.add_argument("--data", required=True, help="Dataset YAML path")
    p.add_argument("--model", default="yolov8s.pt", help="Starting checkpoint, e.g. yolov8s.pt or models/forestry_best.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default=None, help="0 for first NVIDIA GPU, cpu for CPU, omit for automatic selection")
    p.add_argument("--workers", type=int, default=0, help="Data-loader workers; 0 is a safe Windows default")
    p.add_argument("--project", default="runs")
    p.add_argument("--name", default="forestry_train")
    p.add_argument("--patience", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path.resolve()}")

    model_path = Path(args.model)
    if ("/" in args.model or "\\" in args.model) and not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path.resolve()}")

    model = YOLO(args.model)
    options = dict(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=args.patience,
        exist_ok=False,
    )
    if args.device is not None:
        options["device"] = args.device
    model.train(**options)
    print("Training finished. Check the run folder for weights and metrics.")


if __name__ == "__main__":
    # Main guard is important for Windows multiprocessing.
    main()
