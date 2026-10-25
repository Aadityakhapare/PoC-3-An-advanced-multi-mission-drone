"""Run YOLO inference on an image, folder, video, webcam, or supported stream."""
import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Run Forestry YOLO inference.")
    p.add_argument("--source", required=True, help="Image/video/folder path, webcam index (0), or stream URL")
    p.add_argument("--model", default="models/forestry_best.pt", help="Path to model weights")
    p.add_argument("--conf", type=float, default=0.35, help="Confidence threshold between 0 and 1")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None, help="0 for first NVIDIA GPU, cpu for CPU, omit for automatic selection")
    p.add_argument("--project", default="runs/predict")
    p.add_argument("--name", default="forestry_test")
    p.add_argument("--show", action="store_true", help="Display preview window")
    p.add_argument("--save-txt", action="store_true", help="Save predicted labels as text files")
    p.add_argument("--save-conf", action="store_true", help="Include confidence in saved label files")
    p.add_argument("--max-det", type=int, default=300)
    return p.parse_args()


def main():
    args = parse_args()
    if not 0 <= args.conf <= 1:
        raise ValueError("--conf must be between 0 and 1.")
    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found: {model_path.resolve()}\n"
            "Download the model from the GitHub Releases page or pass the correct path with --model."
        )
    source = int(args.source) if args.source.isdigit() else args.source
    model = YOLO(str(model_path))
    options = dict(
        source=source, conf=args.conf, imgsz=args.imgsz,
        project=args.project, name=args.name, exist_ok=True,
        save=True, show=args.show, save_txt=args.save_txt,
        save_conf=args.save_conf, max_det=args.max_det,
    )
    if args.device is not None:
        options["device"] = args.device
    results = model.predict(**options)
    print(f"Finished. Results saved under: {Path(args.project) / args.name}")
    print(f"Processed {len(results)} result item(s).")


if __name__ == "__main__":
    main()
