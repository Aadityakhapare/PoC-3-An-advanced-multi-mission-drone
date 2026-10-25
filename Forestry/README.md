# Forestry — YOLOv8 Detection

YOLOv8 object detection project for:
- `0`: animal
- `1`: fallen_tree
- `2`: tree

## Install

Python 3.10 or 3.11 recommended.

```bash
git clone REPOSITORY_URL
cd Forestry
python -m venv .venv
```

Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

Windows CMD:
```bat
.venv\Scripts\activate.bat
```

Linux/macOS:
```bash
source .venv/bin/activate
```

Install packages:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For NVIDIA GPU acceleration, install a compatible CUDA-enabled PyTorch build using https://pytorch.org/get-started/locally/ . Check with:
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Get the model

Download the `.pt` model asset from this repository's GitHub **Releases** page and save it as `models/forestry_best.pt`. If no release exists yet, the repository owner must create one and attach the weights.

## Test images/videos

```bash
python test.py --source "path/to/image.jpg" --model models/forestry_best.pt
python test.py --source "path/to/video.mp4" --model models/forestry_best.pt
python test.py --source "path/to/images_folder" --model models/forestry_best.pt
python test.py --source 0 --model models/forestry_best.pt --show
```

Adjust confidence threshold and image size:
```bash
python test.py --source "path/to/video.mp4" --model models/forestry_best.pt --conf 0.35 --imgsz 640
```

Predictions are saved to `runs/predict/forestry_test/` by default.

## Train / fine-tune

Edit `configs/data.yaml` to match the paths to your YOLO-format dataset.

Train from the standard YOLOv8 small checkpoint:
```bash
python train.py --data configs/data.yaml --model yolov8s.pt --epochs 50 --imgsz 640 --batch 8
```

Fine-tune from the Forestry checkpoint:
```bash
python train.py --data configs/data.yaml --model models/forestry_best.pt --epochs 20 --imgsz 640 --batch 8 --device 0 --workers 0
```

Use `--device cpu` for CPU-only training. `--device 0` selects the first NVIDIA GPU. `--workers 0` is a safe Windows starting point. Training output is saved under `runs/forestry_train/`; the best checkpoint is usually `runs/forestry_train/weights/best.pt`.

## Expected dataset layout

```text
dataset/
├── train/images/
├── train/labels/
├── valid/images/
├── valid/labels/
├── test/images/
└── test/labels/
```

Ensure label class IDs match the class order in `configs/data.yaml`.

## Important

- Dataset and trained weights are kept out of the Git repository by default.
- Check you have permission to redistribute datasets and weights before publishing them.
- Model predictions can be wrong; evaluate on representative footage before relying on them.

Docs: https://docs.ultralytics.com/ and https://pytorch.org/get-started/locally/
