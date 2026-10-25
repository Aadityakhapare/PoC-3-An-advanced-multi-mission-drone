# Fire and Smoke Detection Using YOLOv8

## Project Overview

This project implements a computer-vision-based Fire and Smoke Detection system using YOLOv8. It detects two classes in images and videos: **Fire and Smoke**.

The trained model is evaluated on images and video footage to identify visible fire and smoke regions. The project forms part of the Fire and Smoke Detection module under **PoC-3: An Advanced Multi-Mission Drone for Intrusion Detection and Inspection Using an AI-Based Real-Time System**.

## Technologies Used

- Python
- YOLOv8n and Ultralytics
- Roboflow
- Google Colab
- NVIDIA Tesla T4 GPU
- PyTorch
- OpenCV
- FFmpeg
- Pillow
- Matplotlib

## Dataset

The project uses the **Fire and Smoke Videos** dataset from Roboflow, Version 3, exported in YOLO format.

The dataset contains two classes:

1. Fire
2. Smoke

The dataset includes images and their corresponding annotations in YOLO object-detection format.

### Dataset Split

| **Split** | **Images** |
|---|---:|
| Training | 844 |
| Validation | 106 |
| Testing | 105 |
| **Total** | **1,055** |

## Model Training

The model was trained using a pretrained YOLOv8n checkpoint on Google Colab with an NVIDIA Tesla T4 GPU.

| **Parameter** | **Value** |
|---|---|
| Model | YOLOv8n |
| Image size | 640 × 640 |
| Training epochs | 50 |
| Training device | NVIDIA Tesla T4 GPU |
| Dataset platform | Roboflow |
| Trained weights | `best.pt` |

The best-performing model checkpoint was saved as `best.pt` and used for subsequent image and video inference.

## Test Results

The model was evaluated using the validation dataset.

| **Metric** | **Result** |
|---|---:|
| Precision | 76.22% |
| Recall | 66.21% |
| mAP@50 | 77.02% |
| F1-score | 70.86% |

These results represent the reported validation performance. Detection results on unseen images and videos may differ.

### Class-Wise Performance

| **Class** | **Precision** | **Recall** | **mAP@50** |
|---|---:|---:|---:|
| Fire | 73.0% | 53.3% | 67.2% |
| Smoke | 79.4% | 79.1% | 86.8% |

The model achieved stronger detection performance for smoke than for fire. Improving fire detection recall is an important area for further development.

## Image and Video Detection

The trained model was tested on both images and video footage.

The inference workflow supports:

- Loading the trained YOLOv8 model.
- Detecting fire and smoke in images.
- Drawing bounding boxes around detected objects.
- Processing video frames for fire and smoke detection.
- Saving annotated prediction results.
- Converting video output to MP4 format using FFmpeg.

The notebook tested the model on 18 images and processed a video containing 495 frames.

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Aadityakhapare/PoC-3-An-advanced-multi-mission-drone.git
cd PoC-3-An-advanced-multi-mission-drone
```

Navigate to the `Fire and Smoke Detection` directory.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Alternatively, install Ultralytics directly:

```bash
pip install ultralytics
```

FFmpeg may be installed separately if video conversion is required. Ensure it is available in your system PATH.

### 3. Check the model file

Place `best.pt` in the working directory or provide the correct path to the trained model.

### 4. Run detection on an image

```python
from ultralytics import YOLO

model = YOLO("best.pt")

results = model.predict(
    source="test_image.jpg",
    conf=0.25,
    save=True
)
```

Replace `test_image.jpg` with the path to your input image.

### 5. Run detection on a video

```python
from ultralytics import YOLO

model = YOLO("best.pt")

results = model.predict(
    source="test_video.mp4",
    conf=0.25,
    save=True
)
```

Replace `test_video.mp4` with the path to your input video. The annotated results are saved in the prediction output directory.

### 6. Run the notebook

Open `Fire_Smoke_Detection_Model.ipynb` in Google Colab or Jupyter Notebook.

For training or dataset download, configure Roboflow authentication securely. Do not publish API keys or authenticated dataset download URLs in the notebook.

## Limitations

- The model detects only the two trained classes: Fire and Smoke.
- Detection performance may vary with lighting, image quality, smoke density, object size, and camera distance.
- The reported fire recall indicates that some fire instances may be missed.
- Testing on a limited set of images and video footage does not establish reliability in all environments.
- Real-time drone deployment and onboard inference require additional testing and optimization.

## Future Improvements

- Increase the diversity and size of the training dataset.
- Improve fire detection recall and reduce false negatives.
- Evaluate the model on additional independent test data.
- Optimize inference for drone hardware.
- Integrate the model with a live drone camera feed.
- Develop an alert-generation mechanism for potential fire incidents.
- Validate detection performance under different environmental conditions.

## Applications

Potential applications include:

- Forest fire monitoring.
- Industrial fire and smoke detection.
- Aerial surveillance and inspection.
- Infrastructure safety monitoring.
- Early visual identification of potential fire hazards.

These are potential applications and require further validation before operational deployment.

## Author

Developed as part of a computer-vision and deep-learning project under **PoC-3: An Advanced Multi-Mission Drone for Intrusion Detection and Inspection Using an AI-Based Real-Time System**, at C-DAC ACTS, Pune.

## Results and Screenshots

### Training Results

[View Training Results](results.png)

### Confusion Matrix

[View Confusion Matrix](confusion_matrix.png)

### Model Weights

The trained model is available in `best.pt`, subject to the repository's file availability and size limits.

