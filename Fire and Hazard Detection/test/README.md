# Fire and Smoke Detection Using YOLOv8

## 1. Project Overview

Fire and Smoke Detection is a computer-vision module developed as part of **PoC-3: An Advanced Multi-Mission Drone for Intrusion Detection and Inspection Using an AI-Based Real-Time System** at C-DAC ACTS, Pune.

The objective of this module is to detect fire and smoke in images and videos using deep-learning-based object detection. The system uses the YOLOv8 object-detection model to identify fire and smoke and generate predictions with bounding boxes and confidence scores.

This module is intended to support drone-based monitoring and inspection by helping identify potential fire and smoke hazards in visual data.

## 2. Project Objectives

- Develop a computer-vision workflow for fire and smoke detection.
- Prepare and use an annotated dataset for model training and evaluation.
- Train and evaluate a YOLOv8n object-detection model.
- Visualize training results and evaluate model performance.
- Test the trained model on images and videos.
- Prepare the model and workflow for potential integration into a drone-based monitoring system.

## 3. Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Object Detection | YOLOv8n |
| Deep Learning Framework | PyTorch |
| Model Library | Ultralytics |
| Dataset Management and Annotation | Roboflow |
| Development Environment | Google Colab |
| GPU | NVIDIA Tesla T4 |
| Image Processing | OpenCV, Pillow |
| Video Processing | FFmpeg |
| Visualization | Matplotlib |

## 4. Dataset Information

The model was trained using the **Fire and Smoke Videos dataset from Roboflow, Version 3**, in YOLO format.

The dataset contains annotated images belonging to the following classes:

- **Fire:** Visible flames or fire.
- **Smoke:** Visible smoke.

### Dataset Distribution

| Dataset Split | Number of Images |
|---|---:|
| Training | 844 |
| Validation | 106 |
| Testing | 105 |
| **Total** | **1,055** |

The dataset was organized into training, validation, and testing subsets to support model training and performance evaluation.

## 5. Model Training Configuration

The YOLOv8n model was trained using the following configuration:

| Parameter | Value |
|---|---|
| Model | YOLOv8n |
| Initial Weights | Pretrained YOLOv8n |
| Image Size | 640 × 640 |
| Number of Epochs | 50 |
| Training Environment | Google Colab |
| GPU | NVIDIA Tesla T4 |
| Model Framework | Ultralytics and PyTorch |
| Best Model Weights | `best.pt` |

The trained model weights are saved in the `best.pt` file.

## 6. Model Evaluation Results

The model was evaluated using standard object-detection metrics.

### Overall Performance

| Evaluation Metric | Result |
|---|---:|
| Precision | 76.22% |
| Recall | 66.21% |
| mAP@50 | 77.02% |
| F1-Score | 70.86% |

### Class-Wise Performance

| Class | Precision | Recall | mAP@50 |
|---|---:|---:|---:|
| Fire | 73.0% | 53.3% | 67.2% |
| Smoke | 79.4% | 79.1% | 86.8% |

### Interpretation

The model achieved an overall mAP@50 of 77.02%. Smoke detection achieved a higher recall and mAP@50 than fire detection in the reported evaluation results.

The results indicate that the model can identify fire and smoke in the evaluated dataset, although further improvements are needed, particularly to reduce missed fire detections.

*Note: These figures represent the reported evaluation results and do not guarantee the same performance in real-world drone operations.*

## 7. Training Results and Confusion Matrix

### Training Results

The following image presents the model's training results and evaluation information.

![Training Results](results.png)

### Confusion Matrix

The confusion matrix provides a visual summary of the model's classification performance across the evaluated classes.

![Confusion Matrix](confusion_matrix.png)

## 8. Testing

The trained model was tested on images and video to inspect its fire and smoke detection predictions.

### Image Testing

Image testing was used to check whether the model could identify fire and smoke and display bounding boxes around detected objects.

### Video Testing

The model was also tested on a video containing 495 frames. Video inference was used to inspect predictions across successive frames.

The reported testing activity included 18 images and a video containing 495 frames.

The results from these tests provide an initial assessment of model behavior. Broader testing with diverse environmental conditions is necessary before deployment.

## 9. Project Files

The folder contains the following files:

| File | Description |
|---|---|
| `Fire_Smoke_Detection_Model.ipynb` | Notebook containing the model workflow and experiments |
| `best.pt` | Trained YOLOv8n model weights |
| `args.yaml` | Training configuration and arguments |
| `requirements.txt` | Python dependencies |
| `results.png` | Training results visualization |
| `confusion_matrix.png` | Confusion matrix visualization |
| `README.md` | Project documentation |

## 10. Installation and Setup

### Step 1: Clone the Repository

Clone the main project repository:

```bash
git clone https://github.com/Aadityakhapare/PoC-3-An-advanced-multi-mission-drone.git
```

### Step 2: Navigate to the Project Folder

```bash
cd "PoC-3-An-advanced-multi-mission-drone/Fire and Hazard Detection/test"
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

Ensure that Python and the required dependencies are installed before running the model.

## 11. Running Inference

The following examples demonstrate how to run inference using the trained model with the Ultralytics library.

### Image Inference

```python
from ultralytics import YOLO

model = YOLO("best.pt")

results = model.predict(
    source="test_image.jpg",
    conf=0.25,
    save=True
)
```

Replace `test_image.jpg` with the path to the image you want to test.

### Video Inference

```python
from ultralytics import YOLO

model = YOLO("best.pt")

results = model.predict(
    source="test_video.mp4",
    conf=0.25,
    save=True
)
```

Replace `test_video.mp4` with the path to your video file.

The prediction output is saved by Ultralytics when `save=True` is enabled. These examples assume that the model weights and input files are accessible from the current working directory.

## 12. Limitations

- The model's performance depends on the quality and diversity of the training dataset.
- Smoke can be difficult to distinguish from clouds, fog, dust, and other visually similar objects.
- Fire detection performance may vary with flame size, lighting, distance, and environmental conditions.
- The reported evaluation results do not establish reliability in every real-world scenario.
- Live drone deployment and real-time performance must be evaluated separately before operational use.

## 13. Future Improvements

Potential improvements include:

- Expanding the dataset with diverse fire and smoke examples.
- Improving the representation of small, distant, and early-stage fires.
- Adding challenging background examples to reduce false detections.
- Improving recall, precision, and overall detection performance.
- Testing model performance on more varied images and videos.
- Evaluating inference speed and resource requirements for drone hardware.
- Integrating the detection module with a drone-based monitoring system after suitable validation.

## 14. Applications

Potential applications of fire and smoke detection include:

- Forest and wildfire monitoring.
- Industrial site inspection.
- Fire hazard surveillance.
- Monitoring difficult-to-access areas.
- Drone-assisted visual inspection and emergency assessment.

These are potential applications; operational suitability depends on further testing and validation.


