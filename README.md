# Real Time Hand Hygiene Monitoring 🧼👐

Welcome!
This project builds a real-time hand hygiene monitoring system that tracks and analyzes the WHO-recommended 7-step handwashing protocol using only a standard webcam.

The AI uses Computer Vision with YOLOv8-Nano models and MediaPipe hand tracking for accurate pose detection.

## Project Overview
There are two main versions of the monitoring system:

| File Name        | Description |
|------------------|-------------|
| `main.py`        | Complete 7-step WHO handwashing protocol monitoring with file.html interface |
| `app.py`         | Single pose testing system with index.html for individual pose validation |
| `accuracy.py`    | Calculate final real-time model accuracy from detection logs |
| `fps_calculate.py` | Analyze frame rate performance during detection |
| `fps_graph.py`   | Generate FPS performance visualization graphs |
| `templates/`     | HTML templates for web interface |
| `├── file.html`  | Full 7-step handwashing monitoring interface |
| `├── index.html` | Single pose testing interface |
| `static/`        | SureWash logo and branding assets |
| `requirements.txt` | List of required Python packages |
| `README.md`      | Instructions about the project (this file) |

## How to Set Up and Run

### 1. Install Required Libraries
Open your terminal or command prompt and install the required libraries by running:

```bash
pip install flask opencv-python mediapipe ultralytics numpy matplotlib
```

### 2. Make Sure These Files Are Together
Make sure the following files are in the same folder:
- main.py (complete 7-step system)
- app.py (single pose testing)
- accuracy.py
- fps_calculate.py
- fps_graph.py
- templates/file.html (7-step interface)
- templates/index.html (single pose interface)
- static/ (SureWash logo)
- requirements.txt
- README.md

### 3. To Run the Complete 7-Step Handwashing System

```bash
python main.py
```
- Opens browser at `http://localhost:5000` with file.html interface
- Monitors all 7 WHO handwashing steps sequentially
- Progress tracked separately for left and right hands
- Complete handwashing protocol validation

### 4. To Run Single Pose Testing

```bash
python app.py
```
- Opens browser at `http://localhost:5000` with index.html interface
- Test individual handwashing poses one at a time
- Useful for validating specific pose detection
- Single pose real-time integration testing

### 5. To Check Performance Results

```bash
python accuracy.py
```
- Calculates real-time accuracy from detection logs
- Shows precision, recall, and F1-score metrics

```bash
python fps_calculate.py
```
- Analyzes frame rate performance
- Generates FPS statistics and logs

## What I Learned
- How to create custom datasets and annotate handwashing poses using LabelImg
- How to train YOLOv8-Nano models for real-time object detection
- How to integrate MediaPipe for hand landmark tracking and left/right classification
- How to build Flask web applications with live video streaming
- How AI can provide real-time feedback for healthcare compliance monitoring

## Project Requirements

You must have Python and these libraries installed:
- flask
- opencv-python
- mediapipe
- ultralytics
- numpy
- matplotlib

You can install all at once by running:

```bash
pip install flask opencv-python mediapipe ultralytics numpy matplotlib
```

## Author
**Sanjana Kottalavadi Shivashankarappa**