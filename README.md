# Fabric Defect Detection using YOLOv8

## Overview
This project focuses on fabric defect detection using YOLOv8 for object detection and classification. It identifies and classifies multiple fabric defects within the same object, improving quality control in textile manufacturing.

## Features
- Detects and classifies fabric damages into 7 defect categories.
- Supports multiple defect detection in a single fabric sample.
- Uses YOLOv8 for efficient object detection.
- Provides a backend for predictions and a frontend for visualization.

## Dataset Structure
The dataset is structured into three folders:
- `train/` - Training images
- `valid/` - Validation images
- `test/` - Testing images

**Number of Classes (`nc`):** 7

## Training the Model
The model is trained using `Yolo_train.py` for 50 epochs on the provided dataset.
```bash
python Yolo_train.py
```

## Accuracy Calculation
The script `accuracy.py` is used to evaluate the trained model's performance.
```bash
python accuracy.py
```

## Backend Prediction
The backend fapp.py processes images and predicts fabric defects using the trained YOLOv8 model.
```bash
python fapp.py
```
use this command to run the backend server

## Frontend
* index.html - Provides an interface for users to upload fabric images.
* style.css - Handles the UI styling.

## Conclusion
This project enhances automated fabric defect detection by leveraging YOLOv8 for accurate and efficient defect identification, reducing manual inspection efforts in textile industries.

