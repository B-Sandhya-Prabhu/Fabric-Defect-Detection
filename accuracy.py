import os
import cv2
import time
import numpy as np
from ultralytics import YOLO
from sklearn.metrics import classification_report, accuracy_score

# ========== LOAD THE TRAINED YOLOv8 MODEL ==========
model = YOLO('/model/FDD_yolo_training/yolov8_fdd_model/weights/best.pt')

# ========== SET THE TEST IMAGE FOLDER ==========
test_folder = './test/images'  # Path to the folder containing test images
ground_truth_folder = './test/labels'  # Path to folder containing ground truth labels (YOLO format)

# ========== VARIABLES TO TRACK INFERENCE TIME AND PREDICTIONS ==========
prediction_times = []
true_labels = []  # Store ground-truth labels
predicted_labels = []  # Store predicted labels
ious = []  # Store IoUs for detected objects

# Function to calculate Intersection over Union (IoU)
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)

    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

# Function to parse YOLO labels from text file
def parse_yolo_label(label_path):
    boxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as file:
            for line in file.readlines():
                data = line.strip().split()
                if len(data) == 5:
                    cls = int(data[0])
                    x, y, w, h = map(float, data[1:])
                    boxes.append((cls, x, y, w, h))
    return boxes

# Convert normalized YOLO box to absolute coordinates
def denormalize_box(box, img_width, img_height):
    cls, x, y, w, h = box
    x1 = int((x - w / 2) * img_width)
    y1 = int((y - h / 2) * img_height)
    x2 = int((x + w / 2) * img_width)
    y2 = int((y + h / 2) * img_height)
    return cls, x1, y1, x2, y2

# ==========  LOOP THROUGH TEST IMAGES ==========
for image_file in os.listdir(test_folder):
    if image_file.endswith(('.jpg', '.png', '.jpeg')):  # Check for image files only
        image_path = os.path.join(test_folder, image_file)
        label_path = os.path.join(ground_truth_folder, image_file.replace('.jpg', '.txt').replace('.png', '.txt'))

        # Read image dimensions
        image = cv2.imread(image_path)
        img_height, img_width = image.shape[:2]

        # Start timing the prediction
        start_time = time.time()
        results = model(image_path)  # YOLO prediction
        end_time = time.time()

        # Calculate prediction time
        prediction_times.append(end_time - start_time)

        # Parse ground truth labels
        ground_truth_boxes = parse_yolo_label(label_path)

        # Parse predicted boxes
        pred_boxes = results[0].boxes.xyxy.cpu().numpy()  # Predicted bounding boxes
        pred_classes = results[0].boxes.cls.cpu().numpy()  # Predicted classes

        for gt_box in ground_truth_boxes:
            gt_cls, gt_x1, gt_y1, gt_x2, gt_y2 = denormalize_box(gt_box, img_width, img_height)
            true_labels.append(gt_cls)

            # Match with predicted boxes
            best_iou = 0
            best_pred_cls = None
            for pred_box, pred_cls in zip(pred_boxes, pred_classes):
                iou = calculate_iou((gt_x1, gt_y1, gt_x2, gt_y2), pred_box)
                if iou > best_iou:
                    best_iou = iou
                    best_pred_cls = int(pred_cls)

            ious.append(best_iou)
            predicted_labels.append(best_pred_cls)

# ========== CALCULATE METRICS ==========
# Remove None values where no prediction was made
clean_true_labels = [true for true, pred in zip(true_labels, predicted_labels) if pred is not None]
clean_predicted_labels = [pred for pred in predicted_labels if pred is not None]
clean_ious = [iou for iou in ious if iou > 0]

# Classification metrics
accuracy = accuracy_score(clean_true_labels, clean_predicted_labels)
classification_report_str = classification_report(clean_true_labels, clean_predicted_labels, 
                target_names=['baekra', 'color issues', 'contamination', 'cut', 'gray stitch', 'selvet', 'stain'])

# IoU statistics
avg_iou = sum(clean_ious) / len(clean_ious) if clean_ious else 0

# ========== TIME STATISTICS ==========
min_time = min(prediction_times)
max_time = max(prediction_times)
avg_time = sum(prediction_times) / len(prediction_times)

# ========== PRINT RESULTS ==========
print("\n====================== MODEL SUMMARY ======================")
print(f"Number of test images: {len(prediction_times)}")
print(f"Prediction time per image (seconds):")
print(f"Min: {min_time:.4f}s, Max: {max_time:.4f}s, Avg: {avg_time:.4f}s")
print(f"Total prediction time for {len(prediction_times)} images: {sum(prediction_times):.4f}s")
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print(f"Average IoU: {avg_iou:.4f}")
print("\n==================== CLASSIFICATION REPORT ====================")
print(classification_report_str)