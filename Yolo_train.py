from ultralytics import YOLO

# Load the YOLO model
model = YOLO('yolov8n.pt')  # Use 'yolov8n.pt', 'yolov8s.pt', etc., based on your requirements

# Train the model
model.train(data='fdd.yaml',  # Path to the YAML file
            epochs=50,                 # Number of epochs (adjust as needed)
            imgsz=640,                 # Image size
            batch=16,                  # Batch size (adjust based on available GPU memory)
            workers=8,                 # Number of data loader workers
            project='/model/FDD_yolo_training',  # Directory to save results
            name='yolov8_fdd_model')   # Name of the experiment
