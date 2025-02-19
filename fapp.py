from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import base64
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# Load the trained YOLO model
model = YOLO('/model/FDD_yolo_training/yolov8_fdd_model/weights/best.pt')  # Adjust the path

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    np_image = np.frombuffer(image_file.read(), np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    # Run inference
    results = model(image)
    print(results)
    # Annotate the image
    annotated_image = results[0].plot()

    # Encode the annotated image as Base64
    _, buffer = cv2.imencode('.jpg', annotated_image)
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'annotated_image': encoded_image})

if __name__ == '__main__':
    app.run(debug=True)
