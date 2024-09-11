import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Load the YOLOv8 model (use the model path if you have a custom one)
model = YOLO('yolov8n.pt')

def detect_objects(image):
    # Convert PIL image to OpenCV format
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Perform inference
    results = model(img)
    
    # Extract detections
    detections = results[0].boxes
    count = len(detections)  # Number of detected objects
    
    # Extract class names
    class_names = results[0].names
    detected_classes = [class_names[int(box.cls)] for box in detections]
    
    return count, detected_classes

# Streamlit app
st.title("YOLOv8 Object Detection")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Open the image file
    image = Image.open(uploaded_file)
    
    # Detect objects
    object_count, detected_classes = detect_objects(image)
    
    # Display results
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write('Number of objects detected:', object_count)
    st.write('Detected objects:')
    for obj in detected_classes:
        st.write('- ' + obj)
