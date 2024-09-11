import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

model=YOLO('yolov8n.pt')

def detect_objects(image,thres=0.16):
    img=np.array(image)
    img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    results=model(img,conf=thres)
    detections=results[0].boxes
    count=len(detections)
    class_names=results[0].names
    detected_classes=[class_names[int(box.cls)] for box in detections]
    return count
st.title("Object Detector")
file=st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
if file is not None:
    image=Image.open(file)
    count=detect_objects(image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write('Number of objects detected:',count)
