import streamlit as st
import cv2
import numpy as np
import time
from datetime import datetime

# Load custom JavaScript
def load_custom_js():
    with open("static/script.js", "r") as js_file:
        js_code = js_file.read()
        st.markdown(f'<script>{js_code}</script>', unsafe_allow_html=True)

# Load custom CSS
def load_custom_css():
    with open("static/styles.css", "r") as css_file:
        css_code = css_file.read()
        st.markdown(f'<style>{css_code}</style>', unsafe_allow_html=True)

# Main Streamlit app
def main():
    st.title("My Streamlit App")

    # Load custom JavaScript and CSS
    load_custom_js()
    load_custom_css()


# Load the YOLO model and classes
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

# The object the system will detect for now, it is person, cat, dog , sheep, cow ,bottel, chair
class_ids_to_detect = [0, 15, 16, 18, 19, 39, 56, 63, 67] 

# Function to perform real-time detection
def detect_objects(frame):
    # Preprocess the frame
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Get output layer names
    layer_names = net.getUnconnectedOutLayersNames()

    # Forward pass (check the correctness of model and configuration paths)
    try:
        outputs = net.forward(layer_names)
    except Exception as e:
        print("Error during forward pass:", e)
        return []

    # Forward pass
    outputs = net.forward(layer_names)

    # Process the detection results
    detected_objects = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and class_id in class_ids_to_detect:
                label = str(classes[class_id])
                detected_objects.append(label)

                # Draw bounding boxes on the frame
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Define colors for different classes (modify as needed)
                colors = {"person": (0, 255, 0), "cow": (255, 0, 0), "cat": (0, 0, 255), "dog": (255, 255, 0), "laptop": (0, 255, 255), "chair": (255, 0, 255), "bottel": (128, 0, 128)}

                # Draw bounding boxes with class-specific colors
                color = colors.get(label, (0, 0, 0))  # Default to black if class not found
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return detected_objects

st.title("Stray Cattel Detection")

# Create a streamlit element for displaying the webcam feed
video_feed = st.empty()

# Streamlit app loop
cap = cv2.VideoCapture(0)  # Use the laptop's webcam as the video source
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Object detection
    detected_objects = detect_objects(frame)

    # Display the frame with detected objects
    frame_with_boxes = frame  # We've already drawn the boxes on the frame
    video_feed.image(frame_with_boxes, channels="BGR")

    # Display current object detected
    if detected_objects:
        current_object = detected_objects[0]
    else:
        current_object = "No object detected"

    st.write("Current Object Detected:", current_object)
    st.write("Time:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    # Save data to a file (update_data/data.txt)
    with open("update_data/data.txt", "a") as data_file:

        data_file.write(f"Object: {current_object}, Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Sleep for a while to control the frame rate
    time.sleep(0.1)
