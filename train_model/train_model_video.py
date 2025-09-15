import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.video_processing import detect_object_on_table

def preprocess_frame(frame, img_size=(180, 180)):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
    frame_resized = cv2.resize(frame, img_size)  # Resize to model input size
    frame_normalized = frame_resized / 255.0  # Normalize pixel values
    return np.expand_dims(frame_normalized, axis=0)  # Add batch dimension

def classify_detected_frames(model, video_path, selected_frames, class_names, img_size=(180, 180)):
    video_frames_with_object = detect_object_on_table(video_path, start_time=4.0, selected_frames=selected_frames)
    
    if len(video_frames_with_object) > 0:
        print(f"Found {len(video_frames_with_object)} frames with objects for classification.")
        for i, frame in enumerate(video_frames_with_object):
            preprocessed_frame = preprocess_frame(frame, img_size)
            predictions = model.predict(preprocessed_frame)
            predicted_class = np.argmax(predictions)
            predicted_class_name = class_names[predicted_class]
            print(f"Frame {i+1} predicted as class: {predicted_class_name}")
            
            plt.imshow(frame)
            plt.axis('off')
            plt.title(f"Frame {i+1}, Predicted class: {predicted_class_name}")
            plt.show()
    else:
        print("No frames with objects found.")

def create_annotated_video(video_path, model, class_names, selected_frames, output_path="annotated_video.mp4", img_size=(180, 180)):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    last_annotation_time = -2.0
    last_class_name = None
    selected_frame_times = {round(f, 2) for f in selected_frames}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, 2)
        if current_time in selected_frame_times:
            preprocessed_frame = preprocess_frame(frame, img_size)
            predictions = model.predict(preprocessed_frame)
            predicted_class = np.argmax(predictions)
            last_class_name = class_names[predicted_class]
            last_annotation_time = current_time

        if current_time - last_annotation_time <= 2.0 and last_class_name is not None:
            cv2.rectangle(frame, (50, 50), (frame_width - 50, frame_height - 50), (0, 255, 0), 2)
            cv2.putText(frame, last_class_name, (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Annotated video saved to {output_path}")