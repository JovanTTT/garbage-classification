import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def load_video(video_path, img_size=(180, 180)):

    cap = cv2.VideoCapture(video_path) #ucitavanje videa
    frames = []
    
    while(cap.isOpened()):
        ret, frame = cap.read() #ret true dok se video ne zavrsi, frame trenutnni okvir videa
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        frame_resized = cv2.resize(frame, img_size)  # Resize the frame, sve u (180,180)
        frames.append(frame_resized)

    cap.release() # sadrzi sve frejmove tj slike
    return np.array(frames)

# Example usage
#video_frames = load_video('Data/video/classification_video.mp4')

#plt.imshow(video_frames[50])  # Prikazivanje  frejma
#plt.axis('off')  # Isključivanje osa
#plt.show()