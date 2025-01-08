import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# def load_video(video_path, img_size=(180, 180)):

#     cap = cv2.VideoCapture(video_path) #ucitavanje videa
#     frames = []
    
#     while(cap.isOpened()):
#         ret, frame = cap.read() #ret true dok se video ne zavrsi, frame trenutnni okvir videa
#         if not ret:
#             break
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
#         frame_resized = cv2.resize(frame, img_size)  # Resize the frame, sve u (180,180)
#         frames.append(frame_resized)

#     cap.release() # sadrzi sve frejmove tj slike
#     return np.array(frames)

# # Example usage
# #video_frames = load_video('Data/video/classification_video.mp4')

# #plt.imshow(video_frames[50])  # Prikazivanje  frejma
# #plt.axis('off')  # Isključivanje osa
# #plt.show()


def detect_object_on_table(video_path, img_size=(180, 180), start_time=3.0, selected_frames=[]):  
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Uzimanje broja frejmova u sekundi
    frames_with_object = []
    
    prev_frame = None  # Prethodni frejm koji ćemo koristiti za poređenje

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Dobijanje trenutnog vremena u sekundi
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        
        if current_time < start_time:
            continue

        # Dodaj logiku za selektovanje frejmova na osnovu vremena ili drugih uslova
        # Na primer, koristi frejmove koji se nalaze u određenim vremenskim intervalima
        if selected_frames and current_time not in selected_frames:
            continue  # Preskoči frejm ako trenutni vreme nije u selektovanom intervalu

        # Konvertovanje frejma u sivu sliku (gray scale)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Ako je prethodni frejm None, inicijalizuj ga
        if prev_frame is None:
            prev_frame = gray_frame
            continue
        
        # Izračunavanje razlike između trenutnog i prethodnog frejma
        frame_diff = cv2.absdiff(prev_frame, gray_frame)
        
        # Primenjujemo pra threshold za detekciju promena
        _, thresh = cv2.threshold(frame_diff, 50, 255, cv2.THRESH_BINARY)
        
        # Pronaći konture promena
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Podesi minimalnu veličinu promena
                # Kada je promena dovoljno velika, to znači da je predmet verovatno postavljen
                frame_resized = cv2.resize(frame, img_size)
                frames_with_object.append(frame_resized)
                break  # Uzimamo samo prvi frejm gde je promena detektovana

        # Ažuriraj prethodni frejm
        prev_frame = gray_frame

    cap.release()
    return np.array(frames_with_object)

# Primer upotrebe
# Definiši koje vremenske tačke želiš da koristiš (u sekundama)
selected_frames = [05.0,12.0, 20.0, 29.0, 37.0]  # Na primer, samo frejmovi u 5s, 10s, i 15s

video_frames_with_object = detect_object_on_table('Data/video/classification_video.mp4', start_time=4.0, selected_frames=selected_frames)

def display_all_selected_frames(frames):
    for i, frame in enumerate(frames):
        plt.imshow(frame)
        plt.axis('off')
        plt.title(f"Frame {i+1}")
        plt.show()  # Prikazivanje frejmova
        
        # Pauza između frejmova (na primer 1 sekunda)
        cv2.waitKey(1000)  # Pauza u milisekundama (1000 ms = 1 sekunda)

# Pretpostavimo da je 'video_frames_with_object' lista frejmova koje si izabrala
if len(video_frames_with_object) > 0:
    display_all_selected_frames(video_frames_with_object)
else:
    print("No frames with objects found.")