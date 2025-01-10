# from train_model.train_model2.train_model2 import (build_efficientnet, train_model)
# from src.data_processing import (process_data, process_data_scaled)

# # Preprocess data
# train_data, val_data, test_data, class_names = process_data("Data/train", "Data/val", "Data/test")
    
# # Build models
# models = {
#     "EfficientNet": build_efficientnet(),
# }
    
# # Train and save each model
# for model_name, model in models.items():
#     print(f"Training {model_name}...")
#     trained_model, history = train_model(model, train_data, val_data, epochs=10)
#     trained_model.save(f"models/{model_name}.h5")
#     print(f"{model_name} saved to models/{model_name}.h5")

from src.data_processing import process_data
from src.data_processing import show_sample_images_with_predictions
from train_model.train_model2.train_model2 import build_efficientnet, train_model
from train_model.train_model1 import build_cnn, train_cnn_model
from src.video_processing import detect_object_on_table
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt




def preprocess_frame(frame, img_size=(180, 180)):
    frame_resized = cv2.resize(frame, img_size)  # Promeni dimenzije slike
    frame_resized = np.expand_dims(frame_resized, axis=0)  # Dodaj dimenziju za batch
    frame_resized = frame_resized / 255.0  # Normalizuj slike (ako je potrebno)
    return frame_resized

# Pozivanje funkcije za detekciju objekta i klasifikaciju
def classify_detected_frames(model, video_path, selected_frames, img_size=(180, 180)):
    video_frames_with_object = detect_object_on_table(video_path, start_time=4.0, selected_frames=selected_frames)
    
    if len(video_frames_with_object) > 0:
        print(f"Found {len(video_frames_with_object)} frames with objects for classification.")
        for i, frame in enumerate(video_frames_with_object):
            preprocessed_frame = preprocess_frame(frame, img_size)
            # Klasifikacija frejma
            predictions = model.predict(preprocessed_frame)
            predicted_class = np.argmax(predictions)  # Predikcija klase
            predicted_class_name = class_names[predicted_class] 
            print(f"Frame {i+1} predicted as class: {predicted_class_name}")
            
            # Prikazivanje rezultata
            plt.imshow(frame)
            plt.axis('off')
            plt.title(f"Frame {i+1}, Predicted class: {predicted_class_name}")
            plt.show()
            
            # Pauza između frejmova (ako je potrebno)
            cv2.waitKey(1000)  # Pauza u milisekundama (1000 ms = 1 sekunda)
    else:
        print("No frames with objects found.")


def create_annotated_video(video_path, model, class_names, selected_frames, output_path="annotated_video.mp4", img_size=(180, 180)):

    # Otvaranje videa
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Inicijalizacija za zapisivanje izlaznog videa
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Priprema promenljivih
    last_annotation_time = -2.0  # Početna vrednost za poslednje anotacije, predictions
    last_class_name = None       # Početna vrednost za klasu poslednje anotacije
    selected_frame_times = {round(f, 2) for f in selected_frames}  # Pretvori u skup za bržu proveru

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Trenutno vreme u videu (sekunde)
        current_time = round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, 2)

        if current_time in selected_frame_times:
            # Preprocesiranje frejma
            preprocessed_frame = preprocess_frame(frame, img_size)
            # Predikcija klase
            predictions = model.predict(preprocessed_frame)
            predicted_class = np.argmax(predictions)
            last_class_name = class_names[predicted_class]
            last_annotation_time = current_time  # Ažuriraj vreme poslednje anotacije

        # Prikaz poslednje anotacije ako je unutar vremenskog okvira od 2 sekunde
        if current_time - last_annotation_time <= 2.0 and last_class_name is not None:
            cv2.rectangle(frame, (50, 50), (frame_width - 50, frame_height - 50), (0, 255, 0), 2)  # Placeholder 
            cv2.putText(frame, last_class_name, (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) #write class name

        # Dodaj frejm u izlazni video
        out.write(frame)

    # Zatvaranje
    cap.release()
    out.release()
    print(f"Annotated video saved to {output_path}")


# Preprocess data
train_data, val_data, test_data, class_names = process_data("Data/train", "Data/val", "Data/test")

#Define class weights (optional, based on dataset analysis)
#Klase s manjim brojem uzoraka dobijaju vece tezine da bi bilo jednako
class_weights = {0: 1.0, 1: 2.0, 2: 1.5, 3: 1.8, 4: 1.2}  # Adjust as needed


# model_type = 'efficientnet'  # Change to 'efficientnet' to use EfficientNet

# model = build_efficientnet()

# # Train the model
# trained_model, history = train_model(model, train_data, val_data, epochs=20, class_weights=class_weights)

# # Save the trained model
# trained_model.save(f"models/{model_type}.h5")
# print(f"Model saved to models/{model_type}.h5")

# # Evaluate the model on the test set
# test_loss, test_accuracy = trained_model.evaluate(test_data)
# print(f"Test accuracy: {test_accuracy}")


# CNN model
model_type = 'cnn'
model = build_cnn()

# Train the model
trained_model, history = train_cnn_model(model, train_data, val_data, epochs=15, class_weights=class_weights)

# Save the trained model
trained_model.save(f"models/{model_type}.h5")
print(f"Model saved to models/{model_type}.h5")

# Evaluate the model on the test set
test_loss, test_accuracy = trained_model.evaluate(test_data)
print(f"Test accuracy: {test_accuracy}")


show_sample_images_with_predictions(model, train_data, class_names)

# Video classification
selected_frames = [5.0, 12.0, 20.0, 29.0, 37.0]  # The time points you want to use

video_path = 'Data/video/classification_video.mp4'

# classification with images
classify_detected_frames(trained_model, video_path, selected_frames)

# clasification with new video
output_path = 'Data/video/annotated_classification_video.mp4'
create_annotated_video(video_path, trained_model, class_names, selected_frames, output_path)


