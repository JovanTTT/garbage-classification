from src.data_processing import process_data_first
from src.data_processing import show_sample_images_with_predictions
from train_model.train_model2.train_model2 import build_efficientnet, train_model
from train_model.train_model1 import build_cnn, train_cnn_model
from tensorflow.keras.models import load_model
from train_model.train_model_video import classify_detected_frames, create_annotated_video



# Preprocess data
train_data, val_data, test_data, class_names = process_data_first("Data/train", "Data/val", "Data/test")

#Define class weights (optional, based on dataset analysis)
#Klase s manjim brojem uzoraka dobijaju vece tezine da bi bilo jednako
# class_weights = {0: 1.0, 1: 2.0, 2: 1.5, 3: 1.8, 4: 1.2}  # Adjust as needed
class_weights = {
    0: 1.184,  
    1: 0.955,  
    2: 1.165,  
    3: 0.803,  
    4: 0.993   
}




# model_type = 'efficientnet_novi'  # Change to 'efficientnet' to use EfficientNet

# model = build_efficientnet()

# # Train the model
# trained_model, history = train_model(model, train_data, val_data, epochs=10, class_weights=class_weights)

# # Save the trained model
# trained_model.save(f"models/{model_type}.h5")
# print(f"Model saved to models/{model_type}.h5")

# # Evaluate the model on the test set
# test_loss, test_accuracy = trained_model.evaluate(test_data)
# print(f"Test accuracy: {test_accuracy}")






# # CNN model
# model_type = 'cnn'
# model = build_cnn()

# # Train the model
# trained_model, history = train_cnn_model(model, train_data, val_data, epochs=15, class_weights=class_weights)

# # Save the trained model
# trained_model.save(f"models/{model_type}_nonscaled.h5")
# print(f"Model saved to models/{model_type}.h5")







model_path = "models/efficientnet_novi.h5"
trained_model = load_model(model_path)
print(f"Model loaded from {model_path}")

# Evaluate the model on the test set
test_loss, test_accuracy = trained_model.evaluate(test_data)
print(f"Test accuracy: {test_accuracy}")


show_sample_images_with_predictions(trained_model, train_data, class_names)

# Video classification
selected_frames = [5.0, 12.0, 20.0, 29.0, 37.0]  # The time points you want to use

video_path = 'Data/video/classification_video.mp4'

# classification with images
classify_detected_frames(trained_model, video_path, selected_frames, class_names)

# clasification with new video
output_path = 'Data/video/annotated_classification_video.mp4'
create_annotated_video(video_path, trained_model, class_names, selected_frames, output_path)


