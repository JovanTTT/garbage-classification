import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def predict_image(image_path, model_path, class_names):
    # Load the model
    model = load_model(model_path)

    # Load and preprocess the image
    img_height, img_width = 180, 180  # Image size should match the model's training input
    img = load_img(image_path, target_size=(img_height, img_width))
    img_array = img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch (batch size = 1)

    # Predict the class
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]  # Get the index of the highest probability
    predicted_class = class_names[predicted_class_index]
    confidence_score = predictions[0][predicted_class_index]  # Get the confidence score of the predicted class

    return predicted_class, confidence_score

# Example usage:
if __name__ == "__main__":
    # Replace with the path to your image, model, and class names
    image_path = "Data/test_pictures/karton.jpg"
    model_path = "models/efficientnet_nonscaled.h5"
    class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic']  # Replace with your class names

    predicted_class, confidence_score = predict_image(image_path, model_path, class_names)
    print(f"The predicted class for the image is: {predicted_class}")
    print(f"Confidence score: {confidence_score:.2f}")





