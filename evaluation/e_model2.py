import tensorflow as tf
from tensorflow.keras.models import load_model

def evaluate_model(model_path, test_data):
    # Load the trained model
    model = load_model(model_path)

    # Evaluate the model on the test data
    loss, accuracy = model.evaluate(test_data)

    return loss, accuracy

# Example usage:
if __name__ == "__main__":
    # Replace with the actual path to your model and test dataset
    model_path = "models/EfficientNet.h5"
    
    # Load your test data (ensure you have it preprocessed like your training data)
    img_height, img_width = 180, 180
    test_data = tf.keras.utils.image_dataset_from_directory(
        "Data/test", 
        image_size=(img_height, img_width),
        batch_size=32,
        shuffle=False
    )

    # Evaluate the model
    loss, accuracy = evaluate_model(model_path, test_data)
    
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
