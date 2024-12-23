import tensorflow as tf
from tensorflow.keras.models import load_model

def evaluate_model(model_path, test_data):
    """Load a trained model and evaluate it on the test data."""
    model = load_model(model_path)

    # Evaluate the model
    loss, accuracy = model.evaluate(test_data)
    
    return loss, accuracy

# Example usage
if __name__ == "__main__":
    model_path = "models/cnn.h5"  # Specify path to the saved model
    
    # Load and preprocess test data
    test_data = tf.keras.utils.image_dataset_from_directory(
        "Data/test", 
        image_size=(180, 180),
        batch_size=32,
        shuffle=False
    )
    
    # Evaluate the model
    loss, accuracy = evaluate_model(model_path, test_data)
    
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")