import tensorflow as tf
from tensorflow.keras.models import load_model
import os


def evaluate_model(model_path, test_data):
    """Load a trained model and evaluate it on the test data."""
    model = load_model(model_path)

    # Evaluate the model
    loss, accuracy = model.evaluate(test_data)

    return loss, accuracy


# Example usage
if __name__ == "__main__":

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    model_path = os.path.join(base_dir, "models", "cct_1.h5")

    test_path = os.path.join(base_dir, "Data", "test")

    # Load and preprocess test data
    test_data = tf.keras.utils.image_dataset_from_directory(
        test_path,
        image_size=(180, 180),
        batch_size=32,
        shuffle=False
    )

    # Evaluate the model
    loss, accuracy = evaluate_model(model_path, test_data)

    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")