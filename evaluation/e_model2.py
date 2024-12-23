import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

def display_predictions(test_data, model_path, class_names, num_images=5):
    # Load the trained model
    model = load_model(model_path)

    # Create a list of all images and labels in the test dataset
    all_images, all_labels = [], []
    for images, labels in test_data:
        all_images.append(images)
        all_labels.append(labels)
    
    all_images = np.concatenate(all_images)
    all_labels = np.concatenate(all_labels)
    
    # Randomly select indices for displaying
    random_indices = np.random.choice(len(all_images), size=num_images, replace=False)
    selected_images = all_images[random_indices]
    true_labels = all_labels[random_indices]
    
    # Predict the classes for the selected images
    predictions = model.predict(selected_images)
    predicted_classes = np.argmax(predictions, axis=1)
    confidence_scores = np.max(predictions, axis=1)  # Get confidence scores
    
    # Plot the selected images with predictions and accuracy
    plt.figure(figsize=(15, 10))
    plt.suptitle("Some Examples", fontsize=16)  # Add a title to the whole figure
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(selected_images[i].astype("uint8"))
        plt.axis("off")
        plt.title(
            f"True: {class_names[true_labels[i]]}\nPred: {class_names[predicted_classes[i]]}\nAccuracy: {confidence_scores[i]*100:.2f}%",
            fontsize=10
        )
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title
    plt.show()

# Example usage
if __name__ == "__main__":
    model_path = "models/efficientnet_nonscaled.h5"
    class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic']
    
    img_height, img_width = 180, 180
    test_data = tf.keras.utils.image_dataset_from_directory(
        "Data/test",
        image_size=(img_height, img_width),
        batch_size=32,
        shuffle=False
    )
    
    display_predictions(test_data, model_path, class_names, num_images=5)
