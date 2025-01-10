from src.data_processing import process_data
from train_model.train_model2.train_model2 import build_efficientnet, train_model

# Preprocess data
train_data, val_data, test_data, class_names = process_data("Data/train", "Data/val", "Data/test")

# Define class weights (optional, based on dataset analysis)
class_weights = {0: 1.0, 1: 2.0, 2: 1.5, 3: 1.8, 4: 1.2}  # Adjust as needed

model_type = 'efficientnet'  # Change to 'efficientnet' to use EfficientNet

model = build_efficientnet()

# Train the model
trained_model, history = train_model(model, train_data, val_data, epochs=20, class_weights=class_weights)

# Save the trained model
trained_model.save(f"models/{model_type}_nonscaled.h5")
print(f"Model saved to models/{model_type}.h5")

# Evaluate the model on the test set
test_loss, test_accuracy = trained_model.evaluate(test_data)
print(f"Test accuracy: {test_accuracy}")
