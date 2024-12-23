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
from train_model.train_model2.train_model2 import build_efficientnet, train_model
from train_model.train_model1 import build_cnn, train_cnn_model

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


#Define model type and initialize the CNN model
model_type = 'cnn'
model = build_cnn()

# Train the model
trained_model, history = train_cnn_model(model, train_data, val_data, epochs=20, class_weights=class_weights)

# Save the trained model
trained_model.save(f"models/{model_type}.h5")
print(f"Model saved to models/{model_type}.h5")

# Evaluate the model on the test set
test_loss, test_accuracy = trained_model.evaluate(test_data)
print(f"Test accuracy: {test_accuracy}")
