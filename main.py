from train_model.train_model2.train_model2 import (build_efficientnet, train_model)
from src.data_processing import (process_data, process_data_scaled)

# Preprocess data
train_data, val_data, test_data, class_names = process_data("Data/train", "Data/val", "Data/test")
    
# Build models
models = {
    "EfficientNet": build_efficientnet(),
}
    
# Train and save each model
for model_name, model in models.items():
    print(f"Training {model_name}...")
    trained_model, history = train_model(model, train_data, val_data, epochs=10)
    trained_model.save(f"models/{model_name}.h5")
    print(f"{model_name} saved to models/{model_name}.h5")