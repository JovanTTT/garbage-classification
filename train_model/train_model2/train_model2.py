import tensorflow as tf
from tensorflow.keras import Sequential, layers, optimizers
from tensorflow.keras.applications import EfficientNetB0, ResNet50, MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping

IMG_SHAPE = (180, 180, 3) 
NUM_CLASSES = 5  

# 2. EfficientNet
def build_efficientnet(input_shape=IMG_SHAPE, num_classes=NUM_CLASSES):
    """Builds and returns an EfficientNetB0 model."""
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze base model
    model = Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Function to train a model
def train_model(model, train_data, val_data, epochs=10):
    """Trains the model and returns the trained model and history."""
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(train_data, validation_data=val_data, epochs=epochs, callbacks=[early_stopping])
    return model, history