import tensorflow as tf
from tensorflow.keras import Sequential, layers, optimizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping

IMG_SHAPE = (180, 180, 3)
NUM_CLASSES = 5

# Build model with fine-tuning
def build_efficientnet(input_shape=IMG_SHAPE, num_classes=NUM_CLASSES):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = True  # Fine-tune the entire base model

    model = Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=optimizers.AdamW(learning_rate=1e-4, clipnorm=1.0),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# Function to train a model
def train_model(model, train_data, val_data, epochs=20, class_weights=None):
    """Trains the model and returns the trained model and history."""
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=[early_stopping],
        class_weight=class_weights
    )
    return model, history
