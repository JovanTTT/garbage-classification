from tensorflow.keras import layers, optimizers, Model # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

# CCT implementation
from tensorflow.keras import layers, optimizers, Model
from tensorflow.keras.models import Sequential

def build_cct(input_shape=(180, 180, 3), num_classes=5):
    # Data augmentation
    data_augmentation = Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    # Input
    inputs = layers.Input(shape=input_shape)
    x = data_augmentation(inputs)

    # Convolutional layers
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Regularization
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Dense (simulacija transformer dela)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)

    # Output
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    # Create and compile model
    model = Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Training function
def train_cct(model, train_data, val_data, epochs=20):
    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=[early_stopping]
    )
    return model, history
