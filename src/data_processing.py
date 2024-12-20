import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


from tensorflow import keras
from tensorflow.keras import layers


# data_train_path = 'Data/train'
# data_test_path = 'Data/test'
# data_val_path = 'Data/val'

def process_data(data_train_path, data_test_path, data_val_path):
    img_width = 180
    img_height = 180

    data_train = tf.keras.utils.image_dataset_from_directory( #ucitava slike u train, 1672
    data_train_path,
    shuffle = True,
    image_size = (img_width, img_height),
    batch_size = 32,
    validation_split = False
    )

    data_val = tf.keras.utils.image_dataset_from_directory(#358
    data_val_path,
    image_size = (img_height, img_width),
    batch_size = 32,
    shuffle = False,
    validation_split = False
    )

    data_test = tf.keras.utils.image_dataset_from_directory(#360
    data_test_path,
    image_size = (img_width, img_height),
    shuffle = False,
    batch_size = 32,
    validation_split = False
    )

    data_cat = data_train.class_names

    return data_train, data_val, data_test, data_cat



def process_data_scaled(data_train_path, data_val_path, data_test_path):
    """Processes and scales the data."""
    img_width = 180
    img_height = 180

    data_train = tf.keras.utils.image_dataset_from_directory(
        data_train_path,
        shuffle=True,
        image_size=(img_width, img_height),
        batch_size=32
    )

    data_val = tf.keras.utils.image_dataset_from_directory(
        data_val_path,
        shuffle=False,
        image_size=(img_width, img_height),
        batch_size=32
    )

    data_test = tf.keras.utils.image_dataset_from_directory(
        data_test_path,
        shuffle=False,
        image_size=(img_width, img_height),
        batch_size=32
    )

    data_cat = data_train.class_names

    # Apply normalization (scaling)
    data_train_scaled = data_train.map(lambda x, y: (x / 255.0, y))
    data_val_scaled = data_val.map(lambda x, y: (x / 255.0, y))
    data_test_scaled = data_test.map(lambda x, y: (x / 255.0, y))

    return data_train_scaled, data_val_scaled, data_test_scaled, data_cat


