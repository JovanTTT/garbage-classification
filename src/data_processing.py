# import tensorflow as tf
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt 


# from tensorflow import keras
# from tensorflow.keras import layers


# # data_train_path = 'Data/train'
# # data_test_path = 'Data/test'
# # data_val_path = 'Data/val'

# def process_data(data_train_path, data_test_path, data_val_path):
#     img_width = 180
#     img_height = 180

#     data_train = tf.keras.utils.image_dataset_from_directory( #ucitava slike u train, 1672
#     data_train_path,
#     shuffle = True,
#     image_size = (img_width, img_height),
#     batch_size = 32,
#     validation_split = False
#     )

#     data_val = tf.keras.utils.image_dataset_from_directory(#358
#     data_val_path,
#     image_size = (img_height, img_width),
#     batch_size = 32,
#     shuffle = False,
#     validation_split = False
#     )

#     data_test = tf.keras.utils.image_dataset_from_directory(#360
#     data_test_path,
#     image_size = (img_width, img_height),
#     shuffle = False,
#     batch_size = 32,
#     validation_split = False
#     )

#     data_cat = data_train.class_names

#     return data_train, data_val, data_test, data_cat



# def process_data_scaled(data_train_path, data_val_path, data_test_path):
#     """Processes and scales the data."""
#     img_width = 180
#     img_height = 180

#     data_train = tf.keras.utils.image_dataset_from_directory(
#         data_train_path,
#         shuffle=True,
#         image_size=(img_width, img_height),
#         batch_size=32
#     )

#     data_val = tf.keras.utils.image_dataset_from_directory(
#         data_val_path,
#         shuffle=False,
#         image_size=(img_width, img_height),
#         batch_size=32
#     )

#     data_test = tf.keras.utils.image_dataset_from_directory(
#         data_test_path,
#         shuffle=False,
#         image_size=(img_width, img_height),
#         batch_size=32
#     )

#     data_cat = data_train.class_names

#     # Apply normalization (scaling)
#     data_train_scaled = data_train.map(lambda x, y: (x / 255.0, y))
#     data_val_scaled = data_val.map(lambda x, y: (x / 255.0, y))
#     data_test_scaled = data_test.map(lambda x, y: (x / 255.0, y))

#     return data_train_scaled, data_val_scaled, data_test_scaled, data_cat


import tensorflow as tf
import matplotlib.pyplot as plt 

def process_data(data_train_path, data_test_path, data_val_path):
    img_width = 180
    img_height = 180

    data_train = tf.keras.utils.image_dataset_from_directory(
        data_train_path,
        shuffle=True,
        image_size=(img_width, img_height),
        batch_size=32,
        validation_split=False
    )

    data_val = tf.keras.utils.image_dataset_from_directory(
        data_val_path,
        image_size=(img_width, img_height),
        batch_size=32,
        shuffle=False,
        validation_split=False
    )

    data_test = tf.keras.utils.image_dataset_from_directory(
        data_test_path,
        image_size=(img_width, img_height),
        shuffle=False,
        batch_size=32,
        validation_split=False
    )

    # Analyze class distribution
    class_names = data_train.class_names

    # Normalize datasets
    data_train = data_train.map(lambda x, y: (x / 255.0, y))
    data_val = data_val.map(lambda x, y: (x / 255.0, y))
    data_test = data_test.map(lambda x, y: (x / 255.0, y))


    # print(f"Class names: {class_names}")
    # for class_name in class_names:
    #      print(f"{class_name}: {sum([labels.numpy().tolist().count(class_name) for _, labels in data_train])}")

    def count_images_in_dataset(dataset):
     count = 0
     for images, labels in dataset:
        count += images.shape[0]  # The number of images in the current batch
     return count

    # Total number of images in training, validation and test set
    print("-----------------------------------------------------------------")
    train_images_count = count_images_in_dataset(data_train)
    print(f"TRAINING DATA: {train_images_count}")

    val_images_count = count_images_in_dataset(data_val)
    print(f"VALIDATION DATA: {val_images_count}")

    test_images_count = count_images_in_dataset(data_test)
    print(f"TEST DATA: {test_images_count}")


    print("-----------------------------------------------------------------")
    print(f"CLASS NAMES: {class_names}")
    
    class_counts = {class_name: 0 for class_name in class_names}  # Initialization of counters for each class
    
    # Iterating through the dataset to count images by class, FOR TRAINING DATA
    for images, labels in data_train:
        for label in labels.numpy():  # Converting a tensor to a numpy array
            class_counts[class_names[label]] += 1
    
    print("-----------------------------------------------------------------")
    print("NUMBER OF IMAGES FOR TRAINING DATA (BY CLASSES)")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count}")

     # Funkcija za prikazivanje uzoraka slika
    def show_sample_images(data_train):
        fig, axes = plt.subplots(1, len(class_names), figsize=(15, 15))

        for i, class_name in enumerate(class_names):
            # Dohvati slike iz trening skupa po klasama
            class_images = [image for image, label in data_train.unbatch() if class_names[label.numpy()] == class_name]
        
            # Prikazivanje samo jedne slike iz te klase
            ax = axes[i]
            ax.imshow(class_images[0])  # Prikazivanje prve slike iz klase
            ax.set_title(f"{class_name}")
            ax.axis('off')  # Isključivanje prikaza osa

       # plt.show()

    # Pozivanje funkcije za prikazivanje uzoraka slika
    show_sample_images(data_train)

    return data_train, data_val, data_test, class_names
