import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


from tensorflow import keras
from tensorflow.keras import layers


data_train_path = 'Data/train'
data_test_path = 'Data/test'
data_val_path = 'Data/val'

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

data_cat = data_train.class_names #imena kategorija u data_cat


# data_iterator = data_train.as_numpy_iterator()
# batch = data_iterator.next()
# fig, ax = plt.subplots(ncols=4, figsize=(20,20))
# for idx, img in enumerate(batch[0][:4]):
#      ax[idx].imshow(img.astype(int))
#      ax[idx].title.set_text(batch[1][idx])
#      plt.show()



print(f"Broj klasa: {len(data_cat)}")
print(f"Imena klasa: {data_cat}")

# Ukupan broj slika u trening skupu
train_images_count = len(data_train.file_paths)
print(f"Trening podaci: {train_images_count}")

# Ukupan broj slika u validacionom skupu
val_images_count = len(data_val.file_paths)
print(f"Validacija podaci: {val_images_count}")

# Ukupan broj slika u test skupu
test_images_count = len(data_test.file_paths)
print(f"Test podaci: {test_images_count}")

#Ovo se koristi kao novi dataset
data_test_scale = data_test.map(lambda x,y: (x/255, y)) #vrednosti piksela se skaliraju od 0 do 1, bolje funkcionisu kad su podaci normalizovani
data_test_scale.as_numpy_iterator().next()

data_train_scale = data_train.map(lambda x,y: (x/255, y))
data_train_scale.as_numpy_iterator().next()

data_val_scale = data_val.map(lambda x,y: (x/255, y))
data_val_scale.as_numpy_iterator().next()
