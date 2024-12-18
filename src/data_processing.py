# import pandas as pd
# import os
# import tensorflow as tf


# train_files = pd.read_csv('Data/one-indexed-files-notrash_train.txt', sep = ' ', header = None)[0].tolist()
# test_files = pd.read_csv('Data/one-indexed-files-notrash_test.txt', sep = ' ', header = None)[0].tolist()
# val_files = pd.read_csv('Data/one-indexed-files-notrash_val.txt', sep = ' ', header = None)[0].tolist()




# image_folder = 'Data/Garbage classification'

# # Funkcija koja proverava postojanje slike u folderu
# def find_image_in_folder(image_name, image_folder):
#     for file in os.listdir(image_folder):
#         if file == image_name:  # Proverava da li naziv slike odgovara fajlu u folderu
#             return os.path.join(image_folder, file)
#     return None

# # Pretraga slika iz train_files foldera
# train_image_paths = []
# for image_name in train_files:
#     image_path = find_image_in_folder(image_name, image_folder)
#     if image_path:
#         train_image_paths.append(image_path)

# print(train_image_paths)  # Ovo će ispisati sve putanje do slika u folderu