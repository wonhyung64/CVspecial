#%%
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

#%%
ds_train = tfds.load(
    "imagenet2012",
    data_dir="D:/won/data/tfds",
    split="train",
    shuffle_files=False,
    download=True,
    as_supervised=True,
)
datasets = iter(ds_train)
img, label = next(datasets)
tf.keras.preprocessing.image.array_to_img(img)

img1 = tf.keras.preprocessing.image.array_to_img(img1)
img2 = tf.keras.preprocessing.image.array_to_img(img2)
img3 = tf.keras.preprocessing.image.array_to_img(img3)
img4 = tf.keras.preprocessing.image.array_to_img(img6)

import os
os.getcwd()
os.mkdir(f"{os.getcwd()}/ex")
save_dir = f"{os.getcwd()}/ex"

img4.save(f"{save_dir}/img4.png", "png")
os.listdir(save_dir)


tf.keras.preprocessing.image.array_to_img(
    tfa.image.mean_filter2d(img2, constant_values=4) - img2
)
tf.keras.preprocessing.image.array_to_img(
    tfa.image.gaussian_filter2d(img2, sigma=4.0) - img2
)
tf.keras.preprocessing.image.array_to_img(img2 - img2)
