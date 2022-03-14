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
tf.keras.preprocessing.image.array_to_img(img1)
tf.keras.preprocessing.image.array_to_img(img2)
tf.keras.preprocessing.image.array_to_img(img3)


tf.keras.preprocessing.image.array_to_img(
    tfa.image.mean_filter2d(img2, constant_values=4) - img2
)
tf.keras.preprocessing.image.array_to_img(
    tfa.image.gaussian_filter2d(img2, sigma=4.0) - img2
)
tf.keras.preprocessing.image.array_to_img(img2 - img2)
