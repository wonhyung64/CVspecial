#%%
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import matplotlib
from utils import load_data, preprocessing, draw_array

#%%
if __name__ == "__main__":
    imgs = load_data()
    img0, img1, img2, img3 = preprocessing(imgs)
    draw_array(img0)
    draw_array(img1)

    filter0 = tfa.image.gaussian_filter2d(img0, filter_shape=(7, 7), sigma=30.0)
    filter0 = tf.cast(filter0, dtype=tf.float32)
    draw_array(filter0)

    filter1 = tfa.image.gaussian_filter2d(img1, filter_shape=(7, 7), sigma=20.0) - img1
    filter1 = tf.cast(filter1, dtype=tf.float32)
    draw_array(filter1)

    draw_array(filter0 + filter1)

    filter2 = tfa.image.gaussian_filter2d(img2, filter_shape=(7, 7), sigma=30.0)
    filter2 = tf.cast(filter2, dtype=tf.float32)
    draw_array(filter2)

    filter3 = tfa.image.gaussian_filter2d(img3, filter_shape=(7, 7), sigma=30.0) - img3
    filter3 = tf.cast(filter3, dtype=tf.float32)
    draw_array(filter3)

    draw_array(filter2 + filter3)


#%%
