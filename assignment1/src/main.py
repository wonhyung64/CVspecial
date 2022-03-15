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
import numpy as np
x_in = np.array([[
  [[2], [1], [2], [0], [1]],
  [[1], [3], [2], [2], [3]],
  [[1], [1], [3], [3], [0]],
  [[2], [2], [0], [1], [1]],
  [[0], [0], [3], [1], [2]], ]])
kernel_in = np.array([
 [ [[2, 0.1]], [[3, 0.2]] ],
 [ [[0, 0.3]],[[1, 0.4]] ], ])
x = tf.constant(x_in, dtype=tf.float32)
kernel = tf.constant(kernel_in, dtype=tf.float32)
tf.shape(x)
tmp = tf.expand_dims(img0, axis=0)
tf.shape(tmp)
tf.shape(kernel)
draw_array(img0)
draw_array(tfa.image.sharpness(img0, 3.))
draw_array(tf.where(img0 > 127.5, 255., 0.))
gray_img0 = tf.expand_dims(tf.reduce_sum(img0, axis=-1) / 3, axis=-1)
draw_array(gray_img0)
import matplotlib.pyplot as plt
gray_img0_ = tf.keras.preprocessing.image.array_to_img(gray_img0)
threshold_filters = tf.where(gray_img0 > 127.5, 255., 0.)
tf.keras.preprocessing.image.array_to_img(threshold_filters)
plt.imshow(gray_img0_)

