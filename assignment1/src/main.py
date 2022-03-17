#%%
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from utils import (
    load_data, 
    preprocessing, 
    draw_array,
    gaussian_filter,
    mean_filter,
    sharpen_filter,
    gray_filter,
    thresholding_filter,
    highpass_filter,
)
#%%
if __name__ == "__main__":
    imgs = load_data()
    img0, img1, img2, img3 = preprocessing(imgs)

    filter0 = gaussian_filter(img0, 7, 30.)
    draw_array(filter0)

    filter1 = gaussian_filter(img1, 7, 30.)
    draw_array(filter1)

    filter1_detail = highpass_filter(img1, filter1)
    draw_array(filter1_detail)
    
    draw_array(filter0 + filter1_detail)

    filter2 = gaussian_filter(img2, 7, 30.)
    draw_array(filter2)

    filter3 = gaussian_filter(img3, 7, 30.)
    draw_array(filter3)

    filter3_detail = highpass_filter(img3, filter3)
    draw_array(filter3_detail)

    draw_array(filter2 + filter3_detail)

    draw_array(tfa.image.mean_filter2d(img0, filter_shape=(7,7)))
    



#%%
draw_array(tfa.image.sharpness(img0, 3.))
draw_array(tf.where(img0 > 127.5, 255., 0.))
gray_img0 = tf.expand_dims(tf.reduce_sum(img0, axis=-1) / 3, axis=-1)
draw_array(gray_img0)

gray_img0_ = tf.keras.preprocessing.image.array_to_img(gray_img0)
threshold_filters = tf.where(gray_img0 > 127.5, 255., 0.)
tf.keras.preprocessing.image.array_to_img(threshold_filters)

