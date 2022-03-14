#%%
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from utils import (
    load_data,
    preprocessing,
    draw_array
)

#%%
if __name__ == "__main__":
    imgs = load_data()
    img0, img1, img2, img3 = preprocessing(imgs)
    
    draw_array(img0)
    draw_array(img1)
    draw_array(img2)
    draw_array(img3)



#%%


tf.keras.preprocessing.image.array_to_img(
    tfa.image.mean_filter2d(img2, constant_values=4) - img2
)
tf.keras.preprocessing.image.array_to_img(
    tfa.image.gaussian_filter2d(img2, sigma=4.0) - img2
)
tf.keras.preprocessing.image.array_to_img(img2 - img2)
