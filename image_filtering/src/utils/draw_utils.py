#%%
import tensorflow as tf
import matplotlib.pyplot as plt

#%%
def array_to_img(img: tf.Tensor) -> None:
    """
    draw tensor img

    Args:
        img (tf.Tensor): tensor array img
    """
    img = tf.keras.preprocessing.image.array_to_img(img)

    return img
