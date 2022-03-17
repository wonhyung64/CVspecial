#%%
import tensorflow as tf
import matplotlib.pyplot as plt
import PIL

#%%
def array_to_img(img: tf.Tensor) -> None:
    """
    draw tensor img

    Args:
        img (tf.Tensor): tensor array img
    """
    img = tf.keras.preprocessing.image.array_to_img(img)

    return img


def draw_outputs(imgs: PIL.Image.Image, cmap: str or None) -> None:
    """
    draw imgs

    Args:
        imgs (PIL.Image.Image): result imgs
        cmap (strorNone): color map option
    """
    plt.rcParams["figure.figsize"] = (20.0, 80.0)
    for i in range(len(imgs)):
        plt.subplot(1, 4, i + 1)
        plt.axis("off")
        plt.imshow(imgs[i], cmap=cmap)
    plt.show()
