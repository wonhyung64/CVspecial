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


def view_from_far(img: tf.Tensor, ratio: float = 0.5) -> tf.Tensor:
    """
    view img from far

    Args:
        img (tf.Tensor): img
        ratio (float, optional): ratio to reduce img

    Returns:
        tf.Tensor: reduced size img
    """
    img_size = tf.shape(img)[:2]
    far_img_size = tf.cast(tf.cast(img_size, dtype=tf.float32) * ratio, dtype=tf.int32)
    far_img = tf.image.resize(img, far_img_size)
    padding_shape = tf.cast(
        [
            (img_size - far_img_size) / 2,
            (img_size - far_img_size) / 2,
            tf.constant([0.0, 0.0], dtype=tf.float64),
        ],
        tf.int32,
    )
    far_img = tf.pad(far_img, padding_shape)

    return far_img
