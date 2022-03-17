#%%
import tensorflow as tf
import tensorflow_addons as tfa

#%%
def gray_filter(img: tf.Tensor) -> tf.Tensor:
    """
    gray_scale filtering

    Args:
        img (tf.Tensor): RGB_scale img

    Returns:
        tf.Tensor: gray_scale img
    """
    img = tf.reduce_sum(img, axis=-1) / 3.0
    img = tf.expand_dims(img, axis=-1)

    return img


def thresholding_filter(img: tf.Tensor, threshold: float = 127.5) -> tf.Tensor:
    """
    thresholding filtering

    Args:
        img (tf.Tensor): img
        threshold (float, optional): assign 255. bigger than threshold o.w. 0. Defaults to 127.5.

    Returns:
        tf.Tensor: thresholding filterd img
    """
    img = tf.where(img > threshold, 255.0, 0.0)

    return img


def sharpen_filter(img: tf.Tensor, factor: float) -> tf.Tensor:
    """
    sharpen filtering

    Args:
        img (tf.Tensor): img
        factor (float): alpha

    Returns:
        tf.Tensor: sharpened img
    """
    img = tfa.image.sharpness(img, factor)

    return img


def gaussian_filter(img: tf.Tensor, filter_size: int, sigma: float) -> tf.Tensor:
    """
    gaurssian filtering

    Args:
        img (tf.Tensor): img
        filter_size (int): shape of filter
        sigma (float): sigma of gaussian

    Returns:
        tf.Tensor: gaussian filterd img
    """
    img = tfa.image.gaussian_filter2d(
        img, filter_shape=(filter_size, filter_size), sigma=sigma
    )

    return img


def mean_filter(img: tf.Tensor, filter_size: int) -> tf.Tensor:
    """
    mean filtering

    Args:
        img (tf.Tensor): img
        filter_size (int): shape of filter

    Returns:
        tf.Tensor: mean filterd img
    """
    img = tfa.image.mean_filter2d(img, filter_shape=(filter_size, filter_size))

    return img


def highpass_filter(img: tf.Tensor, smoothed_img: tf.Tensor) -> tf.Tensor:
    """
    high-pass filtering

    Args:
        img (tf.Tensor): original img
        smoothed_img (tf.Tensor): smoothed img

    Returns:
        tf.Tensor: high-pass filterd img
    """

    return img - smoothed_img
