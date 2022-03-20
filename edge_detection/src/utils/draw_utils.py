import numpy as np
import PIL
from PIL import Image


def array_to_img(img: np.ndarray) -> PIL.Image:
    """
    convert ndarray to image

    Args:
        img (np.ndarray): img array

    Returns:
        PIL.Image: PIL img
    """
    img = Image.fromarray(img)

    return img
