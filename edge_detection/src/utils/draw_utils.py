import numpy as np
import matplotlib.pyplot as plt
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