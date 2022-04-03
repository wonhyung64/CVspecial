import numpy as np
import cv2
from typing import Union


def sobel_edge(img: np.ndarray, grad: str, kernel_size: int) -> np.ndarray:
    """
    sobel edge detector

    Args:
        img (np.ndarray): img
        grad (str): direction to calculate gradient
        kernel_size (int): kernel size

    Returns:
        np.ndarray: sobel edge detected img
    """
    if grad == "x":
        dx, dy = 1, 0
    elif grad == "y":
        dx, dy = 0, 1
    elif grad == "xy":
        dx, dy = 1, 1
    img = cv2.Sobel(src=img, ddepth=-1, dx=dx, dy=dy, ksize=kernel_size)

    return img


def canny_edge(
    img: np.ndarray, threshold1: Union[int, float], threshold2: int or float,
) -> np.ndarray:
    """
    canny edge detector

    Args:
        img (np.ndarray): img
        threshold1 (intorfloat): threshold 1
        threshold2 (intorfloat): threshold 2

    Returns:
        np.ndarray: canny edge detected img
    """
    img = cv2.Canny(image=img, threshold1=threshold1, threshold2=threshold2)

    return img
