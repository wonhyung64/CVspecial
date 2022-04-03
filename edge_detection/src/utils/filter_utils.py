from dataclasses import dataclass
import numpy as np
import cv2


def gray_filter(img: np.ndarray) -> np.ndarray:
    """
    convert RGB img array to gray scale img array

    Args:
        img (np.ndarray): RGB img

    Returns:
        np.ndarray: gray scale img
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img


def gaussian_filter(img: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    """
    gaussian filter

    Args:
        img (np.ndarray): img
        kernel_size (int): kernel size
        sigma (float): parameter sigma of gaussian dist.

    Returns:
        np.ndarray: gaussian fliterd img
    """
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)

    return img


def laplassian_filter(img: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    """
    laplassian filter

    Args:
        img (np.ndarray): img
        kernel_size (int): kernel size
        sigma (float): parameter sigma of gaussian dist.

    Returns:
        np.ndarray: laplassian filterd img
    """
    img = cv2.Laplacian(img, ddepth=-1, ksize=kernel_size, scale=sigma)

    return img


def thresholding_filter(img: np.ndarray, threshold: int or float) -> np.ndarray:
    """
    thresholding filter

    Args:
        img (np.ndarray): img
        threshold (float): threshold

    Returns:
        np.ndarray: thresholding filterd img
    """
    img = np.where(img > threshold, 0, 255).astype(np.uint8)

    return img
