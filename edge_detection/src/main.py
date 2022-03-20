#%%
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import cv2
from typing import List

#%%
def load_data() -> List:
    """
    Load images

    Returns:
        List: list of ex images
    """
    img_dir = f"{os.getcwd()}/ex"
    img_dir = [f"{img_dir}/{img}" for img in os.listdir(img_dir) if "jpg" in img or "png" in img]
    imgs = [np.array(Image.open(img)) for img in img_dir]

    return imgs

imgs = load_data()
img = imgs[0]

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

img_gray = gray_filter(img)
array_to_img(img_gray)

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

img_blur = gaussian_filter(img_gray, 7, 0.)
array_to_img(img_blur)

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

img_blur_log = laplassian_filter(img_gray, 5, 1)
array_to_img(img_blur_log)
#%%
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
    if grad == "x": dx, dy = 1, 0
    elif grad == "y": dx, dy = 0, 1
    elif grad == "xy": dx, dy = 1, 1
    img = cv2.Sobel(src=img, ddepth=-1, dx=dx, dy=dy, ksize=kernel_size)

    return img
    
sobel_x = sobel_edge(img_blur, "x", 5)
sobel_y = sobel_edge(img_blur, "y", 5)
sobel_xy = sobel_edge(img_blur, "xy", 5)

array_to_img(sobel_x)
array_to_img(sobel_y)
array_to_img(sobel_xy)
array_to_img(sobel_x + sobel_y)


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

threshold = 127.5
sobel = thresholding_filter(sobel_x + sobel_y, threshold)
array_to_img(sobel)


def canny_edge(img: np.ndarray, threshold1: int or float, threshold2: int or float) -> np.ndarray:
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

gaussian_canny = canny_edge(img_blur, threshold1=100, threshold2=200.)
log_canny = canny_edge(img_blur_log, threshold1=100, threshold2=200)
array_to_img(gaussian_canny)
array_to_img(log_canny)


# %%
