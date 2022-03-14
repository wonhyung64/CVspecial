#%%
import os
import tensorflow as tf
from PIL import Image
from pathlib import Path
from typing import List

#%%
def load_data() -> List:
    """
    LOAD example image data

    Returns:
        List: imgs
    """
    cwd = Path(os.getcwd())
    ex_dir = f"{cwd}/ex"
    img_lst = os.listdir(ex_dir)

    imgs = []
    for i in range(len(img_lst)):
        img_dir = f"{ex_dir}/{img_lst[i]}"
        img = Image.open(img_dir)
        img = tf.convert_to_tensor(img)
        img = tf.cast(img, dtype=tf.float32)
        imgs = [img] + imgs

    return imgs


#%%
def preprocessing(imgs: List) -> List:
    """
    crop and resize example image data

    Args:
        imgs (List): imgs

    Returns:
        List: preprocessed imgs
    """
    img0 = tf.image.resize(imgs[0], (200, 200))
    img1 = tf.image.resize(imgs[1], (200, 200))
    img2 = imgs[2][35:235, 85:285, ...]
    img3 = imgs[3][35:235, 135:335, ...]
    imgs = [img0, img1, img2, img3]

    return imgs

