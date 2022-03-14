#%%
import os
import tensorflow as tf
from pathlib import Path
from PIL import Image
from typing import List

#%%
def load_data() -> List:
    """
    LOAD ex data

    Returns:
        List: imgs
    """
    cwd = Path(os.getcwd())
    parent_wd = cwd.parent
    ex_dir = f"{parent_wd}/ex"
    img_lst = os.listdir(ex_dir)

    imgs = []
    for i in range(len(img_lst)):
        img_dir = f"{ex_dir}/{img_lst[i]}"
        img = Image.open(img_dir)
        img = tf.convert_to_tensor(img)
        imgs = [img] + imgs

    return imgs


#%%
imgs = load_data()
tf.shape(imgs[0])
tf.shape(imgs[1])

img0 = imgs[0]
tf.keras.preprocessing.image.array_to_img(img0)
tf.keras.preprocessing.image.array_to_img(img1)
tf.kerasimg1[30:230,...]
img1 = imgs[1]
img2 = imgs[2][20:220, 10:360, ...]
img3 = imgs[3][35:235, 60:410, ...]
tf.keras.preprocessing.image.array_to_img(img2)
tf.keras.preprocessing.image.array_to_img(img3)
tf.keras.preprocessing.image.array_to_img(imgs[1][15:215, 10:360, ...])
tf.keras.preprocessing.image.array_to_img(tf.image.central_crop(imgs[0], 0.7))
tmp = tf.image.crop_and_resize(tf.expand_dims(imgs[0], 0), boxes=[[0.0, 0.0, 0.5, 0.5]], crop_size = (250, 250), box_indices=[0])


