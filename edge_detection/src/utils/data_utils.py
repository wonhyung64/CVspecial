import os
import numpy as np
from PIL import Image
from typing import List

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