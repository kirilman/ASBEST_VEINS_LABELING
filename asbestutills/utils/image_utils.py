
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import cv2 
from tqdm import tqdm

IMAGE_EXTENTIONS = (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")

def resize(image,f=0.3):
    """Resize 'image' by 'f' in both dimensions."""
    newDim = (int(f*image.shape[0]),int(f*image.shape[1]))
    return cv2.resize(image, (newDim[1], newDim[0]), interpolation=cv2.INTER_CUBIC)


def resize_dir(path2dir, path2save, scale = 0.5):
    files = [x for x in Path(path2dir).rglob("*") if x.suffix in IMAGE_EXTENTIONS]
    for f in tqdm(files):
        img = cv2.imread(str(f))
        img = resize(img,scale)
        cv2.imwrite(str(path2save / f.name), img)