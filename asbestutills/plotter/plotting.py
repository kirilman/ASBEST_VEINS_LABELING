import cv2
import matplotlib.pyplot as plt
from asbestutills._reader import read_segmentation_labels
from asbestutills._path import list_ext, list_images
from pathlib import Path
import numpy as np

def draw_obounding_box(img, norm_box, thickness = 8,color = 125):
    h,w,c = img.shape
    for box in norm_box:
        box = np.array(box[1:])
        box[0::2]*=w
        box[1::2]*=h
        box = box.astype(np.int32)
        x0,y0,x1,y1,x2,y2,x3,y3 = box
        img =  cv2.line(img, (x0,y0),(x1,y1), thickness = thickness,color= color)
        img =  cv2.line(img, (x1,y1),(x2,y2), thickness = thickness,color= color)
        img =  cv2.line(img, (x2,y2),(x3,y3), thickness = thickness,color= color)
        img =  cv2.line(img, (x3,y3),(x0,y0), thickness = thickness,color= color)
    return img

def plot_images(path2label, path2images, path2save, thickness = 8,color = 125):
    path2save = Path(path2save)
    f_labels = {Path(x).stem: Path(path2label) / Path(x) for x in list_ext(path2label)}
    f_images = {Path(x).stem: Path(path2images) / Path(x) for x in list_images(path2images)}

    for name,f_path in f_labels.items():
        labels = read_segmentation_labels(f_labels[name])
        img = cv2.imread(str(f_images[name]))
        img = draw_obounding_box(img, labels, thickness,color)
        cv2.imwrite(str(path2save / "{}.jpeg".format(name)), img)
