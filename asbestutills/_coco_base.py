import os
import numpy as np
import json
import cv2
from pycocotools import mask as cocoutils
from typing import List
from .plotter.plotting import drawpoly, drawline

# -----------------------------------
def _ann2mask(ann, h, w):
    """
    transform instant segmentation into binary image,
      with shape h, w.

    Paramters
    ----------
    ann: dict,
      annotation instance in COCO format.
    h,w: int, int,
      required output shape.

    Returns
    ------------
    ndarray: numpy 2d array image mask with values:
       0 for background,1 for object.
    """
    segm = ann["segmentation"]
    rles = cocoutils.frPyObjects(segm, h, w)
    rle = cocoutils.merge(rles)
    instant_mask = cocoutils.decode(rle)
    return instant_mask


def _masks2image(masks):
    """
    transform instant masks into image
      expect to have masks in the format:
      instances x height x width.

    Paramters
    ----------
    masks: ndarray,
      3d array in form instances x height x width.

    Returns
    ------------
    ndarray: numpy 3d array image
      with random colors for each instat.
    """
    img_ = [[x * mask for x in np.random.randint(0, 255, 3)] for mask in masks]

    img_ = np.array(img_).sum(axis=0)

    img_ = img_.astype(float)
    img_ = (img_ - img_.min()) / (img_.max() - img_.min())
    img_ = (255 * img_).astype(np.uint8)

    return img_.transpose((1, 2, 0))


# -----------------------------------
def _masks2d(masks):
    """
    transform instant masks into 2d image
      expect to have masks in the format:
      instances x height x width.

    Paramters
    ----------
    masks: ndarray,
      3d array in form instances x height x width.

    Returns
    ------------
    ndarray: numpy 2d array image
      with different values for different instances.

    Note
    -----
    if masks are overlaped, the upper value is selected.
    """
    h, w = masks.shape[1:3]
    mask = np.zeros((h, w), dtype=np.uint8)
    for i, mask_ in enumerate(masks):
        mask += mask_ * (i + 1)
        if mask.max() > i + 1:
            mask[mask > i + 1] = i + 1
    return mask


# -----------------------------------
def _image_with_bbox(img, bboxes, color=0, thickness=10):
    """
    Get image with drawn bounding boxes.

    Parameters
    ----------
    img: ndarray,
      image in form height x width x channels.
    bboxes: list[list[int]],
      bounding boxes in format [[x0,y0,w,h]].
    color: int; [int,int,int],
      color for box bounds, int format for brightness,
      [int,int,int] format for color.
    thikness: int,
      thikness for box bounds.

    Returns
    ----------
    ndarray: image array with drawn bounding boxes.
    """
    for bbox in np.atleast_2d(bboxes).astype(int):
        x0, y0 = bbox[:2]
        x1, y1 = bbox[:2] + bbox[2:]
        img = cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness)
    return img


def _image_with_contours(
    image, polygones: List[np.ndarray], color=(133, 255, 24), thickness=6
):
    """
    Get image with drawn contours.
    """
    for p in polygones:
        if not isinstance(p, np.ndarray):
            p = np.array(p)
        if p.shape[1] != 2:
            assert "Shape of polygones (n,2)"
    image = cv2.polylines(image, polygones, True, color, thickness)
    return image

def _image_with_obbox(img, obboxes, color=0, thickness = 1, dotted = False, gap = 10):
    """
    Get image with drawn oriented bounding boxes.

    Parameters
    ----------
    img: ndarray,
      image in form height x width x channels.
    obboxes: list[list[int]],
      orientide bounding boxes in format [[x0,y0,x1,y1,x2,y2,x3,y3]].
    color: int; [int,int,int],
      color for box bounds, int format for brightness,
      [int,int,int] format for color.
    thikness: int,
      thikness for box bounds.
    Returns
    ----------
    ndarray: image array with drawn bounding boxes.
    """
    if dotted:
        for box in obboxes:
            try:
                box = np.round(box).astype(np.int32)
                drawline(img, (box[0], box[1]), (box[2], box[3]), color, thickness, gap)
                drawline(img, (box[2], box[3]), (box[4], box[5]), color, thickness, gap)
                drawline(img, (box[4], box[5]), (box[6], box[7]), color, thickness, gap)
                drawline(img, (box[6], box[7]), (box[0], box[1]), color, thickness, gap)
            except Exception as e:
                pass
    else:
        for box in obboxes:
            try:
                c = np.round(box).astype(np.int32)
                img = cv2.line(img, (c[0],c[1]),(c[2],c[3]), color=color, thickness = thickness)
                img = cv2.line(img, (c[2],c[3]),(c[4],c[5]), color=color, thickness = thickness)
                img = cv2.line(img, (c[4],c[5]),(c[6],c[7]), color=color, thickness = thickness)
                img = cv2.line(img, (c[6],c[7]),(c[0],c[1]), color=color, thickness = thickness)
            except Exception as e:
                pass
    return img
