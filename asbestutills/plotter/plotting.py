import cv2
import matplotlib.pyplot as plt
from asbestutills._reader import read_segmentation_labels
from asbestutills._path import list_ext, list_images
from pathlib import Path
import numpy as np
from asbestutills._path import list_images
import os
import sys
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from typing import List, Tuple, Union


def model_keypoint(image, model, **kwargs):
    results = model(image, iou=0.4)
    kpnt = results[0].keypoints.xy.detach().cpu().numpy().astype(int)
    image = _image_with_keypoint(image, kpnt, **kwargs)
    return image


def _image_with_keypoint(image, keypoint, thickness=8, color=(0, 160, 0)):
    """
    Draw keypoint on image
    results[0].keypoints.xy.detach().cpu().numpy().astype(int):

    """
    for coords in keypoint:
        coords = coords.reshape(-1)
        p1 = coords[:2]
        p2 = coords[2:4]
        p3 = coords[4:6]
        p4 = coords[6:8]
        image = cv2.circle(image, p1, 12, color, 2 * thickness)
        image = cv2.circle(image, p2, 12, color, 2 * thickness)
        image = cv2.line(image, p1, p2, color, thickness)
        image = cv2.circle(image, p3, 12, color, 2 * thickness)
        image = cv2.circle(image, p4, 12, color, 2 * thickness)
    return image


def draw_obounding_box(img, norm_box, thickness=8, color=125):
    h, w, c = img.shape
    for box in norm_box:
        box = np.array(box[1:])
        box[0::2] *= w
        box[1::2] *= h
        box = box.astype(np.int32)
        x0, y0, x1, y1, x2, y2, x3, y3 = box
        img = cv2.line(img, (x0, y0), (x1, y1), thickness=thickness, color=color)
        img = cv2.line(img, (x1, y1), (x2, y2), thickness=thickness, color=color)
        img = cv2.line(img, (x2, y2), (x3, y3), thickness=thickness, color=color)
        img = cv2.line(img, (x3, y3), (x0, y0), thickness=thickness, color=color)
    return img


def plot_images(path2label, path2images, path2save, thickness=8, color=125):
    path2save = Path(path2save)
    f_labels = {Path(x).stem: Path(path2label) / Path(x) for x in list_ext(path2label)}
    f_images = {
        Path(x).stem: Path(path2images) / Path(x) for x in list_images(path2images)
    }

    for name, f_path in f_labels.items():
        labels = read_segmentation_labels(f_labels[name])
        img = cv2.imread(str(f_images[name]))
        img = draw_obounding_box(img, labels, thickness, color)
        cv2.imwrite(str(path2save / "{}.jpg".format(name)), img)


def max_size(x1, y1, x2, y2, x3, y3, x4, y4):
    dx = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    dy = np.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2)
    return max(dx, dy)


class Plotter:
    def __init__(self, path2image, f_plot):
        self.files = [Path(path2image) / x for x in list_images(path2image)]
        self.f_plot = f_plot

    def __iter__(self):
        self.a = 1
        return self

    def __next__(self, **kw):
        if len(self.files) > 0:
            fp = self.files.pop()
            image = cv2.imread(str(fp))
            image = self.f_plot(image, kw)
            return image
        else:
            raise StopIteration


def plot_with_yolo(
    model, path2image, path2save, thickness=6, color=(0, 100, 255), **kwarg
):
    f_images = {
        Path(x).stem: Path(path2image) / Path(x) for x in list_images(path2image)
    }
    path2save = Path(path2save)
    for name, fpath in f_images.items():
        image = cv2.imread(str(fpath))
        results = model(image, iou=0.4, **kwarg)
        if len(results) == 0:
            continue

        for k, box in enumerate(results[0].obb.xyxyxyxy):
            box = box.detach().cpu().numpy()
            x0, y0 = box[0].astype(np.int64)
            x1, y1 = box[1].astype(np.int64)
            x2, y2 = box[2].astype(np.int64)
            x3, y3 = box[3].astype(np.int64)
            image = cv2.line(
                image, (x0, y0), (x1, y1), thickness=thickness, color=color
            )
            image = cv2.line(
                image, (x1, y1), (x2, y2), thickness=thickness, color=color
            )
            image = cv2.line(
                image, (x2, y2), (x3, y3), thickness=thickness, color=color
            )
            image = cv2.line(
                image, (x3, y3), (x0, y0), thickness=thickness, color=color
            )
            d = max_size(x0, y0, x1, y1, x2, y2, x3, y3)
            x1, y1, x2, y2 = results[0].obb[k].xyxy[0]
            xc = int(x1 + (x2 - x1) / 2)
            yc = int(y1 + (y2 - y1) / 2)
            image = cv2.putText(image, f"{int(d)}", (xc, yc), 2, 1, (0, 128, 255), 3)
        cv2.imwrite(str(path2save / fpath.name), image)


def yolo2xyxy(xc, yc, w, h):
    """
    Parameters
    ----------
    xc : float   X coord center point in Yolo format
    yc : float   Y coord center point in Yolo format
    w  : float   Weight of box in Yolo format
    h  : float   Height of box in Yolo format
    Returns
    -------
    x1 : float top left x
    y1 : float top left y
    x2 : float bottom right x
    y2 : float bottom right y
    """
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return x1, y1, x2, y2


def plot_masks(segments: List[np.ndarray], fig=None, color=[0, 0, 1], alpha=1):
    if fig:
        fig = fig
        ax = fig.gca()
    else:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca()

    for i, label in enumerate(segments):
        polygon = Polygon([(x, y) for x, y in zip(label[1::2], label[2::2])], alpha)
        polygon.set_color(color)
        polygon.set_alpha(alpha)
        ax.add_patch(polygon)
        plt.ylim(0, 1024)
        plt.xlim(0, 1024)
    return fig


def plot_bboxs(image, bboxs, color=None, line_thickness=None, sline=cv2.LINE_AA):
    res_image = image.copy()
    color = color or [255, 0, 0]
    tl = (
        line_thickness
        or round(0.002 * (res_image.shape[0] + res_image.shape[1]) / 2) + 1
    )
    scale_h, scale_w = res_image.shape[:2]
    for bbox in bboxs:
        scale_x = bbox[[0, 2]] * scale_w
        scale_y = bbox[[1, 3]] * scale_h
        c1 = (int(scale_x[0]), int(scale_y[0]))
        c2 = (int(scale_x[1]), int(scale_y[1]))
        res_image = cv2.rectangle(res_image, c1, c2, color, tl, lineType=sline)
    return res_image


def drawline(img, pt1, pt2, color, thickness=1, gap=10):
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + 0.5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + 0.5)
        p = (x, y)
        pts.append(p)
    s = pts[0]
    e = pts[0]
    i = 0
    for p in pts:
        s = e
        e = p
        if i % 2 == 1:
            cv2.line(img, s, e, color, thickness)
        i += 1


def drawpoly(img, pts, color, thickness):
    s = pts[0]
    e = pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s = e
        e = p
        drawline(img, s, e, color, thickness)


def drawrect(img, pt1, pt2, color, thickness=1):
    pts = [pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])]
    drawpoly(img, pts, color, thickness)


def read_img(path):
    image = Image.read(path)


def read_segmentation_labels(p):
    with open(p, "r") as f:
        lines = f.readlines()
        return [np.fromstring(line, sep=" ") for line in lines]


def draw_bbox(img, path2label, color=(0, 255, 1), lw=2):
    labels = read_segmentation_labels(path2label)
    if len(img.shape) > 2:
        h, w, _ = img.shape
    else:
        h, w = img.shape
    arr_label = []
    for label in labels:
        xc, yc, wl, hl = label[1:5]
        b = yolo2xyxy(xc, yc, wl, hl)
        arr_label.append(b)

    arr_label = np.array(arr_label)
    arr_label[:, [0, 2]] *= w
    arr_label[:, [1, 3]] *= h
    annotator = Annotator(img, lw)
    for box in arr_label:
        box = np.array(box).reshape(1, -1)
        annotator.add_box(
            box.astype(np.int32).tolist()[0],
            color=color,
        )
    return annotator.img


def draw_box_with_keypoints(path2label, path2image, rescale=True, color=(0, 255, 0)):
    distance = lambda p1, p2: np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    labels = read_segmentation_labels(path2label)
    image = cv2.imread(path2image)
    if rescale:
        image = image[::2, ::2, :]
    h, w, _ = image.shape
    new_image = image
    new_image = draw_bbox(new_image, path2label, color=color, lw=1)
    counter = 0
    for a in labels:
        arr_label = np.array(a)
        arr_label[[5, 7]] *= w
        arr_label[[6, 8]] *= h
        arr_label[[9, 11]] *= w
        arr_label[[10, 12]] *= h

        c = np.round(arr_label).astype(np.int32)
        # xyxy = xywhn2xyxy(arr_label[1:5].reshape(1,-1), w, h)[0]
        d1 = max(arr_label[3] * w, arr_label[4] * h)
        d2 = distance((c[5], c[6]), (c[7], c[8]))
        if abs(d1 - d2) / max(d1, d2) * 100 > 20:
            color = (255, 0, 0)
            counter += 1
        else:
            color = (0, 255, 0)
        thickness = 2
        new_image = cv2.circle(
            np.array(new_image), (c[5], c[6]), 3, color=color, thickness=3
        )
        new_image = cv2.circle(new_image, (c[7], c[8]), 3, color=color, thickness=3)
        new_image = cv2.line(
            np.array(new_image), (c[5], c[6]), (c[7], c[8]), color=color, thickness=3
        )

        new_image = cv2.circle(
            np.array(new_image),
            (c[9], c[10]),
            3,
            color=(255, 0, 255),
            thickness=thickness,
        )
        new_image = cv2.circle(
            new_image, (c[11], c[12]), 3, color=(255, 0, 255), thickness=thickness
        )
        new_image = cv2.line(
            np.array(new_image),
            (c[9], c[10]),
            (c[11], c[12]),
            color=color,
            thickness=thickness,
        )

    new_image = cv2.putText(
        np.array(new_image),
        f"{counter/len(labels):.2f}",
        (50, 50),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=2,
        color=(255, 0, 15),
        thickness=3,
    )
    return new_image


class Annotator:
    def __init__(self, img, line_width=None):
        self.img = img if isinstance(img, Image.Image) else Image.fromarray(img)
        self.draw = ImageDraw.Draw(self.img)
        h, w = self.img.size
        self.lw = line_width or max(round(sum([h, w]) / 2 * 0.003), 2)  # line width
        # print(self.img.size)

    def masks(self, segments: List[np.array], color=[0, 0, 0], alpha=0.9):
        """
        List segments: np.array([[x1, y1], [x2, y2], .. ,[xn,yn]])
        """
        assert isinstance(segments, List), "segments not List"
        for segment in segments:
            assert len(segment.shape) == 2, "Every segment have shape size 2"
        image = self.img.copy()
        for segment in segments:
            image = cv2.fillPoly(np.array(image), pts=[segment], color=color)

        image = cv2.addWeighted(np.array(self.img), alpha, image, 1 - alpha, 0.0)
        # Копируем обратно
        self.img = Image.fromarray(image)
        self.draw = ImageDraw.Draw(self.img)

    def result(self):
        return self.img

    def add_box(
        self,
        box: List,
        label: str = "",
        color: Union[Tuple, List] = (128, 128, 128),
        style=None,
        thickness=3,
    ):
        """
        box: [x1, y1, x2, y2]
        Add bbox on image
        """
        if isinstance(box, np.ndarray):
            box = box.tolist()
        if len(np.array(self.img).shape) == 1:
            color = np.mean(color)
        if style == "dotted":
            image = np.array(self.img)
            x1, y1, x2, y2 = box
            drawrect(image, (x1, y1), (x2, y2), color, thickness)
            self.img = Image.fromarray(image)
            self.draw = ImageDraw.Draw(self.img, width=thickness)
        else:
            self.draw.rectangle(box, width=self.lw, outline=color[0])

    def add_polygone(self, polygone: List[Tuple[float, float]], color=(128, 128, 0)):
        # [Tuple()]
        self.draw.polygon(polygone, width=self.lw, outline=color)

    def add_text(self, x, y, text, color=(0, 256, 0), thickness=2):
        image = np.array(self.img)
        image = cv2.putText(
            image,
            str(text),
            (int(x), int(y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            thickness,
            cv2.LINE_AA,
        )
        self.img = Image.fromarray(image)

    def save(self, filename):
        self.img.save(filename, quality=100)

    @property
    def image(self):
        return np.array(self.img.convert("RGB"))
