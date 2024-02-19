import cv2
import matplotlib.pyplot as plt
from asbestutills._reader import read_segmentation_labels
from asbestutills._path import list_ext, list_images
from pathlib import Path
import numpy as np
from asbestutills._path import list_images


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
        results = model(image, iou=0.4)
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
        cv2.imwrite(str(path2save / "{}.jpg".format(name)), image)


# from ultralytics import YOLO

# model = YOLO("/storage/reshetnikov/runs/obb_compare/trans_obb_v8x2/weights/best.pt")
# plot_with_yolo(
#     model,
#     "/storage/reshetnikov/test_transport/fold/Fold_0/train/",
#     "/storage/reshetnikov/runs/obb_compare/predict/cmp_train/",
#     color=(0, 255, 0),
# )
