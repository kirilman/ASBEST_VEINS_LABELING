import numpy as np
import os
import pandas as pd

from tqdm import main
from tqdm import tqdm
try:
    from ._path import list_ext, list_images
except:
    from _path import list_ext, list_images

from pathlib import Path
from PIL import Image
import json
from typing import List
from pycocotools.coco import COCO
from pycocotools import mask
try:
    from .utils.geometry import (
        coords_other_line,
        point_intersection,
        vec_from_points,
        line_from_points,
        dot_product_angle,
        correct_sequence,
        coords_other_line_by_coords,
        coords_max_line,
        position,
        distance,
        distance_to_perpendicular,
        coords_main_line,
        coords_other_line,
    )
except:
    from utils.geometry import (
        coords_other_line,
        point_intersection,
        vec_from_points,
        line_from_points,
        dot_product_angle,
        correct_sequence,
        coords_other_line_by_coords,
        coords_max_line,
        position,
        distance,
        distance_to_perpendicular,
        coords_main_line,
        coords_other_line,
    )
import argparse
from pylabel import importer
from collections import defaultdict
import cv2

def polygone_area(x, y):
    return 0.5 * np.array(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def yolo2coco(xc, yc, w, h, image_width, image_height):
    xc, w = xc * image_width, w * image_width
    yc, h = yc * image_height, h * image_height
    xmin = xc - (w / 2)
    ymin = yc - (h / 2)
    return xmin, ymin, w, h


def segment2box(x_coords, y_coords):
    xl = np.min(x_coords)
    yl = np.min(y_coords)
    h = np.max(y_coords) - yl
    w = np.max(x_coords) - xl
    return xl, yl, w, h


def box2segment(box: List):
    """
    box: List coco format
    """
    x, y, w, h = box
    segment = []
    segment.append(x)
    segment.append(y)
    segment.append(x + w)
    segment.append(y)
    segment.append(x + w)
    segment.append(y + h)
    segment.append(x)
    segment.append(y + h)
    return segment


def ellipse_parameters(x, y):
    from skimage.measure import EllipseModel

    a_points = np.array([x, y]).T
    ell = EllipseModel()
    ell.estimate(a_points)
    return ell.params


def clear_negative_values(x):
    if x <= 0:
        return 0.001
    else:
        return x


def correct(points, W, H):
    # points = [p1,p2,p3,p4]
    p_new = None
    for k, (x, y) in enumerate(points):
        if x < 0:
            p_left = point_intersection(*points[k - 1], x, y, 0, 0, 0, W)
            if k == 3:
                p_right = point_intersection(*points[0], x, y, 0, 0, 0, W)
            else:
                p_right = point_intersection(*points[k + 1], x, y, 0, 0, 0, W)

            y_middle = p_right[1] + (p_right[1] - p_left[1]) / 2
            p_new = round(p_left[0]), y_middle
        elif y < 0:
            p_left = point_intersection(*points[k - 1], x, y, 0, 0, H, 0)
            if k == 3:
                p_right = point_intersection(*points[0], x, y, 0, 0, H, 0)
            else:
                p_right = point_intersection(*points[k + 1], x, y, 0, 0, H, 0)

            y_middle = p_right[1] + (p_right[1] - p_left[1]) / 2
            p_new = round(p_left[0]), y_middle
        if p_new:
            points[k] = p_new
            p_new = None
    return points

def mask_to_polygons(mask: np.ndarray) -> List[np.ndarray]:
    """
    Converts a binary mask to a list of polygons.

    Parameters:
        mask (np.ndarray): A binary mask represented as a 2D NumPy array of
            shape `(H, W)`, where H and W are the height and width of
            the mask, respectively.

    Returns:
        List[np.ndarray]: A list of polygons, where each polygon is represented by a
            NumPy array of shape `(N, 2)`, containing the `x`, `y` coordinates
            of the points. Polygons with fewer points than `MIN_POLYGON_POINT_COUNT = 3`
            are excluded from the output.
    """

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    return [
        np.squeeze(contour, axis=1)
        for contour in contours
    ]

class Yolo2Coco:
    def __init__(
        self,
        path_label: str = "",
        path_image: str = "",
        path_save_json: str = "",
        step_sampling: int = 1,
    ):
        self.path_label = path_label
        self.path_image = path_image
        self.step_sampling = step_sampling
        self.image_paths = {
            Path(p).stem: os.path.join(self.path_image, p)
            for p in list_images(self.path_image)
        }
        self.label_paths = {
            Path(p).stem: os.path.join(self.path_label, p)
            for p in list_ext(self.path_label)
        }
        self.path_save_json = path_save_json

    def get_image_path(self, image_name):
        """
        Return:
        image_path: Path, path to image
        """
        return self.image_paths[image_name]

    def get_label_path(self, file_name):
        return self.label_paths[file_name]

    def get_image_hw(self, image_name):
        """
        Get image height and weight
        Returns
        ----------
        height: int
        weight: int
        """
        image_path = self.get_image_path(image_name)
        image = np.array(Image.open(image_path))
        (
            height,
            weight,
        ) = (
            image.shape[0],
            image.shape[1],
        )  # Важно
        return height, weight

    def _collect_images(self):
        """
        Return
        -----------
        images: list[dist], collected images
        """
        images = {}
        img_id = 1
        for f_path in self.image_paths.values():
            h, w = self.get_image_hw(Path(f_path).stem)
            image_dict = {
                "id": img_id,
                "file_name": Path(f_path).name,
                "width": w,
                "height": h,
                "licence": "",
                "date_captured": 0,
            }
            images[Path(f_path).stem] = image_dict
            img_id += 1
        return images

    def _collect_annotations(self, images_ids):
        """
        YOLO.txt : cls, (x1,y1), (x2,y2) ...(xn,yn)
        Return
        -----------
        annotations: list[dict], annotation dict
        categories : list[int], classes

        """
        anno_id = 0
        annotations = []
        categories = []
        fname_list = list_ext(self.path_label, "txt")
        for _, fname in tqdm(enumerate(fname_list)):
            with open(self.get_label_path(fname.split(".")[0]), "r") as f:
                lines = f.readlines()
            h, w = self.get_image_hw(Path(fname).stem)

            for line in lines:
                data = np.fromstring(line, sep=" ")
                if len(data) < 2:
                    continue
                o_cls, segment = data[0], data[1:]
                o_cls += 1
                image_id = images_ids[Path(fname).stem]["id"]
                if len(segment) == 4:
                    bbox = yolo2coco(
                        segment[0], segment[1], segment[2], segment[3], w, h
                    )
                    annotations.append(
                        {
                            "id": anno_id,
                            "image_id": image_id,
                            "category_id": int(o_cls),
                            "segmentation": [box2segment(bbox)],
                            "area": bbox[2] * bbox[3],
                            "bbox": bbox,
                            "iscrowd": 0,
                        }
                    )
                else:
                    x_coords, y_coords = segment[0::2] * w, segment[1::2] * h
                    if len(x_coords) > 16:
                        x_coords = x_coords[:: self.step_sampling]
                        y_coords = y_coords[:: self.step_sampling]
                    coco_segment = []
                    for x, y in zip(x_coords, y_coords):
                        coco_segment.append(x)
                        coco_segment.append(y)

                    area = polygone_area(x_coords, y_coords)
                    if area > 625:
                        annotations.append(
                            {
                                "id": anno_id,
                                "image_id": image_id,
                                "category_id": int(o_cls),
                                "segmentation": [coco_segment],
                                "area": polygone_area(x_coords, y_coords),
                                "bbox": segment2box(x_coords, y_coords),
                                "iscrowd": 0,
                            }
                        )

                if not o_cls in categories:
                    categories.append(int(o_cls))
                anno_id += 1

        return annotations, categories

    def convert(self):
        images = self._collect_images()
        annotations, classes = self._collect_annotations(images)
        info = {
            "year": "2023",
            "version": "1.0",
            "description": "Asbest dataset",
            "contributor": "",
            "url": "https://data.mendeley.com/v1/datasets/pfdbfpfygh/draft?preview=1",
            "date_created": "",
        }
        licenses = [
            {
                "url": "https://data.mendeley.com/v1/datasets/pfdbfpfygh/draft?preview=1",
                "id": 1,
                "name": "openpits asbestos",
            }
        ]
        class_names = {0: "stone", 1: "stone", 2: "yolo_stone"}
        categories = [
            {"id": _cls, "name": class_names[_cls], "supercategory": ""}
            for _cls in classes
        ]
        data = {
            "info": info,
            "licenses": licenses,
            "images": list(images.values()),
            "annotations": annotations,
            "categories": categories,
        }
        with open(self.path_save_json, "w") as f:
            json.dump(data, f)
        print("Save result to", self.path_save_json)


def coco2obb(path2json, path2save):
    """
        Convert coco polygon coordinates to obb format coordinates.
        The obbox is rotated along the main axis of the approximating ellipse!
        Save the result in *.txt files in path2save directory
    Args:
        path2json (str): json with coco format
        path2save (str): save directory
    """
    coco = COCO(path2json)
    frame = pd.DataFrame(coco.anns).T
    df_image = pd.DataFrame(coco.imgs).T
    image_dict = df_image.T.to_dict()
    print(image_dict)
    fname = str(
        df_image[df_image.id == frame.iloc[0].image_id]["file_name"]
        .values[0]
        .split(".")[0]
    )
    file_out = open(Path(path2save) / (fname + ".txt"), "w")

    for k, row in frame.iterrows():
        IMAGE_W = image_dict[row.image_id]["width"]
        IMAGE_H = image_dict[row.image_id]["height"]

        try:
            x_coords = np.array(row.segmentation[0][::2])  # /IMAGE_W
            y_coords = np.array(row.segmentation[0][1::2])  # /IMAGE_H
            xc, yc, a, b, theta = ellipse_parameters(x_coords, y_coords)
        except:
            print("Failed to obtain ellipse parameters for ", row)
            continue
        x1, y1, x2, y2 = coords_main_line(xc, yc, a, theta)
        x1, y1, x2, y2 = coords_other_line(xc, yc, b, theta)  # b axes
        ox1, oy1, ox2, oy2, ox3, oy3, ox4, oy4 = coords_obb(x1, y1, x2, y2, a, theta)

        # if any(t < 0 for t in (ox1, oy1, ox2, oy2, ox3, oy3, ox4, oy4)):
        # continue
        ox1 = clear_negative_values(ox1)
        oy1 = clear_negative_values(oy1)
        ox2 = clear_negative_values(ox2)
        oy2 = clear_negative_values(oy2)

        ox3 = clear_negative_values(ox3)
        oy3 = clear_negative_values(oy3)
        ox4 = clear_negative_values(ox4)
        oy4 = clear_negative_values(oy4)

        cls_id = row.category_id - 1
        cls_id = "stone"
        current_fname = str(
            df_image[df_image.id == row.image_id]["file_name"].values[0].split(".")[0]
        )
        if fname == current_fname:
            line = (
                "{:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {} 0\n".format(
                    ox1, oy1, ox2, oy2, ox3, oy3, ox4, oy4, cls_id
                )
            )
            file_out.write(line)

        else:
            file_out.close()
            fname = current_fname
            file_out = open(Path(path2save) / (fname + ".txt"), "a")
            line = (
                "{:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {} 0\n".format(
                    ox1, oy1, ox2, oy2, ox3, oy3, ox4, oy4, cls_id
                )
            )
            file_out.write(line)
    file_out.close()
    return True


def coco2obb_maxline(path2json, path2save, norm=True):
    coco = COCO(path2json)
    frame = pd.DataFrame(coco.anns).T
    df_image = pd.DataFrame(coco.imgs).T
    image_dict = df_image.T.to_dict()
    df_image = pd.DataFrame(coco.imgs).T
    fname = str(
        df_image[df_image.id == frame.iloc[0].image_id]["file_name"]
        .values[0]
        .split(".")[0]
    )
    file_out = open(Path(path2save) / (fname + ".txt"), "w")
    print(fname, file_out)
    number = 0
    current_line = 0
    for k, row in frame.iterrows():
        IMAGE_W = image_dict[row.image_id]["width"]
        IMAGE_H = image_dict[row.image_id]["height"]
        try:
            x_coords = np.array(row.segmentation[0][::2])  # /IMAGE_W
            y_coords = np.array(row.segmentation[0][1::2])  # /IMAGE_H
        except:
            print("Failed to obtain x_coords, y_coords for ", row)
            continue

        ax1, ay1, ax2, ay2 = coords_max_line(x_coords, y_coords)
        if ax2 > ax1 and ay2 > ay1:
            ax2, ax1 = ax1, ax2
            ay2, ay1 = ay1, ay2
        Points = [(x, y) for x, y in zip(x_coords, y_coords)]
        max_dist_left = 0
        max_dist_right = 0
        for point in Points:
            if position(point[0], point[1], ax1, ay1, ax2, ay2) > 0:
                A, B, C = line_from_points((ax1, ay1), (ax2, ay2))
                d = distance_to_perpendicular(A, B, C, point[0], point[1])
                if abs(d) > max_dist_right:
                    max_dist_right = d
                    bx2, by2 = point
            else:
                A, B, C = line_from_points((ax1, ay1), (ax2, ay2))
                d = distance_to_perpendicular(A, B, C, point[0], point[1])
                if abs(d) > max_dist_left:
                    max_dist_left = d
                    bx1, by1 = point

        px1, px2 = point_intersection(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2)
        n1, n2 = vec_from_points((px1, px2), (ax1, ay1))
        m1, m2 = vec_from_points((px1, px2), (px1 + 1000, px2))
        theta = dot_product_angle([n1, n2], [m1, m2])

        A, B, C = line_from_points((ax1, ay1), (ax2, ay2))
        h2 = distance_to_perpendicular(A, B, C, bx2, by2)
        h1 = distance_to_perpendicular(A, B, C, bx1, by1)
        alpha = np.pi / 2 - theta
        if ay1 < px2:
            alpha = np.pi / 2 + theta
        dy = h1 * np.sin(alpha)
        dx = h1 * np.cos(alpha)
        # coords obb obx1, oby1, ...
        obx1 = ax1 + dx
        oby1 = ay1 - dy
        obx4 = ax2 + dx
        oby4 = ay2 - dy
        dy = h2 * np.sin(alpha)
        dx = h2 * np.cos(alpha)
        obx2 = ax1 - dx
        oby2 = ay1 + dy
        obx3 = ax2 - dx
        oby3 = ay2 + dy

        # if any(x < 0 for x in [obx1, oby1, obx2, oby2, obx3, oby3, obx4, oby4]):
        #     # print('pass ', int(obx1), int(oby1), int(obx2), int(oby2), int(obx3), int(oby3), int(obx4), int(oby4))
        #     number += 1
        #     try:
        #         # points = correct([(obx1, oby1), (obx2, oby2), (obx3, oby3), (obx4, oby4)])
        #         points = []
        #         for x, y in [(obx1, oby1), (obx2, oby2), (obx3, oby3), (obx4, oby4)]:
        #             x = clear_negative_values(x)
        #             y = clear_negative_values(y)
        #             points.append((x, y))
        #         obx1, oby1 = points[0]
        #         obx2, oby2 = points[1]
        #         obx3, oby3 = points[2]
        #         obx4, oby4 = points[3]
        #         # point_intersection()
        #     except:
        #         continue
        # obx1 = clear_negative_values(obx1)
        # oby1 = clear_negative_values(oby1)
        # obx2 = clear_negative_values(obx2)
        # oby2 = clear_negative_values(oby2)
        # obx3 = clear_negative_values(obx3)
        # oby3 = clear_negative_values(oby3)
        # obx4 = clear_negative_values(obx4)
        # oby4 = clear_negative_values(oby4)
        op1, op2, op3, op4 = correct_sequence(
            (obx1, oby1), (obx2, oby2), (obx3, oby3), (obx4, oby4)
        )
        data = np.hstack((op1, op2, op3, op4))
        if norm:
            data[::2] = data[::2] / IMAGE_W
            data[1::2] = data[1::2] / IMAGE_H

        cls_id = 0
        current_fname = str(
            df_image[df_image.id == row.image_id]["file_name"].values[0].split(".")[0]
        )
        if fname == current_fname:
            line = "{:d} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} \n".format(
                cls_id, *data
            )
            file_out.write(line)

        else:
            file_out.close()
            fname = current_fname
            file_out = open(Path(path2save) / (fname + ".txt"), "a")
            line = "{:d} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} \n".format(
                cls_id, *data
            )
            file_out.write(line)
        current_line += 1
        if current_line % 1000 == 0:
            print(current_line, " ", current_fname)
    file_out.close()
    print("Quantity outside the image ", number)


def coco2box_keypoints(path2json, path2save, second_line=True):
    """
        Convert coco format to bounding box with keypoint for max line

    Args:
        path2json (_type_): _description_
        path2save (_type_): _description_
    """
    path2save = Path(path2save)
    if not path2save.is_dir():
        path2save.mkdir()
        
    coco = COCO(path2json)
    frame = pd.DataFrame(coco.anns).T
    df_image = pd.DataFrame(coco.imgs).T
    image_dict = df_image.T.to_dict()
    print(image_dict)
    fname = str(
        df_image[df_image.id == frame.iloc[0].image_id]["file_name"]
        .values[0]
        .split(".")[0]
    )
    file_out = open(Path(path2save) / (fname + ".txt"), "w")

    for k, row in frame.iterrows():
        IMAGE_W = image_dict[row.image_id]["width"]
        IMAGE_H = image_dict[row.image_id]["height"]

        try:
            x_coords = np.array(row.segmentation[0][::2]) / IMAGE_W
            y_coords = np.array(row.segmentation[0][1::2]) / IMAGE_H
        except:
            print("Failed to obtain ellipse parameters for ", row)
            continue
        if row.bbox is np.nan:
            continue
        box = np.array(row.bbox, dtype=np.float64)
        box[:2] += box[2:] / 2  # xy top-left corner to center
        box[[0, 2]] /= IMAGE_W  # normalize x
        box[[1, 3]] /= IMAGE_H  # normalize y
        xc, yc, w, h = box
        if len(x_coords) < 4:
            print(k)
            continue
        xm1, ym1, xm2, ym2 = coords_max_line(x_coords, y_coords)
        cls_id = row.category_id - 1
        # cls_id = 'stone'
        current_fname = str(
            df_image[df_image.id == row.image_id]["file_name"].values[0].split(".")[0]
        )
        line = (cls_id, xc, yc, w, h, xm1, ym2, xm2, ym2)

        if second_line:
            xs1, ys1, xs2, ys2 = coords_other_line_by_coords(x_coords, y_coords)
            line = (cls_id, xc, yc, w, h, xm1, ym1, xm2, ym2, xs1, ys1, xs2, ys2)

        write_line = ("%g " * len(line)).rstrip() % line + "\n"
        if fname == current_fname:
            file_out.write(write_line)

        else:
            file_out.close()
            fname = current_fname
            file_out = open(Path(path2save) / (fname + ".txt"), "a")
            file_out.write(write_line)
    file_out.close()


def merge_multi_segment(segments):
    """
    Merge multi segments to one list. Find the coordinates with min distance between each segment, then connect these
    coordinates with one thin line to merge all segments into one.

    Args:
        segments(List(List)): original segmentations in coco's json file.
            like [segmentation1, segmentation2,...],
            each segmentation is a list of coordinates.
    """
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    # record the indexes with min distance between each segment
    for i in range(1, len(segments)):
        idx1, idx2 = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # use two round to connect all the segments
    for k in range(2):
        # forward connection
        if k == 0:
            for i, idx in enumerate(idx_list):
                # middle segments have two indexes
                # reverse the index of middle segments
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                # deal with the first segment and the last one
                if i in [0, len(idx_list) - 1]:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0] : idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in [0, len(idx_list) - 1]:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s

def convert_coco_json(json_dir="../coco/annotations/", save_dir= './', use_segments=False, scls91to80=False):
    """Converts COCO JSON format to YOLO label format, with options for segments and class mapping."""
    # coco80 = coco91_to_coco80_class()

    # Import json
    if Path(json_dir).suffix == ".json":
        json_dir = Path(json_dir).parent
    for json_file in sorted(json_dir.resolve().glob("*.json")):
        fn = Path(save_dir)  # folder name
        fn.mkdir(exist_ok=True)
        print(f'Save path is {fn}')
        with open(json_file) as f:
            data = json.load(f)
        
        # Create image dict
        images = {"%g" % x["id"]: x for x in data["images"]}
        # Create image-annotations dict
        imgToAnns = defaultdict(list)
        for ann in data["annotations"]:
            imgToAnns[ann["image_id"]].append(ann)

        # Write labels file
        for img_id, anns in tqdm(imgToAnns.items(), desc=f"Annotations {json_file}"):
            img = images["%g" % img_id]
            h, w, f = img["height"], img["width"], img["file_name"]

            bboxes = []
            segments = []
            for ann in anns:
                if ann["iscrowd"]:
                    continue
                # The COCO box format is [top left x, top left y, width, height]
                box = np.array(ann["bbox"], dtype=np.float64)
                box[:2] += box[2:] / 2  # xy top-left corner to center
                box[[0, 2]] /= w  # normalize x
                box[[1, 3]] /= h  # normalize y
                if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                    continue

                cls = ann["category_id"] - 1  # class
                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)
                # Segments
                if use_segments:
                    if len(ann["segmentation"]) > 1:
                        s = merge_multi_segment(ann["segmentation"])
                        s = (np.concatenate(s, axis=0) / np.array([w, h])).reshape(-1).tolist()
                    else:
                        s = [j for i in ann["segmentation"] for j in i]  # all segments concatenated
                        s = (np.array(s).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()
                    s = [cls] + s
                    if s not in segments:
                        segments.append(s)

            # Write
            with open((fn / f).with_suffix(".txt"), "a") as file:
                for i in range(len(bboxes)):
                    line = (*(segments[i] if use_segments else bboxes[i]),)  # cls, box or segments
                    file.write(("%g " * len(line)).rstrip() % line + "\n")



def convert_detecton2_to_coco(path2json, path2anno, path2save):
    """
        Convert detection predictions from detecton2 to COCO JSON format
        Args:
            path2json (str): json with detecton2 predict
            path2anno (str): json with coco annotation file  
            path2save (str): json save file
    """
    with open(path2json, 'r') as file:
        results = json.load(file)

    import threading
    def process_data(data, k, new_list):
        try:
            segm = mask_to_polygons(mask.decode(data['segmentation']))
            if len(segm[0])>21:
                segm = [segm[0][::3,:]]
            anno = {"id": k,
                    "image_id": data["image_id"], 
                    "category_id": data["category_id"], 
                    "segmentation": [segm[0].reshape(-1).tolist()]  ,
                    "area": polygone_area(segm[0][:,0], segm[0][:,1]),
                    "bbox": data["bbox"],
                    "iscrowd": 0,
                    }
            new_list.append(anno)
        except:
            pass
    annotations = []
    threads = []
    for k, item in tqdm(enumerate(results)):
        thread = threading.Thread(target=process_data, args=(item,k, annotations))
        thread.start()
        threads.append(thread)
    
    for thread in threads:
        thread.join()
    print(len(annotations))
    # for k, d in tqdm(enumerate(results)):
    #     segm = mask_to_polygons(mask.decode(d['segmentation']))
    #     if len(segm[0])>21:
    #         segm = [segm[0][::3,:]]
    #     anno = {"id": k,
    #             "image_id": d["image_id"], 
    #             "category_id": d["category_id"], 
    #             "segmentation": [segm[0].reshape(-1).tolist()]  ,
    #             "area": polygone_area(segm[0][:,0], segm[0][:,1]),
    #             "bbox": d["bbox"],
    #             "iscrowd": 0,
    #             }
    #     annotations.append(anno)

    with open(path2anno, 'r') as file:
        data_dict = json.load(file)
    data_dict['annotations'] = annotations

    with open(path2save, 'w') as file:
        json.dump(data_dict, file)

def coco2box(path2json, path2images, path2save):
    dataset = importer.ImportCoco(path=path2json, path_to_images=path2images)
    dataset.export.ExportToYoloV5(path2save)


if __name__ == "__main__":
    # conv = Yolo2Coco("/storage/reshetnikov/openpits/fold/Fold_0/test/",
    #                 "/storage/reshetnikov/openpits/fold/Fold_0/test/",
    #                 "/storage/reshetnikov/openpits/fold/Fold_0/anno_test.json")
    # conv.convert()

    parser = argparse.ArgumentParser(
        description="Convert labels to other coordinate system."
    )
    parser.add_argument(
        "--inpt_dir",
        type=str,
        help="Input directory with labels files or path to JSON COCO",
        default="/storage/reshetnikov/open_pits_merge/annotations/merge_add_sam/anno_merge2.json",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        help="Save directory with converted labels files.",
        default="/storage/reshetnikov/open_pits_merge/add_sam/max_line/",
    )

    parser.add_argument(
        "--image_dir",
        type=str,
        help="Save directory with converted labels files or path to anno json files,'detectron' converted ",
        default="/storage/reshetnikov/open_pits_merge/images",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="keypoint",
        help="'coco2obb' - Convert from coco json format to orientited bounding box in txt files; 'obb' - obb; 'yolo2coco'; 'coco2yolo'; 'keypoint' \n",
    )
    args = parser.parse_args()
    print(args, args.type)
    if args.type == "coco2obb":
        coco2obb(args.inpt_dir, args.save_dir)
    elif args.type == "obb":
        coco2obb_maxline(args.inpt_dir, args.save_dir)

    elif args.type == "yolo2coco":
        conv = Yolo2Coco(args.inpt_dir, args.image_dir, args.save_dir)
        conv.convert()
    elif args.type == "coco2yolo":
        coco2box(args.inpt_dir, args.image_dir, args.save_dir)
        # coco2obb("/storage/reshetnikov/open_pits_merge/annotations/annotations.json", '/storage/reshetnikov/open_pits_merge/obb')
    elif args.type == "keypoint":
        coco2box_keypoints(args.inpt_dir, args.save_dir, True)
    elif args.type == "segment":
        convert_coco_json(args.inpt_dir, args.save_dir, True)
    elif args.type == "detectron":
        convert_detecton2_to_coco(args.inpt_dir, args.image_dir, args.save_dir)
    else:
        print(f'{args.type} not found')