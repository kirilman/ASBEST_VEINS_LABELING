from re import X
import numpy as np
import os
import pandas as pd

from tqdm import main
import sys
sys.path.append("./")
from ._path import list_ext, list_images
from pathlib import Path
from PIL import Image
import json
from typing import List

def polygone_area(x,y):
    return 0.5 * np.array(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def yolo2coco(xc, yc, w, h, image_width, image_height):
    xc, w = xc*image_width,  w*image_width
    yc, h = yc*image_height, h*image_height
    xmin = xc - (w/2)
    ymin = yc - (h/2)
    return xmin,ymin, w, h

def segment2box(x_coords, y_coords):
    xl = np.min(x_coords)
    yl = np.min(y_coords)
    h  = np.max(y_coords) - yl
    w  = np.max(x_coords) - xl
    return xl, yl, w, h

def box2segment(box: List):
    """
        box: List coco format
    """
    x, y, h, w = box
    segment = []
    segment.append(x)
    segment.append(y)
    segment.append(x+w)
    segment.append(y)
    segment.append(x+w)
    segment.append(y+h)
    segment.append(x)
    segment.append(y+h)
    return segment

class Yolo2Coco():
    def __init__(self, path_label, path_image, path_save_json):
        self.path_label = path_label
        self.path_image = path_image
        self.image_paths = {Path(p).stem: os.path.join(self.path_image, p) for p in list_images(self.path_image)}
        self.label_paths = {Path(p).stem: os.path.join(self.path_label, p) for p in list_ext(self.path_label)}
        self.path_save_json = path_save_json

    def get_image_path(self, image_name):
        """
            Return:
            image_path: Path, path to image 
        """
        return self.image_paths[image_name]
    
    def get_label_path(self, file_name):
        return self.label_paths[file_name]

    def get_image_hw(self,  image_name):
        '''
            Get image height and weight
            Returns
            ----------
            height: int
            weight: int
        '''
        image_path = self.get_image_path(image_name)
        image = np.array(Image.open(image_path))
        height, weight,  = image.shape[0], image.shape[1]   # Важно 
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
            image_dict = {"id"           : img_id, 
                          "file_name"    : Path(f_path).name,
                          "width"        : w,
                          "height"       : h,
                          "licence"      : "",
                          "date_captured": 0,
                          }
            images[Path(f_path).stem]=image_dict
            img_id+=1
        return images

    def _collect_annotations(self, images_ids):
        '''
            YOLO.txt : cls, (x1,y1), (x2,y2) ...(xn,yn)
            Return
            -----------
            annotations: list[dict], annotation dict 
            categories : list[int], classes

        '''
        anno_id = 1
        annotations = []
        categories = []
        fname_list = list_ext(self.path_label, "txt")
        for _, fname in enumerate(fname_list):
            with open(self.get_label_path(fname.split(".")[0]), 'r') as f:     
                lines = f.readlines() 
            h, w = self.get_image_hw(Path(fname).stem)

            for line in lines:
                data = np.fromstring(line, sep=' ')
                if len(data) < 2:
                    continue
                o_cls, segment = data[0], data[1:]
                
                image_id = images_ids[Path(fname).stem]['id']
                if len(segment) == 4:
                    bbox = yolo2coco(segment[0], segment[1], segment[2], segment[3], w, h)
                    annotations.append({
                        "id": anno_id,
                        "image_id": image_id,
                        "category_id": int(o_cls) + 1,
                        "segmentation": [box2segment(bbox)],
                        "area": bbox[2]*bbox[3],
                        "bbox": bbox,
                        "iscrowd": 0,
                    })
                else:
                    x_coords, y_coords = segment[0::2]*w, segment[1::2]*h
                    coco_segment = []
                    for x,y in zip(x_coords,y_coords):
                        coco_segment.append(x)
                        coco_segment.append(y)

                    annotations.append({
                        "id": anno_id,
                        "image_id": image_id,
                        "category_id": int(o_cls) + 1,
                        "segmentation": [coco_segment],
                        "area": polygone_area(x_coords, y_coords),
                        "bbox": segment2box(x_coords, y_coords),
                        "iscrowd": 0,
                    })
                
                if not o_cls in categories:
                    categories.append(int(o_cls))
                anno_id+=1
            
        return annotations, categories

    def convert(self):
        images = self._collect_images()
        annotations, classes = self._collect_annotations(images)
        info = { "year": "2023", 
                 "version": "1.0",
                 "description": "Asbest dataset",
                 "contributor": "",
                 "url": "https://data.mendeley.com/v1/datasets/pfdbfpfygh/draft?preview=1",
                 "date_created": ""}
        licenses = [{"url": "https://data.mendeley.com/v1/datasets/pfdbfpfygh/draft?preview=1",
                    "id": 1,
                    "name": "openpits asbestos"}]
        class_names = { 1: "stone", 2:"asbest"}
        categories = [ {"id": _cls+1, "name": class_names[_cls+1], "supercategory": "" } for _cls in classes]
        data = {
            "info"       : info,
            "licenses"   : licenses,
            "images"     : list(images.values()),
            "annotations": annotations,
            "categories" : categories,
        }
        with open(self.path_save_json,'w') as f:
            json.dump(data,f)
        print("Save result to", self.path_save_json)

if __name__ == "__main__":


    conv = Yolo2Coco("../dataset/segmentation/train", "../dataset/segmentation/train", 
            "../dataset/segmentation/train/anno_train.json")
    conv.convert()


    conv = Yolo2Coco("../dataset/segmentation/test", "../dataset/segmentation/test", 
            "../dataset/segmentation/test/anno_test.json")
    conv.convert()