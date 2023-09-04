import numpy as np
import os
import pandas as pd

from tqdm import main

from _path import list_ext, list_images
# from ._path import list_ext, list_images

from pathlib import Path
from PIL import Image
import json
from typing import List
from pycocotools.coco import COCO
from utils.geometry import coords_main_line, coords_other_line, coords_obb, coords_max_line, distance, position
import argparse


def polygone_area(x,y):
    return 0.5 * np.array(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def yolo2coco(xc, yc, w, h, image_width, image_height):
    xc, w = xc*image_width,  w*image_width
    yc, h = yc*image_height, h*image_height
    xmin = xc - (w/2)
    ymin = yc - (h/2)
    return xmin, ymin, w, h

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
    x, y, w, h = box
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

def ellipse_parameters(x, y):
    from skimage.measure import EllipseModel
    a_points = np.array([x, y]).T
    ell = EllipseModel()
    ell.estimate(a_points)
    return ell.params

def clear_negative_values(x):
    if x<=0:
        return 0.001
    else:
        return x
    
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


def coco2obb(path2json, path2save):
    """
        Convert coco polygon coordinates to obb format coordinates. Save the result in *.txt files in path2save directory
    Args:
        path2json (str): json with coco format
        path2save (str): save directory
    """
    coco = COCO(path2json)
    frame = pd.DataFrame(coco.anns).T
    df_image = pd.DataFrame(coco.imgs).T
    image_dict = df_image.T.to_dict()
    print(image_dict)
    fname = str(df_image[df_image.id ==  frame.iloc[0].image_id]['file_name'].values[0].split('.')[0])    
    file_out = open(Path(path2save) / (fname + ".txt" ), 'w')    

    for k, row in frame.iterrows():
        IMAGE_W = image_dict[row.image_id]['width']
        IMAGE_H = image_dict[row.image_id]['height']
        
        try:
            x_coords = np.array(row.segmentation[0][::2])#/IMAGE_W
            y_coords = np.array(row.segmentation[0][1::2])#/IMAGE_H
            xc, yc, a, b, theta = ellipse_parameters(x_coords, y_coords)
        except:
            print("Failed to obtain ellipse parameters for ", row)
            continue
        x1, y1, x2, y2 = coords_main_line(xc, yc, a, theta)
        x1, y1, x2, y2 = coords_other_line(xc, yc, b, theta) # b axes
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
        cls_id = 'stone'
        current_fname = str(df_image[df_image.id ==  row.image_id]['file_name'].values[0].split('.')[0])
        if fname == current_fname:
            line = "{:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {} 0\n".format( ox1, oy1, ox2, oy2, ox3, oy3, ox4, oy4, cls_id)
            file_out.write(line)
            
        else:
            file_out.close()
            fname = current_fname
            file_out = open(Path(path2save) / (fname + ".txt" ), 'a')    
            line = "{:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {} 0\n".format( ox1, oy1, ox2, oy2, ox3, oy3, ox4, oy4, cls_id)
            file_out.write(line)
    file_out.close()
    return True




def guide_vector(x1,y1,x2,y2):
    x = [x1, x2]
    y = [y1, y2]
    coefficients = np.polyfit(x, y, 1)    
    return coefficients[0], 1, coefficients[1]

def distance_to_perp(A, B, C, x, y):
    return abs(A*x+B*y+C)/np.sqrt(A**2+B**2)

def line_from_points(p1, p2): 
    a = p2[1] - p1[1]
    b = p2[0] - p1[0]
    c = - a*(p1[0]) + b*(p1[1])
    return a, -b, c

def coco2obb_maxline(path2json, path2save):
    coco = COCO(path2json)
    frame = pd.DataFrame(coco.anns).T
    df_image = pd.DataFrame(coco.imgs).T
    image_dict = df_image.T.to_dict()
    fname = str(df_image[df_image.id ==  frame.iloc[0].image_id]['file_name'].values[0].split('.')[0])    
    file_out = open(Path(path2save) / (fname + ".txt" ), 'w')    
    df_image = pd.DataFrame(coco.imgs).T
    print(df_image.head(15))
    f_out = open('/home/reshetnikov/asbest/yolov8_segmentation/notebook/a.txt', 'w')
    for k, row in frame.iterrows():

        IMAGE_W = image_dict[row.image_id]['width']
        IMAGE_H = image_dict[row.image_id]['height']
        try:
            x_coords = np.array(row.segmentation[0][::2])#/IMAGE_W
            y_coords = np.array(row.segmentation[0][1::2])#/IMAGE_H
        except:
            print("Failed to obtain x_coords, y_coords for ", row)
            continue
        
        x1, y1, x2, y2 = coords_max_line(x_coords, y_coords)
        Points = [(x,y) for x,y in zip(x_coords, y_coords)]
        y_min_point, y_max_point = 0, 0
        ymin = 10000
        ymax = 0
        xmid = x1 + (x2-x1)/2
        ymid = y1 + (y2-y1)/2
        max_dist_left  = 0
        max_dist_right = 0
        n = 0
        for point in Points:
            if position(point[0],point[1], x1, y1, x2, y2)>0:
                main_A, main_B, main_C = line_from_points((x1, y1), (x2, y2))
                d = distance_to_perp(main_A, main_B, main_C, point[0],point[1])
                if abs(d) > max_dist_right:
                    max_dist_right = d
                    y_max_point = point
            else:
                main_A, main_B, main_C = line_from_points((x1, y1), (x2, y2))
                d = distance_to_perp(main_A, main_B, main_C, point[0],point[1])
                if abs(d) > max_dist_left:
                    max_dist_left = d
                    y_min_point = point
                    n+=1
        if y_min_point == 0 or y_max_point == 0:
            continue
            # d = distance((xmid, ymid),point)
            # try:
            #     if position(point[0],point[1], x1, y1, x2, y2)>0:
            #         main_A, main_B, main_C = guide_vector(x1, y1, x2, y2)
            #         d = distance_to_perp(main_A, main_B, main_C, point[0],point[1])
            #         if abs(d) > max_dist_right:
            #             max_dist_right = d
            #             y_max_point = point
            #             n+=1
            #     else:
            #         main_A, main_B, main_C = guide_vector(x1, y1, x2, y2)
            #         d = distance_to_perp(main_A, main_B, main_C, point[0],point[1])
            #         if abs(d) > max_dist_left:
            #             max_dist = d
            #             y_min_point = point
            #             n+=1
            # except:
            #     print(d)
            # d = position(point[0],point[1], x1, y1, x2, y2)

            # if d > 0:
            #     if point[1] > ymax:
            #         y_max_point = point
            #         ymax = point[1]
            # elif d < 0:
            #     if point[1] < ymin:
            #         y_min_point = point
            #         ymin = point[1]
            # elif d == 0:
            #     continue

        # print(y_min_point, y_max_point)
        # try:
            # print(row.image_id, x1, y1, x2, y2, *y_min_point, *y_max_point)
        # print(row.image_id, x1, y1, x2, y2, *y_max_point)
        if row.image_id == 3:
            print(row.image_id, x1, y1, x2, y2, *y_max_point)
            f_out.write("{} {} {} {} {} {} {} {}\n".format(x1, y1, x2, y2, *y_max_point, *y_min_point))

        # except:
            # pass
    
        if row.image_id > 12:

            break
    f_out.close()
if __name__ == "__main__":


    # conv = Yolo2Coco("../dataset/segmentation/train", "../dataset/segmentation/train", 
    #         "../dataset/segmentation/train/anno_train.json")
    # conv.convert()


    # conv = Yolo2Coco("../dataset/segmentation/test", "../dataset/segmentation/test", 
    #         "../dataset/segmentation/test/anno_test.json")
    # conv.convert()
    # conv = Yolo2Coco("/storage/reshetnikov/openpits/fold/Fold_0/train/", 
    #              "/storage/reshetnikov/openpits/fold/Fold_0/train/", 
    #              "/storage/reshetnikov/openpits/fold/Fold_0/anno_train.json")
    # conv.convert()
    # conv = Yolo2Coco("/storage/reshetnikov/openpits/fold/Fold_0/test/", 
    #                 "/storage/reshetnikov/openpits/fold/Fold_0/test/", 
    #                 "/storage/reshetnikov/openpits/fold/Fold_0/anno_test.json")
    # conv.convert()

    parser = argparse.ArgumentParser(description='Convert labels to other coordinate system.')
    parser.add_argument('--inpt_dir', type=str,
                        help='Input directory with files.', default = "/storage/reshetnikov/open_pits_merge/annotations/annotations.json")
    
    parser.add_argument('--save_dir', type=str, help='Save directory with converted labels files.', 
                        default= '/storage/reshetnikov/open_pits_merge/obb')
    
    parser.add_argument('--type', type = str , default='obb_maxline', help="'coco2obb' - Convert from coco json format to orientited bounding box in txt files")
    args = parser.parse_args()
    print(args, args.type)
    if args.type == 'coco2obb':
        coco2obb(args.inpt_dir, args.save_dir)
    elif args.type == 'obb_maxline':
        coco2obb_maxline(args.inpt_dir, args.save_dir)
        # coco2obb("/storage/reshetnikov/open_pits_merge/annotations/annotations.json", '/storage/reshetnikov/open_pits_merge/obb')