import torch
import torchvision
import sys
import pandas as pd
import os
# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import numpy as np 
from pathlib import Path
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from tqdm import tqdm
import json
import cv2

class Sam_processor:
    def __init__(self, 
                 sam_weight = "/storage/reshetnikov/sam/sam_vit_l_0b3195.pth",
                 model_type = "vit_l",
                 device = "cuda:2",
                 points_per_side = 24):
        sam = sam_model_registry[model_type](checkpoint=sam_weight)
        sam.to(device = device)
        mask_generator = SamAutomaticMaskGenerator(
                            model=sam,
                            points_per_side=points_per_side,
                            pred_iou_thresh=0.9,
                            stability_score_thresh=0.92,
                            crop_n_layers=1,
                            # crop_overlap_ratio = 0.5,
                            # crop_n_points_downscale_factor=2,
                            min_mask_region_area=1000,  # Requires open-cv to run post-processing
                        )
    

from pycocotools import mask as mask_utils

def sharpen_kernel(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(image, -1, kernel)

def runs_to_coords(runs, size):
    """
    Преобразует uncompressed RLE-runs в список (x, y) координат объекта.
    Работает за O(N_runs), память — только под точки объекта.
    """
    h, w = size
    coords = []
    idx = 0  # текущий индекс в 1D (column-major: x + y*w)
    # runs: [run0_bg, run1_obj, run2_bg, run3_obj, ...]
    for i, length in enumerate(runs):
        if i % 2 == 1:  # нечётные — объект
            # Добавляем length пикселей, начиная с idx
            for j in range(length):
                pos = idx + j
                x = pos % w          # col
                y = pos // w         # row
                coords.append((x, y))
        idx += length
    return np.array(coords, dtype=np.int32)

def mask_to_coco_annotation(
    mask: np.ndarray,
    image_id: int,
    annotation_id: int,
    category_id: int,
    iscrowd: int = 0
) -> dict:
    # Ensure mask is in uint8 with 0/1
    if mask.dtype == bool:
        mask = mask.astype(np.uint8)
    elif mask.max() > 1:
        mask = (mask > 0).astype(np.uint8)

    # Encode mask to RLE (COCO format: column-major/F-order)
    rle = mask_utils.encode(np.asfortranarray(mask))
    rle["counts"] = rle["counts"].decode("utf-8")  # COCO JSON requires string counts
    
    # Compute bounding box [x, y, width, height] (xywh)
    # Using pycocotools' built-in utility
    bbox = mask_utils.toBbox(rle).tolist()  # [x, y, w, h]
    
    # Compute area
    area = int(mask_utils.area(rle))

    ann = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": rle,  # {"size": [h,w], "counts": "..."}
        "bbox": bbox,
        "area": area,
        "iscrowd": iscrowd
    }
    return ann

def sam_annotate(path2data, path2save):
    path2data = Path(path2data)
    sam2_checkpoint = "/storage/reshetnikov/disser/notebooks/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"  # or _b+.yaml, _s.yaml, etc.

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")

    mask_generator = SAM2AutomaticMaskGenerator(sam2_model, pred_iou_thresh=0.7, 
                                           min_mask_region_area=20,
                                           points_per_side=96,
                                           stability_score_thresh = 0.94)

    with open('/storage/reshetnikov/openpits/annotations/instances_default.json', 'r') as f:
        annotation = json.load(f)
    annotation['annotations'][0]
    anno_json = annotation.copy()
    anno_json['categories']

    f_images = sorted(list(path2data.rglob("*")))
    print(len(f_images))
    images = {}
    img_id = 1
    for f_path in f_images:
        image = cv2.imread(f_path)
        h = image.shape[0]
        w = image.shape[1]
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
    images = list(images.values()) 

    anno_id = 0
    anno = []
    for img_id, f in tqdm(enumerate(f_images)):
        image = cv2.imread(f)
        h_orig, w_orig, _ = image.shape
        image = cv2.resize(image,(int(w_orig/3),int(h_orig/3)))
        print(h_orig, w_orig)
        sharpened = sharpen_kernel(image)
        masks = mask_generator.generate(sharpened)   
        for ids, mask in enumerate(masks):
            m_resize = cv2.resize(mask['segmentation'].astype(np.uint8), (w_orig, h_orig), interpolation=cv2.INTER_NEAREST).astype(bool)
            if m_resize.sum()/(h_orig * w_orig) > 0.4:
                continue
            r = mask_to_coco_annotation(m_resize,img_id+1,anno_id,1)
            anno_id+=1
            anno.append(r)

    
    anno_json['images'] = images
    anno_json['annotations'] = anno

    with open(path2data / 'anno_sam.json','w') as f:
        json.dump(anno_json,f)


def cocorle_2cocopoly(path2json, path2save):
    def rle_polyxy(rle):
        rle_code = mask_utils.frPyObjects(rle, rle["size"][0], rle["size"][1])
        mask = mask_util.decode(rle_code)  # shape: (880, 1400), dtype=np.uint8, values 0/1
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        epsilon = 0.001 * cv2.arcLength(contour, closed=True)
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)
        poly_xy = approx.squeeze().flatten().tolist()
        return poly_xy

    with open(path2json, 'r') as f:
        anno = json.load(f)

    annotation = []
    for a in anno['annotations']:
        new_item = a.copy()
        if not isinstance(a['segmentation'], list):
            poly = rle_polyxy(a['segmentation'])
            new_item['segmentation'] = [poly]
            annotation.append(new_item)
        else:
            annotation.append(new_item)

    anno['annotations'] = annotation
    with open(path2save ,'w') as f:
        json.dump(anno,f)



if __name__ == '__main__':
    path2save = '/storage/reshetnikov/rock_other/gransostav/05.12/05 12 blog/'
    sam_annotate(path2save, path2save)