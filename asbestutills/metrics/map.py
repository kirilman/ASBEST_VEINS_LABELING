from torchmetrics.detection import MeanAveragePrecision
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch import tensor
import cv2 
import pandas as pd

def read_segmentation_labels(p):
    with open(p, 'r') as f:
        lines = f.readlines()
        return [np.fromstring(line, sep=' ').tolist() for line in lines]

def compute_map(path2pred, path2anno, format='xywh', type = 'bbox', save_json = True, path2save = None):
    """
        Compute mAP matric on files txt yolo format using torchmetrics
    Args:
        path2pred (str): path with txt prediction files in yolo format
        path2anno (str): path with txt targets files in yolo format
        format (str)   : format annotation in yolo format xywh or xyxy(segmentation)
        type (str)     : "box" - box prediction
                         "segm" - segmentation prediction x1y1x2y2..xnyn
        save_json (bool): save result to json
        path2save (str) : save path to json file
    """
    file_names = {Path(f).stem:f for f in list(Path(path2pred).glob("*.txt"))}
    file_names_target = {Path(f).stem:f for f in list(Path(path2anno).glob("*.txt"))}
    print(file_names)
    map = []
    if type == 'bbox':
        for fname, fpath in tqdm(file_names.items()): 
            
            with open(fpath,"r") as f:
                data = np.loadtxt(f)
               
            labels = torch.tensor((data[:,0]+1).astype(np.int32), dtype = torch.long) 
            scores = torch.tensor([1.0]*len(labels)) 
            boxes  = torch.tensor(data[:,1:])
            preds = [
              dict(
                boxes=boxes,
                scores=scores,
                labels=labels,
              )
            ]
            with open(file_names_target[fname],"r") as f:
                data = np.loadtxt(f)
                
            labels = torch.tensor((data[:,0]).astype(np.int32),  dtype = torch.long)
            scores = torch.tensor([1.0]*len(labels))
            boxes  = torch.tensor(data[:,1:])
            
            target = [
              dict(
                boxes=boxes,
                labels=labels,
              )
            ]
            metric = MeanAveragePrecision(box_format=format, max_detection_thresholds = [1,100, 1500])
            metric.update(preds, target)

            metric.update(preds, target)
            res = metric.compute()
            res["file"] = Path(fpath.name) 
            map.append(res)   

    else:
        for fname, fpath in tqdm(file_names.items()): 
            try:
                #predicts------------------
                data = read_segmentation_labels(fpath)
                labels = torch.tensor(np.array([x[0]+1 for x in data], dtype = np.int32), dtype = torch.long) 
                scores = torch.tensor([1.0]*len(labels)) 
                boxes  = [x[1:] for x in data]
                scale = 640
                mask_pred = np.zeros((len(data), scale, scale))
                for i, line in enumerate(data):
                    if len(line)<5:
                        print(f"{i}/{len(data)}, ",fpath)
                        continue
                    coords = np.array(line[1:]).reshape(-1,2)
                    coords*=scale
                    coords = coords.astype(np.int32)
                    mask = np.zeros((scale, scale))
                    cv2.fillPoly(mask, [coords], 1)
                    if np.sum(mask)>10:
                        mask = mask.astype(bool)
                    else:
                        mask = np.zeros((scale, scale))
                    mask_pred[i] = mask
        
                
                preds = [
                dict(
                    masks=tensor(mask_pred, dtype=torch.bool),
                    scores=scores,
                    labels=labels,
                )
                ]
                #targets------------------
                data = read_segmentation_labels(file_names_target[fname])
                labels = torch.tensor(np.array([x[0] for x in data], dtype = np.int32), dtype = torch.long) + 1
                boxes  = [x[1:] for x in data]
                mask_tgt = np.zeros((len(data), scale, scale))
                for i, line in enumerate(data):
                    if len(line)<5:
                        print(f"{i}/{len(data)}, ",fpath)
                        continue
                    coords = np.array(line[1:]).reshape(-1,2)
                    coords*=scale
                    coords = coords.astype(np.int32)
                    mask = np.zeros((scale, scale))
                    cv2.fillPoly(mask, [coords], 1)
                    if np.sum(mask)>10:
                        mask = mask.astype(bool)
                    else:
                        mask = np.zeros((scale, scale))
                    mask_tgt[i] = mask
                    
                target = [
                dict(
                    masks=tensor(mask_tgt, dtype=torch.bool),
                    labels=labels,
                )
                ]

                metric = MeanAveragePrecision(box_format=format, iou_type='segm', max_detection_thresholds = [1,100, 1500])
                metric.update(preds, target)
                metric.update(preds, target)
                res = metric.compute()
                res["file"] = Path(fpath.name) 
                map.append(res)   
            except Exception as err:
                print(err)
    if save_json:
        map_np = []
        if path2save == None:
            path2save = Path('map_result.csv')
        for item in map:
            values = [x.item() for x in list(item.values())[:-1]] + [item['file']]
            d = dict(zip(item.keys(),values))
            map_np.append(d)
            pd.DataFrame(map_np).to_csv(path2save,index = False)
    return map


map = compute_map("/storage/reshetnikov/yolov8_rotate/stages/runs/splite_comp/validation/obb/conf_0.25/labels/",
                  "/storage/reshetnikov/open_pits_merge/merge_fraction/split/obb/", format='xyxy', type='segm', 
                   path2save='/storage/reshetnikov/yolov8_rotate/stages/runs/splite_comp/validation/obb/map_025.csv')

print(map)