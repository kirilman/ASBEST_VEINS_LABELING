from torchmetrics.detection import MeanAveragePrecision
from torchmetrics.segmentation import MeanIoU
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch import tensor
import cv2 
import pandas as pd
from asbestutills._converter import yolo2coco,box2segment
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import gc
import os

def read_segmentation_labels(p):
    with open(p, 'r') as f:
        lines = f.readlines()
        return [np.fromstring(line, sep=' ').tolist() for line in lines]
    

def calculate_mertics(path2pred, path2label, format, cal_iou = False, scale = 640):
    data = read_segmentation_labels(path2pred)
    labels = torch.tensor(np.array([x[0]+1 for x in data], dtype = np.int32), dtype = torch.long) 
    scores = torch.tensor([1.0]*len(labels)) 
    boxes  = [x[1:] for x in data]
    mask_pred = np.zeros((len(data), scale, scale))
    for i, line in enumerate(data):
        if len(line)<5:
            print(f"{i}/{len(data)}, ",path2pred)
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
    data = read_segmentation_labels(path2label)
    labels = torch.tensor(np.array([x[0] for x in data], dtype = np.int32), dtype = torch.long) + 1
    boxes  = [x[1:] for x in data]
    mask_tgt = np.zeros((len(data), scale, scale))
    for i, line in enumerate(data):
        if len(line)<5:
            print(f"{i}/{len(data)}, ",path2label)
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
    
    metric = MeanAveragePrecision(box_format=format, iou_type='segm', max_detection_thresholds = [1,100, 1500], backend="faster_coco_eval")
    
    metric.update(preds, target)
    res = metric.compute()
    if cal_iou:
        res['iou'] = _cumpute_iou(preds, target) 
    res["file"] = Path(path2pred).name
    return res 


def compute_map(path2pred, path2anno, format='xywh', type = 'bbox', cal_iou = False, save_csv = True, path2save = None, max_workers = 10):
    """
        Compute mAP metric on files txt yolo format using torchmetrics
    Args:
        path2pred (str): path with txt prediction files in yolo format
        path2anno (str): path with txt targets files in yolo format
        format (str)   : format annotation in yolo format xywh or xyxy(segmentation)
        type (str)     : "bbox" - box prediction
                         "segm" - segmentation prediction x1y1x2y2..xnyn
        save_json (bool): save result to json
        path2save (str) : save path to json file
    """
    file_names = {Path(f).stem:f for f in list(Path(path2pred).glob("*.txt"))}
    file_names_target = {Path(f).stem:f for f in list(Path(path2anno).glob("*.txt"))}
    print(len(file_names.items()), len(file_names_target.items()))
    map = []
    assert type in ('bbox', 'segm'), f"Expected argument `type` to be one of ('bbox', 'segm') but got {type}"
    if type == 'bbox':
        for fname, fpath in tqdm(file_names.items()): 
            # print(fname, sep = '\n', flush=True)
            with open(fpath,"r") as f:
                data = np.loadtxt(f)
            if len(data) == 0:
                continue
            if len(data.shape) == 1:
                continue
                # data = data.reshape(-1,1)
            #если ключевые точки
            if data.shape[1]>5:
                print(f'{fpath} is keypoint prediction file')
                data = data[:,:5]
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
            if cal_iou:
                res['iou'] = _cumpute_iou(preds, target) 
            res["file"] = Path(fpath.name)
            map.append(res)   

    else:
        scale = 640
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(calculate_mertics, fpath, file_names_target[fname], format, cal_iou):fname for fname, fpath in tqdm(file_names.items())}
                for future in concurrent.futures.as_completed(futures):
                    res = future.result()
                    map.append(res) 
                    del future
                    gc.collect()
        except Exception as err:
            print(err)

    if save_csv:
        map_np = []
        if path2save == None:
            path2save = Path('map_result.csv')
        for item in map:
            values = [x.item() for x in list(item.values())[:-1]] + [item['file']]
            d = dict(zip(item.keys(),values))
            map_np.append(d)
            pd.DataFrame(map_np).to_csv(path2save,index = False)
        print(f'Save map result to {path2save}')
    return map

def _cumpute_iou(pred, target):
    """
    Args:
        preds: 
        target: 
        type (str)     : "bbox" - box prediction
                         "segm" - segmentation prediction x1y1x2y2..xnyn
    """
    if 'boxes' in target[0].keys():
        #-----------------------
        mask = np.zeros((640,640))
        for b in np.array(pred[0]['boxes']):
            box = b
            box = yolo2coco(b[0], b[1], b[2], b[3], 640, 640) #to coco
            box = np.array(box2segment(box), np.int32)
            res = cv2.fillPoly(mask, [box.reshape(-1,2)], color = 1)
        #target-----------
        targ = np.zeros((640,640))
        for b in np.array(target[0]['boxes']):
            box = yolo2coco(b[0], b[1], b[2], b[3], 640, 640) #to coco
            box = np.array(box2segment(box), np.int32)
            res = cv2.fillPoly(targ, [box.reshape(-1,2)], color = 1)
        # mask = xywh2xyxy(preds[0]['boxes'])
        # mask = preds[0]['boxes'].sum(axis=0)
        # targ = target[0]['boxes'].sum(axis=0)
    else:
        mask = np.array(pred[0]['masks'].sum(axis=0))
        targ = np.array(target[0]['masks'].sum(axis=0))
    
    targ[targ>1] = 1
    mask[mask>1] = 1
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
    targ = torch.tensor(targ, dtype=torch.long).unsqueeze(0)
    # print(mask.shape, targ.shape, mask.dtype, targ.dtype,mask.min(), mask.max(), targ.min(), targ.max())
    mean_iou = MeanIoU(num_classes=1)
    return mean_iou(mask, targ)

def predictionformat(coors):
    """
        coors: List[List] список координат из файла YOLO
        Формат предсказаний для txt файла YOLO
    """
    lens = []
    for c in coors:
       lens.append(len(c)) 
    n_min = np.min(lens)
    n_max = np.max(lens)
    if n_max - n_min > 0:
        return 'segm'
    elif n_max == 5:
        return 'box'
    elif n_max == 9:
        return 'obb'
    elif n_max == 13:
        return 'kpnt'
    
def culculate_map(path2dir = "/storage/reshetnikov/yolov8_rotate/stages/runs/splite_comp/var",
                      path2anno = "/storage/reshetnikov/open_pits_merge/merge_fraction/split/images/anno.json",
                    folder_exc = ['cascade_3x_old', 'box_v9'],
                    path2labels = {"obb":"/storage/reshetnikov/open_pits_merge/merge_fraction/split/obb",
                                   "segm":"/storage/reshetnikov/open_pits_merge/merge_fraction/split/segment",
                                   "box":"/storage/reshetnikov/open_pits_merge/merge_fraction/split/yolo"}):
    path2dir = Path(path2dir)
    path2anno = Path(path2anno)
    dirs = sorted(list(Path(path2dir).glob("*/")))
    dirs = dirs[0],dirs[-2],dirs[5]
    print(dirs)
    dirs = [dirs[-2]]
    for _dir in dirs:
        if _dir.parts[-1] in folder_exc:
            continue
        for sub_dir in _dir.glob("*"):
            if sub_dir.is_dir():
                for p1, sdirs, files in os.walk(sub_dir):
                    if 'labels' in sdirs: #заходим в labels
                        p2pred = Path(p1)/sdirs[0]
                        f_labels = list(p2pred.glob("*.txt"))
                        coors = read_segmentation_labels(f_labels[0])
                        ft = predictionformat(coors)
                        path2label = path2labels[ft]
                        if ft == 'box':
                            ft = 'b' + ft
                        elif ft =='obb':
                            ft = 'segm'
                        conf_name = sub_dir.parts[-1]
                        print(sub_dir/f"map_{conf_name}.csv")
                        if ft == 'bbox':
                            res = compute_map(p2pred, path2label, "xywh", ft, True, sub_dir/f"map_{conf_name}.csv")
                        else:
                            res = compute_map(p2pred, path2label, "xyxy", ft, True, sub_dir/f"map_{conf_name}.csv", max_workers = 8)
            # pathlabel = datasources[nfold] 
            # print( dir/sdir.name/f"map_{sdir.name}.csv")
            # res = compute_map(path2pred, pathlabel, "xyxy","segm", True, dir/sdir.name/f"map_{sdir.name}.csv")

if __name__ == '__main__':
    map = compute_map("/storage/reshetnikov/yolov8_rotate/stages/runs/splite_comp/var/box_v10x/conf_0.25/labels/",
                    "/storage/reshetnikov/open_pits_merge/merge_fraction/split/yolo/", format='xywh', type='bbox', 
                    path2save='map_025.csv')

    # map = compute_map("/storage/reshetnikov/yolov8_rotate/stages/runs/splite_comp/var/obb/conf_0.25/labels/",
    #                 "/storage/reshetnikov/open_pits_merge/merge_fraction/split/obb/", format='xyxy', type='segm', 
    #                 path2save='/storage/reshetnikov/yolov8_rotate/stages/runs/splite_comp/var/obb/map_025.csv')

    print(map)