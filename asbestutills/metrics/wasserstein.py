import numpy as np
from ultralytics import YOLO
from scipy.stats import wasserstein_distance
from pathlib import Path
from parzen.statistic import collect_maxsize_obbox_for_prediction, collect_segmentation_maxsize, collect_bbox_maxsize
import pandas as pd
from asbestutills._converter import Yolo2Coco

def var_confidence(path2model, path2source, path2save, conf_step, max_det):
    model = YOLO(model=path2model)
    for conf in np.arange(0.05, 0.85, conf_step):
        c = np.round(conf,2)
        name = "conf_{}".format(c)
        model.predict(source = path2source, save = False, imgsz = 640, conf = c, project = path2save, name = name, save_txt = True, max_det = max_det)
        if model.task == 'segment':
            conv = Yolo2Coco(Path(path2save) / f"conf_{c}/labels/", path2source, Path(path2save) / f"conf_{c}/labels/predict.json")
            conv.convert()

def wasserstein(path2pred, path2label):
    path2pred = Path(path2pred)
    results = {}
    for subdir in path2pred.iterdir():
        if not subdir.is_dir():
            continue
        arr_msizes = []
        names = [x.stem for x in subdir.rglob('*.txt')]
        max_size_label = collect_segmentation_maxsize(path2label, names)
        conf_name = subdir.parts[-1]
        for ssub in subdir.iterdir():
            if not ssub.is_dir():
                continue
            if 'box' in str(subdir):
                max_size = collect_bbox_maxsize(ssub, None)
            elif 'segm' in str(subdir):
                json_files  = [x for x in subdir.rglob('*.json')]
                print(json_files)
                max_size = collect_segmentation_maxsize(json_files[0], names)
            else:
                max_size = collect_maxsize_obbox_for_prediction(ssub, None)
            
            results[conf_name] = wasserstein_distance(max_size_label, max_size)
    df = pd.DataFrame([results]).T
    df.to_csv(path2pred / "conf.csv")

if __name__ == "__main__":
    # path2save = '/storage/reshetnikov/runs/splite/var_conf_for_splite_val/obb_8x'
    # path2label = '/storage/reshetnikov/open_pits_merge/merge_fraction/split/images/anno.json'
    # var_confidence('/storage/reshetnikov/runs/fraction_obb/obb_splite_8x/weights/best.pt',
    #                '/storage/reshetnikov/open_pits_merge/merge_fraction/split/train_split/val/',
    #                path2save,
    #                0.05, max_det = 2500)
    # wasserstein(path2save, path2label)


    # path2save = '/storage/reshetnikov/runs/splite/var_conf_for_splite_val/box_8x'
    # path2label = '/storage/reshetnikov/open_pits_merge/merge_fraction/split/images/anno.json'
    # var_confidence('/storage/reshetnikov/runs/fraction_obb/box_splite_8x/weights/best.pt',
    #                '/storage/reshetnikov/open_pits_merge/merge_fraction/split/train_split/val/',
    #                path2save,
    #                0.05, max_det = 2500)
    # wasserstein(path2save, path2label)

    # path2save = '/storage/reshetnikov/yolov8_rotate/stages/runs/splite_comp/var/box'
    # path2label = '/storage/reshetnikov/open_pits_merge/merge_fraction/split/images/anno.json'

    # var_confidence('/storage/reshetnikov/runs/fraction_obb/box_splite_8x/weights/best.pt',
    #                '/storage/reshetnikov/open_pits_merge/merge_fraction/split/train_split/test/',
    #                path2save,
    #                0.05, max_det = 2500)
    # wasserstein(path2save, path2label)

    path2save = '/storage/reshetnikov/yolov8_rotate/stages/runs/splite_comp/var/segm'
    path2label = '/storage/reshetnikov/open_pits_merge/merge_fraction/split/images/anno.json'
    path2save = '/storage/reshetnikov/yolov8_rotate/stages/runs/splite_comp/validation/segm_v9'
    # var_confidence('/storage/reshetnikov/runs/fraction_obb/segm_splite_8x/weights/best.pt',
    #                '/storage/reshetnikov/open_pits_merge/merge_fraction/split/train_split/test/',
    #                path2save,
    #                0.05, max_det = 2500)
    
    wasserstein(path2save, path2label)