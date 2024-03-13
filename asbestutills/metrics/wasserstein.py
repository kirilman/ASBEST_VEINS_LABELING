import numpy as np
from ultralytics import YOLO
from scipy.stats import wasserstein_distance
from pathlib import Path
from parzen.statistic import collect_maxsize_obbox_for_prediction, collect_segmentation_maxsize
import pandas as pd


def var_confidence(path2model, path2source, path2save, conf_step, max_det):
    model = YOLO(model=path2model)
    for conf in np.arange(0.1, 0.8, conf_step):
        c = np.round(conf,2)
        name = "conf_{}".format(c)
        model.predict(source = path2source, imgsz = 720, conf = c, project = path2save, name = name, save_txt = True, max_det = max_det)

def wasserstein(path2pred, path2label):
    path2pred = Path(path2pred)
    results = {}
    for subdir in path2pred.iterdir():
        arr_msizes = []
        names = [x.stem for x in subdir.rglob('*.txt')]
        max_size_label = collect_segmentation_maxsize(path2label, names)
        conf_name = subdir.parts[-1]
        for ssub in subdir.iterdir():
            max_size = collect_maxsize_obbox_for_prediction(ssub, None)
            results[conf_name] = wasserstein_distance(max_size_label, max_size)
    df = pd.DataFrame([results]).T
    df.to_csv(path2pred / "conf.csv")

if __name__ == "__main__":
    path2save = '/storage/reshetnikov/runs/fraction_obb/var_conf/obb_l'
    path2label = '/storage/reshetnikov/open_pits_merge/merge_fraction/fraction_15/images/anno.json'
    var_confidence('/storage/reshetnikov/runs/fraction_obb/obb_frac_8l/weights/best.pt',
                   '/storage/reshetnikov/open_pits_merge/merge_fraction/fraction_15/fold_obb/Fold_0_out/test',
                   path2save,
                   0.05, max_det = 2500)
    wasserstein(path2save, path2label)