from sklearn.model_selection import KFold
import yaml
import numpy as np
from pathlib import Path
from ._coords_transition import yolo2xyxy, xywh2xyxy
from ._path import list_ext, list_images, _cp_file_list
# from _path import list_ext, list_images, _cp_file_list
import shutil
import cv2
from ultralytics import YOLO
from tqdm import tqdm
import pandas as pd


RANDOM_STATE = 101


class ImageWithBoxs:
    def __init__(self, path2image, box_coords):
        """
        box_coords: x1,y1,x2,y2 normalize
        """
        self.image = cv2.imread(path2image)
        self.box_coords = np.array(box_coords)
        h, w = self.image_height_width()
        self.box_coords[:, [0, 2]] *= w
        self.box_coords[:, [1, 3]] *= h

    def image_height_width(self):
        return self.image.shape[:2]

    def get_image_slice(self, box_id):
        x1, y1, x2, y2 = np.ceil(self.box_coords[box_id]).astype(int)
        return self.image[y1:y2, x1:x2]

    def get_slices(self):
        slices = {}
        for i, box in enumerate(self.box_coords):
            x1, y1, x2, y2 = np.ceil(box).astype(int)
            slices[i] = {"image": self.image[y1:y2, x1:x2, :], "box": [x1, y1, x2, y2]}
        return slices

def split_with_file(path2image: str,
                    path2label: str,
                    path2filesplit: str,
                    path2save: str):
    path2image = Path(path2image)
    path2label = Path(path2label)
    frame = pd.read_csv(path2filesplit)
    train_names = [f.split('.')[0] for f in frame[frame.train == 1].name.to_list()]
    val_names = [f.split('.')[0] for f in frame[frame.val == 1].name.to_list()]
    test_names = [f.split('.')[0] for f in frame[frame.test == 1].name.to_list()]
    f_images = [f for f in list_images(path2image)] 
    f_labels = [f for f in list_ext(path2label)]
    print(f_images)
    # print(f_labels)
    print(train_names)

    path2save = Path(path2save)
    if path2save.exists():
        shutil.rmtree(path2save)
    path2save.mkdir()
    print([path2image / f for f in f_images if Path(f) in train_names])
    _cp_file_list(path2save,"train/", [path2image / f for f in f_images if Path(f).stem in train_names])
    _cp_file_list(path2save,"train/", [path2label / f for f in f_labels if Path(f).stem in train_names])

    _cp_file_list(path2save,"val/", [path2image / f for f in f_images if Path(f).stem in val_names])
    _cp_file_list(path2save,"val/", [path2label / f for f in f_labels if Path(f).stem in val_names])

    _cp_file_list(path2save,"test/", [path2image / f for f in f_images if Path(f).stem in test_names])
    _cp_file_list(path2save,"test/", [path2label / f for f in f_labels if Path(f).stem in test_names])
    yaml_config = {
            "names": ["stone"],
            "nc": 1,
            "path": str(path2save),
            "train": "./train",
            "val": "./val",
            "test": "./test"
        }

    with open(path2save / "config.yaml", "w") as file:
        yaml.dump(yaml_config, file)

def k_fold_split_yolo(
    path2label: str,
    path2image: str,
    path_save_fold: str,
    number_fold: int = 4,
    random_seed=42,
):
    path2label = Path(path2label)
    path2image = Path(path2image)
    l_labels = sorted(list_ext(path2label), key=lambda x: x.split(".")[0])
    l_images = list_images(path2image)
    l_images = sorted(
        filter(lambda x: Path(x).stem in [Path(x).stem for x in l_labels], l_images)
    )
    if len(l_labels) != len(l_images):
        print(
            "The length of arrays does not match {} and {}".format(
                len(l_labels), len(l_images)
            )
        )
        for a, b in zip(l_images, l_labels):
            print(a, " ", b)

    else:
        pass

    path_save_fold = Path(path_save_fold)
    if path_save_fold.exists():
        shutil.rmtree(path_save_fold)
    path_save_fold.mkdir()

    kfold = KFold(number_fold, shuffle=True, random_state=random_seed)
    for kf, (train_indxs, test_indxs) in enumerate(kfold.split(l_labels)):
        name = "Fold_{}".format(kf)
        path_2_fold = path_save_fold / name
        path_2_fold.mkdir()

        train_images = [
            path2image / fname for fname in [l_images[i] for i in train_indxs]
        ]
        _cp_file_list(path_2_fold, "train/", train_images)

        test_images = [path2image / name for name in [l_images[i] for i in test_indxs]]
        _cp_file_list(path_2_fold, "test/", test_images)

        train_labels = [
            path2label / name for name in [l_labels[i] for i in train_indxs]
        ]
        _cp_file_list(path_2_fold, "train/", train_labels)

        test_labels = [path2label / name for name in [l_labels[i] for i in test_indxs]]
        _cp_file_list(path_2_fold, "test/", test_labels)

        yaml_config = {
            "names": ["stone"],
            "nc": 1,
            "path": str(path_2_fold),
            "train": "./train",
            "val": "./test",
        }

        with open(path_2_fold / "config.yaml", "w") as file:
            yaml.dump(yaml_config, file)
        print(kf, len(train_indxs), len(test_indxs), path_2_fold)
    


def is_valid_slice(image, model, conf=0.6):
    image = np.ascontiguousarray(image)
    res = model(image, conf=conf, task="detect")
    boxes = res[0].boxes.xyxyn.detach().cpu().numpy()
    if len(boxes) == 1:
        x1, y1, x2, y2 = boxes[0, :]
        s = (x2 - x1) * (y2 - y1)
        if s > 0.8:
            return True
    else:
        for box in boxes:
            x1, y1, x2, y2 = box[:4]
            s = (x2 - x1) * (y2 - y1)
            if s > 0.6:
                return True
    return False


def neyral_filter(image_with_boxes, model, valid_score):
    """
    Get the indexes of the array bbox_coords for which the neural network has found an object
    """
    indexs = []
    for k, img in image_with_boxes.get_slices().items():
        test_image = np.ascontiguousarray(img["image"])
        if is_valid_slice(test_image, model, valid_score):
            indexs.append(k)
    return indexs


def iou_value(a, b):
    """
    a: List [x1, y1, x2, y2]
    b: List [x1, y1, x2, y2]
    """
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])

    iou_x1 = np.maximum(a[0], b[0])
    iou_y1 = np.maximum(a[1], b[1])
    iou_x2 = np.minimum(a[2], b[2])
    iou_y2 = np.minimum(a[3], b[3])

    iou_w = iou_x2 - iou_x1
    iou_h = iou_y2 - iou_y1
    # print(iou_w, iou_h)
    if iou_w > 0 and iou_h > 0:
        area_iou = iou_w * iou_h
        iou = area_iou / (area_a + area_b - area_iou)
    else:
        iou = 0
    return iou


def non_max_suppression(boxes, scores, min_score=0.5, iou_score=0.4):
    """
    iou_score: float чем меньше, тем меньше остается боксов в boxes
    return: indexs
    """
    if isinstance(boxes, list):
        boxes = np.array(boxes)
    assert boxes.shape[1] == 4, "Input boxes has [x, y, w, h] format"
    indexs = cv2.dnn.NMSBoxes(boxes, scores, min_score, iou_score)
    return indexs


def merge_yolo_annotation(
    path2label: str,
    path2image: str,
    path2other: str,
    path2save: str,
    iou_tresh: float = 0.5,
    merge: bool = True,
):
    """
    Merge yolo annotations from path2label path and path2other path
    for labels whose intersection is less than the value of iou_tresh
    """
    path2other = Path(path2other)
    path2save = Path(path2save)
    f_images = {x.stem: x for x in Path(path2image).glob("*")}
    for path in Path(path2label).glob("*.txt"):
        with open(path, "r") as f1:
            lines = f1.readlines()
        with open(path2other / path.name, "r") as f2:
            other_lines = set(f2.readlines())
        count_del = 0
        for main_line in lines:
            a = (np.fromstring(main_line, dtype=float, sep=" ")[1:]).reshape(1, -1)
            box_1 = xywh2xyxy(a)
            copy_set = other_lines.copy()
            for o_line in other_lines:
                box_2 = xywh2xyxy(
                    (np.fromstring(o_line, dtype=float, sep=" ")[1:]).reshape(1, -1)
                )
                if (
                    iou_value(box_1[0, :4], box_2[0, :4]) > iou_tresh
                ):  # удаляем box у которого пересечение с основным box  если внутри не добавлять
                    copy_set.remove(o_line)
                    count_del += 1
            other_lines = copy_set.copy()
        print(f"Удалили {count_del} строк из  {len(lines)}")
        if merge:
            all_lines = lines + list(other_lines)
        else:
            all_lines = list(other_lines)
        # NWS filter
        t = []
        for l in all_lines:
            e = np.fromstring(l, dtype=float, sep=" ")[1:5]
            t.append(e)
        len_before = len(all_lines)
        # t = np.vstack([np.fromstring(l[1:5], dtype=float, sep=" ") for l in all_lines])
        boxes = np.array(t)
        indexs = non_max_suppression(
            boxes, [1.0] * len(boxes), min_score=0.1, iou_score=0.95
        )
        all_lines = [all_lines[i] for i in indexs.astype(int)]
        print(f"len before {len_before}, box {len(boxes)} after {len(all_lines)}")
        with open(path2save / path.name, "w") as f_out:
            for line in all_lines:
                f_out.writelines(line)


def merge_yolo_anno(
    path2label,
    path2image,
    path2other,
    path2save,
    path2netmodel="/storage/reshetnikov/runs/yolov8/yolov8x_fold_0/weights/best.pt",
    iou_tresh=0.4,
):
    path2other = Path(path2other)
    path2save = Path(path2save)
    model = YOLO(path2netmodel)
    f_images = {x.stem: x for x in Path(path2image).glob("*")}
    for path in Path(path2label).glob("*.txt"):
        with open(path, "r") as f1:
            lines = f1.readlines()
        with open(path2other / path.name, "r") as f2:
            other_lines = set(f2.readlines())

        for main_line in lines:
            box_1 = yolo2xyxy(
                *np.fromstring(main_line, dtype=float, sep=" ")[1:].tolist()
            )
            copy_set = other_lines.copy()
            for o_line in other_lines:
                box_2 = yolo2xyxy(
                    *np.fromstring(o_line, dtype=float, sep=" ")[1:].tolist()
                )

                if (
                    iou_value(box_1, box_2) > iou_tresh
                ):  # удаляем box у которого пересечение с основным box  если внутри не добавлять
                    copy_set.remove(o_line)
            other_lines = copy_set.copy()
        print(other_lines)
        other_lines = list(other_lines)
        # Neural net filtration
        a = [np.fromstring(line, dtype=float, sep=" ")[1:] for line in other_lines]

        bbox_coords = xywh2xyxy(np.array(a))
        f_image = str(f_images[path.stem])
        image_with_bbox = ImageWithBoxs(f_image, bbox_coords)
        indexes = neyral_filter(image_with_bbox, model, valid_score=0.6)
        filter_lines = [other_lines[k] for k in indexes]
        print(len(bbox_coords), len(filter_lines))
        lines = lines + filter_lines
        with open(path2save / path.name, "w") as f_out:
            for line in lines:
                f_out.writelines(line)


def filter_bboxs_by_network(
    path2label,
    path2image,
    path2save,
    path2netmodel="/storage/reshetnikov/runs/yolov8/yolov8x_fold_0/weights/best.pt",
    iou_tresh=0.4,
    score=0.5,
):
    path2save = Path(path2save)
    model = YOLO(path2netmodel)
    f_images = {x.stem: x for x in Path(path2image).glob("*")}
    for path in Path(path2label).glob("*.txt"):
        with open(path, "r") as f1:
            lines = f1.readlines()
        # Neural net filtration
        a = [np.fromstring(line, dtype=float, sep=" ")[1:] for line in lines]

        bbox_coords = xywh2xyxy(np.array(a))
        f_image = str(f_images[path.stem])
        image_with_bbox = ImageWithBoxs(f_image, bbox_coords)
        indexes = neyral_filter(image_with_bbox, model, valid_score=score)
        filter_lines = [lines[k] for k in indexes]
        print(len(bbox_coords), len(filter_lines))

        with open(path2save / path.name, "w") as f_out:
            for line in filter_lines:
                f_out.writelines(line)


if __name__ == "__main__":
    split_with_file("/storage/reshetnikov/open_pits_merge/merge_fraction/split/images",
                    "/storage/reshetnikov/open_pits_merge/merge_fraction/split/keypoint/",
                    "/storage/reshetnikov/open_pits_merge/merge_fraction/split/files_split.csv",
                    "/storage/reshetnikov/open_pits_merge/merge_fraction/split/train_split_kpnt/")
    
    # k_fold_split_yolo(
    #     "/storage/reshetnikov/open_pits_merge/obb_yolo8/anno/",
    #     "/storage/reshetnikov/open_pits_merge/image_jpg/",
    #     "/storage/reshetnikov/open_pits_merge/obb_yolo8/Fold/",
    #     4,
    # )
