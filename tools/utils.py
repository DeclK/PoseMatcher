from mmdet.datasets import CocoDataset
import time
from easydict import EasyDict
import numpy as np

def coco_keypoint_id_table(reverse=False):
    id2name = { 0: 'nose',
                1: 'left_eye',
                2: 'right_eye',
                3: 'left_ear',
                4: 'right_ear',
                5: 'left_shoulder',
                6: 'right_shoulder',
                7: 'left_elbow',
                8: 'right_elbow',
                9: 'left_wrist',
                10: 'right_wrist',
                11: 'left_hip',
                12: 'right_hip',
                13: 'left_knee',
                14: 'right_knee',
                15: 'left_ankle',
                16: 'right_ankle'}
    if reverse:
        return {v: k for k, v in id2name.items()}
    return id2name

def get_skeleton():
    """ My skeleton links, I deleted some links from default coco style.
    """
    SKELETON = EasyDict()
    SKELETON.head = [[0,1], [0,2], [1,3], [2,4]]
    SKELETON.left_arm = [[5, 7], [7, 9]]
    SKELETON.right_arm = [[6, 8], [8, 10]]
    SKELETON.left_leg = [[11, 13], [13, 15]]
    SKELETON.right_leg = [[12, 14], [14, 16]]
    SKELETON.body = [[5, 6], [5, 11], [6, 12], [11, 12]]
    return SKELETON

def get_keypoint_weight(low_weight_ratio=0.1, mid_weight_ratio=0.5):
    """ Get keypoint weight, used in object keypoint similarity,
    `low_weight_names` are points I want to pay less attention.
    """
    low_weight_names = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear']
    mid_weight_names = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']

    logtis = np.ones(17)
    name2id = coco_keypoint_id_table(reverse=True)

    low_weight_id = [name2id[n] for n in low_weight_names]
    mid_weight_id = [name2id[n] for n in mid_weight_names]
    logtis[low_weight_id] = low_weight_ratio
    logtis[mid_weight_id] = mid_weight_ratio

    weights = logtis / np.sum(logtis)
    return weights

def coco_cat_id_table():
    classes = CocoDataset.METAINFO['classes']
    id2name = {i: name for i, name in enumerate(classes)}

    return id2name

def filter_by_catgory(bboxes, scores, labels, names):
    """ Filter labels by classes
    Args:
        - labels: list of labels, each label is a dict
        - classes: list of class names
    """
    id2name = coco_cat_id_table()
    # names of labels
    label_names = [id2name[id] for id in labels]
    # filter by class names
    mask = np.isin(label_names, names)
    return bboxes[mask], scores[mask], labels[mask]

def filter_by_score(bboxes, scores, labels, score_thr):
    """ Filter bboxes by score threshold
    Args:
        - bboxes: list of bboxes, each bbox is a dict
        - score_thr: score threshold
    """
    mask = scores > score_thr
    return bboxes[mask], scores[mask], labels[mask]


class Timer:
    def __init__(self):
        self.start_time = time.time()

    def click(self):
        used_time = time.time() - self.start_time
        self.start_time = time.time()
        return used_time
        
    def start(self):
        self.start_time = time.time()