import time
from pathlib import Path
from PIL import Image
import ffmpeg
import shutil
import os
import tempfile
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
    classes = ( 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
                'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
                'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
                'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
                'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
                'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush')
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

def convert_video_to_playable_mp4(video_path: str) -> str:
    """ Copied from gradio
    Convert the video to mp4. If something goes wrong return the original video.
    """
    try:
        output_path = Path(video_path).with_suffix(".mp4")
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            shutil.copy2(video_path, tmp_file.name)
            command = (ffmpeg
                            .input(tmp_file.name)
                            .output(output_path.name)
                            .overwrite_output()
                            .run())
        # had to delete it by hand on windows
        # link https://stackoverflow.com/questions/15588314/cant-access-temporary-files-created-with-tempfile
        os.remove(tmp_file.name)
    except:
        print(f"Error when converting video to browser-playable format.")

def add_logo_to_video(video_path: str, logo_path: str, video_size,
                      logo_scale=300, shift=(0, 0)):
    """ Add logo to video.
    Args:
        - `video_size` is the size of video, (w, h)
        - `logo_scale` is the width of logo, height is adaptive
        - `shift` is the shift of logo position from the down-right corner
    """
    # resize the logo to a fixed width
    logo = Image.open(logo_path)
    logo_w, logo_h = logo.size
    logo_h = int(logo_h / logo_w * logo_scale)
    logo_w = logo_scale
    logo = logo.resize((logo_w, logo_h))
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            shutil.copy2(video_path, tmp_file.name)
            input_video = ffmpeg.input(tmp_file.name)
            logo = ffmpeg.input(logo_path).filter("scale", logo_scale, -1)
            x = video_size[0] - logo_w - shift[0]
            y = video_size[1] - logo_h - shift[1]
            overlay_video = ffmpeg.overlay(input_video, logo, x=x, y=y)

            command =(ffmpeg.output(overlay_video, video_path)
                            .overwrite_output()
                            .run())
        os.remove(tmp_file.name)
    except:
        print(f"Error when adding logo to video.")

class Timer:
    def __init__(self):
        self.start_time = time.time()

    def click(self):
        used_time = time.time() - self.start_time
        self.start_time = time.time()
        return used_time
        
    def start(self):
        self.start_time = time.time()