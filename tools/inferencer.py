import numpy as np
from .video_reader import VideoReader
from tqdm import tqdm
from mmdeploy_runtime import Detector, PoseDetector

class PoseInferencerV3:
    """ V3 Uses mmdeploy python sdk to inference onnx model. 
    """
    def __init__(self,
                 det_cfg, 
                 pose_cfg,
                 device='cpu') -> None:
        # init
        self.det_cfg = det_cfg
        self.pose_cfg = pose_cfg
        self.device = device

        # build detector
        self.detector = Detector(model_path=det_cfg.model_path,
                                 device_name=device)
        self.pose_detector = PoseDetector(model_path=pose_cfg.model_path,
                                          device_name=device)
        # video count
        self.video_count = 0

    def process_one_image(self, img):
        # next 4 lines are copied from mmdeploy-runtime det_pose.py
        bboxes, labels, _ = self.detector(img)
        keep = np.logical_and(labels == 0, bboxes[..., 4] > 0.6)
        bboxes = bboxes[keep, :4]   # only consider person, label 0
        labels = labels[keep]
        poses = self.pose_detector(img, bboxes)
        
        # empty case
        if len(poses) == 0:
            bboxes = np.zeros((1, 4))
            labels = np.zeros(1)
            poses = np.zeros((1, 17, 3))
            
        keypoints = poses[:, :, :2]
        pts_scores = poses[:, :, 2]

        return bboxes, keypoints

    def inference_video(self, video_path):
        """ Inference a video with detector and pose model
        Return:
            all_pose: a list of keypoints
            all_det: a list of bboxes
        """
        video_reader = VideoReader(video_path)
        all_pose, all_det = [], []

        count = self.video_count + 1
        for frame in tqdm(video_reader, desc=f'Inference video {count}'):
            # inference with detector
            det, pose = self.process_one_image(frame)
            all_det.append(det)
            all_pose.append(pose)
        self.video_count += 1

        return all_det, all_pose


        

        
