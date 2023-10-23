import numpy as np
from .video_reader import VideoReader
from .utils import coco_keypoint_id_table , mp_keypoint_id_table ,  mp_cocoid , getcocokeypoints
from tqdm import tqdm
from mmdeploy_runtime import Detector
import mediapipe as mp
import cv2

class PoseInferencerV3:
    """ V3 Uses mmdeploy python sdk to inference onnx model. 
    """
    def __init__(self,
                 det_cfg, 
                 #pose_cfg,
                 device='cpu') -> None:
        # init
        self.det_cfg = det_cfg
        #self.pose_cfg = pose_cfg
        self.device = device
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5)
        self.mp2coco = mp_cocoid(coco_keypoint_id_table() ,  mp_keypoint_id_table() )
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        # build detector
        self.detector = Detector(model_path=det_cfg.model_path,
                                 device_name=device)
        # self.pose_detector = PoseDetector(model_path=pose_cfg.model_path,
        #                                   device_name=device)
        # video count
        self.video_count = 0

    def process_one_image(self, img):
        # next 4 lines are copied from mmdeploy-runtime det_pose.py
        bboxes, labels, _ = self.detector(img)
        #print("box shape is {} and labels shape is {}" .format(bboxes.shape, labels.shape))
        keep = np.logical_and(labels == 0, bboxes[..., 4] > 0.6)
        bboxes = bboxes[keep, :4]   # only consider person, label 0
        labels = labels[keep]
        results = self.pose.process(cv2.cvtColor(img , cv2.COLOR_BGR2RGB))  #hope to convert to (17,2)
        img_height , img_width , _ = img.shape
        if not results.pose_landmarks:
            joint_2d = np.zeros((17, 2))
        else:
            joint_2d_origin =  np.stack([ np.array([results.pose_landmarks.landmark[i].x*img_width , results.pose_landmarks.landmark[i].y*img_height ] )      for i in range(33)  ])
            joint_2d = getcocokeypoints(joint_2d_origin , self.mp2coco)
        
        if not results.pose_world_landmarks:
            joint_3d = np.zeros((17, 3))
        else:
            joint_3d_origin = np.stack([ np.array([results.pose_world_landmarks.landmark[i].x , results.pose_world_landmarks.landmark[i].y,   results.pose_world_landmarks.landmark[i].z] )      for i in range(33)  ])
            joint_3d = getcocokeypoints(joint_3d_origin , self.mp2coco)
        
        # empty case
        if len(bboxes) == 0:
            bboxes = np.zeros((1, 4))
        #     labels = np.zeros(1)
        #     poses = np.zeros((1, 17, 3))
            
        # keypoints = poses[:, :, :2]
        # pts_scores = poses[:, :, 2]
        #assert(joint_3d.shape == (17,3) and joint_2d.shape == (17,2) and bboxes.shape[1] == 4 ) 
        return bboxes, joint_2d, joint_3d

    def inference_video(self, video_path):
        """ Inference a video with detector and pose model
        Return:
            all_pose: a list of keypoints
            all_det: a list of bboxes
        """
        video_reader = VideoReader(video_path)
        all_pose3d , all_pose2d, all_det = [], [] , []

        count = self.video_count + 1
        for frame in tqdm(video_reader, desc=f'Inference video {count}'):
            # inference with detector
            det, pose2d , pose3d = self.process_one_image(frame)
            #print("det shape is {} pose shape is {}" .format(det.shape , pose.shape)   , flush=True)
            all_det.append(det)
            all_pose2d.append(pose2d)
            all_pose3d.append(pose3d)
        self.video_count += 1

        return all_det, all_pose2d , all_pose3d


        

        
