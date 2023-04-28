import numpy as np
import mmcv
from pathlib import Path
from collections import namedtuple
import cv2 as cv
from tqdm import tqdm
from mmengine.registry import init_default_scope
from mmengine.visualization import Visualizer
from mmpose.apis import inference_topdown, init_model
from mmdet.apis import inference_detector, init_detector
from .utils import filter_by_catgory, filter_by_score, Timer
from .apis import build_onnx_model_and_task_processor, inference_onnx_model


class PoseInferencer:
    def __init__(self,
                 det_cfg, 
                 pose_cfg,
                 device='cpu') -> None:
        # init
        self.det_model_cfg = det_cfg.model_cfg
        self.det_model_ckpt = det_cfg.model_ckpt
        self.pose_model_cfg = pose_cfg.model_cfg
        self.pose_model_ckpt = pose_cfg.model_ckpt
        
        self.detector = init_detector(self.det_model_cfg, 
                                      self.det_model_ckpt,
                                      device=device)
        self.pose_model = init_model(self.pose_model_cfg,
                                     self.pose_model_ckpt,
                                     device=device)

    def process_one_image(self, img):
        init_default_scope('mmdet')
        det_result = inference_detector(self.detector, img)
        det_inst = det_result.pred_instances.cpu().numpy()
        bboxes, scores, labels = (det_inst.bboxes,
                                  det_inst.scores,
                                  det_inst.labels)
        bboxes, scores, labels = filter_by_score(bboxes, scores,
                                                 labels, 0.5)
        bboxes, scores, labels = filter_by_catgory(bboxes, scores, labels, 
                                ['person', 'tennis racket', 'sports ball'])
        person_bboxes, _, _= filter_by_catgory(bboxes, scores,
                                               labels, ['person'])
        # inference with pose model
        init_default_scope('mmpose')
        # TODO: need to handle the non-person case
        pose_result = inference_topdown(self.pose_model, img, person_bboxes)
        keypoints = np.concatenate([r.pred_instances.keypoints 
                                        for r in pose_result])
        pts_scores = np.concatenate([r.pred_instances.keypoint_scores 
                                        for r in pose_result])

        DetInst = namedtuple('DetInst', ['bboxes', 'scores', 'labels'])
        PoseInst = namedtuple('PoseInst', ['keypoints', 'pts_scores'])
        return DetInst(bboxes, scores, labels), PoseInst(keypoints, pts_scores)

    def inference_video(self, video_path, draw_picture=False):
        """ Inference a video with detector and pose model
        Return:
            all_pose: a list of PoseInst, check the namedtuple definition
            all_det: a list of DetInst
        """
        video_reader = mmcv.VideoReader(video_path)
        video_writer = None

        draw_picture = False

        all_pose = []
        all_det = []

        for frame in tqdm(video_reader):
            # inference with detector
            det, pose = self.process_one_image(frame)
            all_pose.append(pose)
            all_det.append(det)
            # draw image
            if draw_picture:
                pass
                # TODO: use opencv to vis, which is much faster

        return all_det, all_pose

class PoseInferencerV2:
    """ V2 Use onnx for detection model, still use pytorch for pose model.
    """
    def __init__(self,
                 det_cfg, 
                 pose_cfg,
                 device='cpu') -> None:
        # init
        self.det_deploy_cfg = det_cfg.deploy_cfg
        self.det_model_cfg = det_cfg.model_cfg
        self.det_backend_files = det_cfg.backend_files

        self.pose_model_cfg = pose_cfg.model_cfg
        self.pose_model_ckpt = pose_cfg.model_ckpt
        
        self.detector, self.task_processor = \
            build_onnx_model_and_task_processor(self.det_model_cfg,
                                                self.det_deploy_cfg,
                                                self.det_backend_files,
                                                device)
        self.pose_model = init_model(self.pose_model_cfg,
                                     self.pose_model_ckpt,
                                     device)

    def process_one_image(self, img):
        init_default_scope('mmdet')
        det_result = inference_onnx_model(self.detector,
                                          self.task_processor,
                                          self.det_deploy_cfg,
                                          img)
        det_inst = det_result[0].pred_instances.cpu().numpy()
        bboxes, scores, labels = (det_inst.bboxes,
                                  det_inst.scores,
                                  det_inst.labels)
        bboxes, scores, labels = filter_by_score(bboxes, scores,
                                                 labels, 0.5)
        bboxes, scores, labels = filter_by_catgory(bboxes, scores, labels, 
                                ['person', 'tennis racket', 'sports ball'])
        person_bboxes, _, _= filter_by_catgory(bboxes, scores,
                                               labels, ['person'])
        # inference with pose model
        init_default_scope('mmpose')
        pose_result = inference_topdown(self.pose_model, img, person_bboxes)
        if len(pose_result) == 0:
            # no person place holder, and bboxes len is 0
            keypoints = np.zeros((1, 17, 2))
            pts_scores = np.zeros((1, 17))
        else:
            keypoints = np.concatenate([r.pred_instances.keypoints 
                                            for r in pose_result])
            pts_scores = np.concatenate([r.pred_instances.keypoint_scores 
                                            for r in pose_result])

        DetInst = namedtuple('DetInst', ['bboxes', 'scores', 'labels'])
        PoseInst = namedtuple('PoseInst', ['keypoints', 'pts_scores'])
        return DetInst(bboxes, scores, labels), PoseInst(keypoints, pts_scores)

    def inference_video(self, video_path, draw_picture=False):
        """ Inference a video with detector and pose model
        Return:
            all_pose: a list of PoseInst, check the namedtuple definition
            all_det: a list of DetInst
        """
        video_reader = mmcv.VideoReader(video_path)
        video_writer = None
        if draw_picture:
            vis = Visualizer(vis_backends=dict(type='LocalVisBackend', 
                                               save_dir='vis'))
        all_pose = []
        all_det = []

        timer = Timer()
        for frame in tqdm(video_reader):
            # inference with detector
            det, pose = self.process_one_image(frame)
            all_pose.append(pose)
            all_det.append(det)
            # TODO: draw image
            if draw_picture:
                timer.start()
                vis.set_image(frame)
                print(f'vis set image time: {timer.click()}')
                vis.draw_bboxes(det[0])
                print(f'draw bboxes: {timer.click()}')
                vis.draw_points(pose[0].reshape(-1, 2))
                print(f'draw points: {timer.click()}')
                image = vis.get_image()
                print(f'get image: {timer.click()}')
                # use opencv to write the image to video
                # import pdb; pdb.set_trace()
                if video_writer is None:
                    fourcc = cv.VideoWriter_fourcc(*'XVID')
                    video_writer = cv.VideoWriter('output1.mp4',
                                                  fourcc,
                                                  video_reader.fps,
                                                  (video_reader.width, video_reader.height))
                video_writer.write(image)
                print(f'write time: {timer.click()}')
                
        if video_writer is not None:
            video_writer.release()

        return all_det, all_pose