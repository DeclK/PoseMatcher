import numpy as np
import mmcv
from pathlib import Path
from collections import namedtuple
import cv2 as cv
from tqdm import tqdm
from mmengine.registry import init_default_scope
from mmpose.apis import inference_topdown, init_model
from mmdet.apis import inference_detector, init_detector
from utils import filter_by_catgory, filter_by_score


class PoseInferencer:
    def __init__(self,
                 det_model_config, 
                 det_model_ckpt,
                 pose_model_config,
                 pose_model_ckpt,
                 device='cpu') -> None:
        # init
        self.det_model_config = det_model_config
        self.det_model_ckpt = det_model_ckpt
        self.pose_model_config = pose_model_config
        self.pose_model_ckpt = pose_model_ckpt
        
        self.detector = init_detector(det_model_config, det_model_ckpt, device=device)
        self.pose_model = init_model(pose_model_config, pose_model_ckpt, device=device)

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
                # vis.set_image(frame)
                # vis.draw_bboxes(det[0])
                # vis.draw_points(pose[0].reshape(-1, 2))
                # image = vis.get_image()
                # # use opencv to write the image to video
                # if video_writer is None:
                #     fourcc = cv.VideoWriter_fourcc(*'XVID')
                #     video_writer = cv.VideoWriter('output1.mp4',
                #         fourcc, v.fps, (v.width, v.height))
                # video_writer.write(image)

        return all_det, all_pose

if __name__ == '__main__':
    det = '/github/Tennis.ai/model_zoo/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco.py'
    det_ckpt = '/github/Tennis.ai/model_zoo/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'

    pose = '/github/Tennis.ai/model_zoo/rtmpose/rtmpose-t_8xb256-420e_aic-coco-256x192/rtmpose-t_8xb256-420e_aic-coco-256x192.py'
    pose_ckpt = '/github/Tennis.ai/model_zoo/rtmpose/rtmpose-t_8xb256-420e_aic-coco-256x192/rtmpose-tiny_simcc-aic-coco_pt-aic-coco_420e-256x192-cfc8f33d_20230126.pth'

    # # init detector
    video_path = '/github/Tennis.ai/assets/tennis4.mp4'

    model = PoseInferencer(det, det_ckpt, pose, pose_ckpt)
    all_det, all_pose = model.inference_video(video_path)
    # some settings

    # process all pose, pack all frames results
    # p[0][0] is an workaround to force only 1 person
    all_keypoints = np.stack([p[0][0] for p in all_pose])
    all_keyscore = np.stack([p[1][0] for p in all_pose])
