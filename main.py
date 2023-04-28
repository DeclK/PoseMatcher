# Inference 2 videos and use dtw to match the pose keypoints.
from tools.inferencer import PoseInferencerV2
from tools.dtw import DTWForKeypoints
from argparse import ArgumentParser
from tqdm import tqdm
import mmengine
import numpy as np
import mmcv
import cv2

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/mark2.py')
    parser.add_argument('--video1', type=str, default='assets/tennis5.mp4')
    parser.add_argument('--video2', type=str, default='assets/tennis1.mp4')
    return parser.parse_args()

def concat(img1, img2, height=1080):
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape

    # Calculate the scaling factor for each image
    scale1 = height / img1.shape[0]
    scale2 = height / img2.shape[0]

    # Resize the images
    img1 = cv2.resize(img1, (int(w1*scale1), int(h1*scale1)))
    img2 = cv2.resize(img2, (int(w2*scale2), int(h2*scale2)))

    # Concatenate the images horizontally
    image = cv2.hconcat([img1, img2])
    return image

def main(cfg):
    # build PoseInferencerV2
    pose_inferencer = PoseInferencerV2(
                        cfg.det_cfg,
                        cfg.pose_cfg,
                        device='cpu')
    
    v1 = mmcv.VideoReader(cfg.video1)
    v2 = mmcv.VideoReader(cfg.video2)
    video_writer = None

    all_det1, all_pose1 = pose_inferencer.inference_video(cfg.video1, draw_picture=True)
    all_det2, all_pose2 = pose_inferencer.inference_video(cfg.video2)

    keypoints1 = np.stack([p.keypoints[0] for p in all_pose1])
    keypoints2 = np.stack([p.keypoints[0] for p in all_pose2])

    dtw_path, oks_unnorm = DTWForKeypoints(keypoints1, keypoints2).get_dtw_path()

    for i, j in tqdm(dtw_path): 
        frame1 = v1[i]
        frame2 = v2[j]
        # concate two frames
        frame = concat(frame1, frame2)
        # width and height of frame
        w, h = frame.shape[1], frame.shape[0]
        if video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter('dtw_compare.mp4', 
                                            fourcc, v1.fps, (w, h))
        video_writer.write(frame)

# all_pose = []
if __name__ == '__main__':
    # add tools to PYTHONPATH
    args = parse_args()
    cfg = mmengine.Config.fromfile(args.config)
    cfg.video1 = args.video1
    cfg.video2 = args.video2

    main(cfg)