# Inference 2 videos and use dtw to match the pose keypoints.
from tools.inferencer import PoseInferencerV2
from tools.dtw import DTWForKeypoints
from tools.visualizer import FastVisualizer
from argparse import ArgumentParser
from tools.utils import convert_video_to_playable_mp4
from tqdm import tqdm
import mmengine
import numpy as np
import mmcv
import cv2

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/mark2.py')
    parser.add_argument('--video1', type=str, default='assets/tennis1.mp4')
    parser.add_argument('--video2', type=str, default='assets/tennis2.mp4')
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

def draw(vis: FastVisualizer, img, keypoint, box, oks, oks_unnorm, draw_score_bar=True):
    vis.set_image(img)
    vis.draw_non_transparent_area(box)
    if draw_score_bar:
        vis.draw_score_bar(oks)
    vis.draw_human_keypoints(keypoint, oks_unnorm)
    return vis.get_image()

def main(cfg):
    # build PoseInferencerV2
    pose_inferencer = PoseInferencerV2(
                        cfg.det_cfg,
                        cfg.pose_cfg,
                        device='cpu')
    
    v1 = mmcv.VideoReader(cfg.video1)
    v2 = mmcv.VideoReader(cfg.video2)
    video_writer = None

    all_det1, all_pose1 = pose_inferencer.inference_video(cfg.video1)
    all_det2, all_pose2 = pose_inferencer.inference_video(cfg.video2)

    keypoints1 = np.stack([p.keypoints[0] for p in all_pose1])  # forced the first pred
    keypoints2 = np.stack([p.keypoints[0] for p in all_pose2])
    boxes1 = np.stack([d.bboxes[0] for d in all_det1])
    boxes2 = np.stack([d.bboxes[0] for d in all_det2])

    dtw_path, oks, oks_unnorm = DTWForKeypoints(keypoints1, keypoints2).get_dtw_path()

    vis = FastVisualizer()
    
    for i, j in tqdm(dtw_path): 
        frame1 = v1[i]
        frame2 = v2[j]

        frame1_ = draw(vis, frame1.copy(), keypoints1[i], boxes1[i],
                       oks[i, j], oks_unnorm[i, j])
        frame2_ = draw(vis, frame2.copy(), keypoints2[j], boxes2[j],
                       oks[i, j], oks_unnorm[i, j], draw_score_bar=False)
        # concate two frames
        frame = concat(frame1_, frame2_)
        # draw logo
        vis.set_image(frame)
        frame = vis.draw_logo().get_image()
        # write video
        w, h = frame.shape[1], frame.shape[0]
        if video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter('dtw_compare.mp4', 
                                            fourcc, v1.fps, (w, h))
        video_writer.write(frame)
    video_writer.release()
    convert_video_to_playable_mp4('dtw_compare.mp4')

if __name__ == '__main__':
    args = parse_args()
    cfg = mmengine.Config.fromfile(args.config)
    cfg.video1 = args.video1
    cfg.video2 = args.video2

    main(cfg)