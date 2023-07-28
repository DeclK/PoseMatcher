# Inference 2 videos and use dtw to match the pose keypoints.
from tools.inferencer import PoseInferencerV3
from tools.dtw import DTWForKeypoints
from tools.visualizer import FastVisualizer
from tools.utils import convert_video_to_playable_mp4, add_logo_to_video
from tools.video_reader import VideoReader
from tqdm import tqdm
import time
from omegaconf import OmegaConf
import numpy as np
import cv2

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
    pose_inferencer = PoseInferencerV3(
                        cfg.det_cfg,
                        cfg.pose_cfg,
                        device='cpu')
    
    v1 = VideoReader(cfg.video1)
    v2 = VideoReader(cfg.video2)
    video_writer = None

    all_bboxes1, all_keyopints1 = pose_inferencer.inference_video(cfg.video1)
    all_bboxes2, all_keyopints2 = pose_inferencer.inference_video(cfg.video2)

    # There might be multi people in the fig, we force to use the first pred
    keypoints1 = np.stack([pts[0] for pts in all_keyopints1])  
    keypoints2 = np.stack([pts[0] for pts in all_keyopints2])
    boxes1 = np.stack([bboxes[0] for bboxes in all_bboxes1])
    boxes2 = np.stack([bboxes[0] for bboxes in all_bboxes2])

    dtw_path, oks, oks_unnorm = DTWForKeypoints(keypoints1, keypoints2).get_dtw_path()

    # output_name with timestamp
    stamp = time.strftime("%y%m%H%M%S", time.localtime()) 
    output_name = 'tennis_' + stamp + '.mp4'

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
        # write video
        w, h = frame.shape[1], frame.shape[0]
        if video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_name, 
                                            fourcc, v1.fps, (w, h))
        video_writer.write(frame)
    video_writer.release()
    convert_video_to_playable_mp4(output_name)
    add_logo_to_video(output_name, 'assets/logo.png', (w, h))

if __name__ == '__main__':

    cfg = OmegaConf.load('configs/mark3.yaml')
    main(cfg)