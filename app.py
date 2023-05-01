# Inference 2 videos and use dtw to match the pose keypoints.
from tools.inferencer import PoseInferencerV2
from tools.dtw import DTWForKeypoints
from tools.visualizer import FastVisualizer
from tools.utils import convert_video_to_playable_mp4
from pathlib import Path
from tqdm import tqdm
import mmengine
import numpy as np
import mmcv
import cv2
import gradio as gr

def concat(img1, img2, height=1080):
    w1, h1, _ = img1.shape
    w2, h2, _ = img2.shape

    # Calculate the scaling factor for each image
    scale1 = height / img1.shape[1]
    scale2 = height / img2.shape[1]

    # Resize the images
    img1 = cv2.resize(img1, (int(h1*scale1), int(w1*scale1)))
    img2 = cv2.resize(img2, (int(h2*scale2), int(w2*scale2)))

    # Concatenate the images horizontally
    image = cv2.hconcat([img1, img2])
    return image

def draw(vis: FastVisualizer, img, keypoint, box, oks, oks_unnorm, 
         draw_human_keypoints=True,
         draw_score_bar=True):
    vis.set_image(img)
    vis.draw_non_transparent_area(box)
    if draw_score_bar:
        vis.draw_score_bar(oks)
    if draw_human_keypoints:
        vis.draw_human_keypoints(keypoint, oks_unnorm)
    return vis.get_image()

def main(video1, video2, draw_human_keypoints,
         progress=gr.Progress(track_tqdm=True)):
    # build PoseInferencerV2
    config = 'configs/mark2.py'
    cfg = mmengine.Config.fromfile(config)
    pose_inferencer = PoseInferencerV2(
                        cfg.det_cfg,
                        cfg.pose_cfg,
                        device='cpu')
    
    v1 = mmcv.VideoReader(video1)
    v2 = mmcv.VideoReader(video2)
    video_writer = None

    all_det1, all_pose1 = pose_inferencer.inference_video(video1)
    all_det2, all_pose2 = pose_inferencer.inference_video(video2)

    keypoints1 = np.stack([p.keypoints[0] for p in all_pose1])  # forced the first pred
    keypoints2 = np.stack([p.keypoints[0] for p in all_pose2])
    boxes1 = np.stack([d.bboxes[0] for d in all_det1])
    boxes2 = np.stack([d.bboxes[0] for d in all_det2])

    dtw_path, oks, oks_unnorm = DTWForKeypoints(keypoints1, keypoints2).get_dtw_path()

    vis = FastVisualizer()
    
    for i, j in tqdm(dtw_path, desc='Visualizing'): 
        frame1 = v1[i]
        frame2 = v2[j]

        frame1_ = draw(vis, frame1.copy(), keypoints1[i], boxes1[i],
                       oks[i, j], oks_unnorm[i, j], draw_human_keypoints)
        frame2_ = draw(vis, frame2.copy(), keypoints2[j], boxes2[j],
                       oks[i, j], oks_unnorm[i, j], draw_human_keypoints, draw_score_bar=False)
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
    # output video file
    convert_video_to_playable_mp4('dtw_compare.mp4')
    output = str(Path('dtw_compare.mp4').resolve())
    return output

if __name__ == '__main__':
    config = 'configs/mark2.py'
    cfg = mmengine.Config.fromfile(config)

    inputs = [
        gr.Video(label="Input video 1"),
        gr.Video(label="Input video 2"),
        "checkbox"
    ]

    output = gr.Video(label="Output video")

    demo = gr.Interface(fn=main, inputs=inputs, outputs=output,
                        allow_flagging='never').queue()
    demo.launch()
