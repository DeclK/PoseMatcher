# Inference 2 videos and use dtw to match the pose keypoints.
from tools.inferencer import PoseInferencerV3
from tools.dtw import DTWForKeypoints
from tools.visualizer import FastVisualizer
from tools.utils import convert_video_to_playable_mp4, add_logo_to_video
from tools.video_reader import VideoReader
from tqdm import tqdm
from omegaconf import OmegaConf
from pathlib import Path
import pickle
import time
import sys
import gradio as gr
import numpy as np
import os
import cv2

BASE_DIR =  Path(sys.argv[0]).parent

def add_ffmpeg_to_path():
    os.environ['PATH'] += os.pathsep + str(BASE_DIR / 'ffmpeg')

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

def draw(vis: FastVisualizer, img, keypoint, box, oks, oks_unnorm, 
         draw_non_transparent_area=True,
         draw_human_keypoints=True,
         draw_score_bar=False):
    vis.set_image(img)
    if draw_non_transparent_area:
        vis.draw_non_transparent_area(box)
    if draw_human_keypoints:
        vis.draw_human_keypoints(keypoint, oks_unnorm)
    if draw_score_bar:
        vis.draw_score_bar(oks)
    return vis.get_image()

def main(video1,
         video2, 
         cache,
         vis_choices,
         progress=gr.Progress(track_tqdm=True)
         ):
    # build PoseInferencerV3
    cfg = OmegaConf.load('configs/mark3.yaml')
    pose_inferencer = PoseInferencerV3(
                        cfg.det_cfg,
                        cfg.pose_cfg,
                        device='cpu')
    
    v1 = VideoReader(video1)
    v2 = VideoReader(video2)
    video_writer = None

    
    # cache
    cache_file = Path('results.cache')
        
    if not cache or not cache_file.exists():
        all_bboxes1, all_keyopints1 = pose_inferencer.inference_video(video1)
        all_bboxes2, all_keyopints2 = pose_inferencer.inference_video(video2)

        # There might be multi people in the fig, we force to use the first pred
        keypoints1 = np.stack([pts[0] for pts in all_keyopints1])  
        keypoints2 = np.stack([pts[0] for pts in all_keyopints2])
        boxes1 = np.stack([bboxes[0] for bboxes in all_bboxes1])
        boxes2 = np.stack([bboxes[0] for bboxes in all_bboxes2])

        dtw_path, oks, oks_unnorm = DTWForKeypoints(keypoints1, keypoints2).get_dtw_path()
        cache_file.touch(exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump([keypoints1, keypoints2, boxes1, boxes2, dtw_path, oks, oks_unnorm], f)
    else:
        assert cache_file.exists()
        with open(cache_file, 'rb') as f:
            keypoints1, keypoints2, boxes1, boxes2, dtw_path, oks, oks_unnorm = pickle.load(f)

    # output_name with timestamp
    stamp = time.strftime("%y%m%H%M%S", time.localtime()) 
    output_name = BASE_DIR.parent / ('tennis_' + stamp + '.mp4')
    output_name = str(output_name)

    vis = FastVisualizer()

    # vis choices
    draw_non_transparent_area = True if "检测框蒙版" in vis_choices else False 
    draw_human_keypoints = True if "人体关键点" in vis_choices else False
    draw_score_bar = True if "匹配得分" in vis_choices else False
    
    for i, j in tqdm(dtw_path, desc='Visualizing'): 
        frame1 = v1[i]
        frame2 = v2[j]

        frame1_ = draw(vis,
                       frame1.copy(),
                       keypoints1[i],
                       boxes1[i],
                       oks[i, j],
                       oks_unnorm[i, j], 
                       draw_non_transparent_area,
                       draw_human_keypoints,
                       draw_score_bar)

        frame2_ = draw(vis,
                       frame2.copy(),
                       keypoints2[j],
                       boxes2[j],
                       oks[i, j],
                       oks_unnorm[i, j],
                       draw_non_transparent_area,
                       draw_human_keypoints)

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

    return output_name   # make it a generator so it can show stop button

if __name__ == '__main__':

    add_ffmpeg_to_path()

    inputs = [
        gr.Video(label="Input video 1", height=180),
        gr.Video(label="Input video 2", height=180),
        gr.Checkbox(value=True, label="使用缓存", 
                    info='result.cache 是在推理中生成的中间结果，使用该选项将会直接使用该缓存结果，避免重复推理。\
                        但如果你提交了与上一次推理不同的视频，请先取消该选项，生成新的缓存后再启用'),
        gr.CheckboxGroup(["检测框蒙版", "人体关键点", "匹配得分"], label="可视化选项",
                          value=["人体关键点", "匹配得分"]),
    ]

    output = gr.Video(label="Output video", height=360)

    demo = gr.Interface(fn=main,
                        inputs=inputs,
                        outputs=output,
                        # might show wrong example preview, just clean the cookie & cache of browser
                        examples=[["assets/tennis1.mp4", "assets/tennis2.mp4", None]],
                        allow_flagging='never',
                        title='PoseMatcher',
                        thumbnail='logo.png',
                        article='### Github @[PoseMatcher](https://github.com/DeclK/PoseMatcher), please contact @[DeclK](https://github.com/DeclK) if you are interested in this project.'
                        ).queue()
    demo.launch(inbrowser=True)
