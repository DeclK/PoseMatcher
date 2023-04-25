import numpy as np
import mmcv
from mmengine.visualization import Visualizer
import cv2 as cv
from tqdm import tqdm
from mmengine.registry import init_default_scope
from mmpose.apis import inference_topdown, init_model
from mmdet.apis import inference_detector, init_detector
from utils import filter_by_catgory, filter_by_score


det = '/github/Tennis.ai/model_zoo/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco.py'
det_ckpt = '/github/Tennis.ai/model_zoo/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'

pose = '/github/Tennis.ai/model_zoo/rtmpose/rtmpose-t_8xb256-420e_aic-coco-256x192/rtmpose-t_8xb256-420e_aic-coco-256x192.py'
pose_ckpt = '/github/Tennis.ai/model_zoo/rtmpose/rtmpose-t_8xb256-420e_aic-coco-256x192/rtmpose-tiny_simcc-aic-coco_pt-aic-coco_420e-256x192-cfc8f33d_20230126.pth'
img = '/github/Tennis.ai/assets/000000197388.jpg'

# # init detector
device = 'cpu'
det_model = init_detector(det, det_ckpt, device=device)
pose_model = init_model(pose, pose_ckpt, device=device)


def process_one_image(img, det_model, pose_model):
    init_default_scope('mmdet')
    det_result = inference_detector(det_model, img)
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
    # TODO: we can only process one person, if 2 persons are detected
    # this will cause error
    pose_result = inference_topdown(pose_model, img, person_bboxes)
    keypoints = np.concatenate([r.pred_instances.keypoints 
                                     for r in pose_result])
    pts_scores = np.concatenate([r.pred_instances.keypoint_scores 
                                     for r in pose_result])

    return (bboxes, scores, labels), (keypoints, pts_scores)

vis = Visualizer()
# use mmcv to deal with video
from pathlib import Path
video1 = Path('/github/Tennis.ai/assets/tennis4.mp4')
video3 = Path('/github/Tennis.ai/assets/tennis3.mp4')
v1 = mmcv.VideoReader(str(video1))
v3 = mmcv.VideoReader(str(video3))
video_writer = None

# draw_picture = False

# # process video1
# # we need to get the keypoints of the both videos
# all_pose = []
# for frame in tqdm(v1):
#     # inference with detector
#     det, pose = process_one_image(frame,
#                                   det_model,
#                                   pose_model)
#     # collect keypoints & pts scores
#     all_pose.append(pose)
#     # draw image
#     if draw_picture:
#         vis.set_image(frame)
#         vis.draw_bboxes(det[0])
#         vis.draw_points(pose[0].reshape(-1, 2))
#         image = vis.get_image()
#         # use opencv to write the image to video
#         if video_writer is None:
#             fourcc = cv.VideoWriter_fourcc(*'XVID')
#             video_writer = cv.VideoWriter('output1.mp4',
#                 fourcc, v1.fps, (v1.width, v1.height))
#         video_writer.write(image)

# keypoints1 = np.stack([p[0][0] for p in all_pose])

# # process video3
# all_pose = []
# for frame in tqdm(v3):
#     # inference with detector
#     det, pose = process_one_image(frame,
#                                   det_model,
#                                   pose_model)
#     # collect keypoints & pts scores
#     all_pose.append(pose)
#     # draw image
#     if draw_picture:
#         vis.set_image(frame)
#         vis.draw_bboxes(det[0])
#         vis.draw_points(pose[0].reshape(-1, 2))
#         image = vis.get_image()
#         # use opencv to write the image to video
#         if video_writer is None:
#             fourcc = cv.VideoWriter_fourcc(*'XVID')
#             video_writer = cv.VideoWriter('output3.mp4',
#                 fourcc, v3.fps, (v3.width, v3.height))
#         video_writer.write(image)
        
# keypoints3 = np.stack([p[0][0] for p in all_pose])
from mmengine.fileio import load

keypoints1, kp1_scores = load('tennis4.pkl')
keypoints3, kp2_scores = load('tennis3.pkl')

from dtw import DTWForKeypoints
dtw_path = DTWForKeypoints(keypoints1, keypoints3).get_dtw_path()

def concat(img1, img2, height=1080):
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape

    # Calculate the scaling factor for each image
    scale1 = height / img1.shape[0]
    scale2 = height / img2.shape[0]

    # Resize the images
    img1 = cv.resize(img1, (int(w1*scale1), int(h1*scale1)))
    img2 = cv.resize(img2, (int(w2*scale2), int(h2*scale2)))

    # Concatenate the images horizontally
    image = cv.hconcat([img1, img2])
    return image

# draw_picture = True
for i, j in tqdm(dtw_path):
    frame1 = v1[i]
    frame3 = v3[j]
    # concate two frames
    frame = concat(frame1, frame3)
    # width and height of frame
    w, h = frame.shape[1], frame.shape[0]
    if video_writer is None:
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        video_writer = cv.VideoWriter('dtw_compare.mp4',
            fourcc, v1.fps, (w, h))
    video_writer.write(frame)

# all_pose = []

# for frame in tqdm(v):
#     # inference with detector
#     det, pose = process_one_image(frame,
#                                   det_model,
#                                   pose_model)
#     # collect keypoints & pts scores
#     all_pose.append(pose)
#     # draw image
#     if draw_picture:
#         vis.set_image(frame)
#         vis.draw_bboxes(det[0])
#         vis.draw_points(pose[0].reshape(-1, 2))
#         image = vis.get_image()
#         # use opencv to write the image to video
#         if video_writer is None:
#             fourcc = cv.VideoWriter_fourcc(*'XVID')
#             video_writer = cv.VideoWriter('output1.mp4',
#                 fourcc, v.fps, (v.width, v.height))
#         video_writer.write(image)

# # process all pose, pack all frames results
# # p[0][0] is an workaround to force only 1 person
# all_keypoints = np.stack([p[0][0] for p in all_pose])
# all_keyscore = np.stack([p[1][0] for p in all_pose])
# from mmengine.fileio import dump
# dump([all_keypoints, all_keyscore], kp_result_file)

# we need few things to do this

# 1. a inferencer
# 2. a visualizer
# 3. a dtw matcher which is already done
# 4. use inferencer to get keypoints, we will need this for 2 series
# 5. use dtw to do 2 series matching
# 6. convert the matching result to a video
# call this is a DualPoseInferencer