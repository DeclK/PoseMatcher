det_cfg = dict(
    deploy_cfg='model_zoo/rtmdet/rtmdet_tiny_8xb32-300e_coco/detection_onnxruntime_static.py',
    model_cfg='model_zoo/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco.py',
    backend_files=['model_zoo/rtmdet/rtmdet_tiny_8xb32-300e_coco/end2end.onnx']
)

pose_cfg = dict(
    model_cfg='model_zoo/rtmpose/rtmpose-t_8xb256-420e_aic-coco-256x192/rtmpose-t_8xb256-420e_aic-coco-256x192.py',
    model_ckpt='model_zoo/rtmpose/rtmpose-t_8xb256-420e_aic-coco-256x192/rtmpose-tiny_simcc-aic-coco_pt-aic-coco_420e-256x192-cfc8f33d_20230126.pth'
)