python tools/deploy.py \
    model_zoo/rtmdet/rtmdet_tiny_8xb32-300e_coco/detection_onnxruntime_static.py \
    model_zoo/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco.py \
    model_zoo/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth \
    assets/onnx_test.jpg \
    --work-dir model_zoo/rtmdet/rtmdet_tiny_8xb32-300e_coco \
    --device cpu \
    --show