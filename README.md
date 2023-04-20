# Tennis.ai

An idea of using deep learning to tutor pose of playing tennis

# Usage
## Model Checkpoints & Configs
Download the checkpoints and configs to the current folder

Because of the messy configs of MMLab, it is not easy to get the model config by the project.

However, OpenMMLab offers a great management tool: OpenMIM to solve this situation.

We need to install corresponding projects through pip, which is easy
```shell
pip install -U openmim
mim install mmdet
mim install mmpose
```

1. [RTMDet](https://github.com/open-mmlab/mmdetection/tree/main/configs/rtmdet)
2. [RTMPose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose)

I organize the model_zoo as 
```python

- model_zoo
    - model_hyper_name_1        # like rtmdet
        - model_1               # like rtmdet_m_8xb32_coco
            checkpoint
            config
        - model_2
            checkpoint
            config
    - model_hyper_name_1
        ...
```

# Purpose

1. In this program, I want to explore the easy way to manipulate openmmlab models, which need me to explore a way to use registry from different projects.
2. I want to build a minimum example of converting pytorch models to onnx.
3. Use this to support tennis sports.