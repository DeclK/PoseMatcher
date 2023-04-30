# Tennis.ai

An idea of using deep learning to tutor pose of playing tennis

## Install
### Python Environment
1. Pytorch. Either cpu or gpu version is ok. This project can run ~8fps on CPU, which is acceptable. Here I offer the example command to install 1.11.0+cu113 version.
   ```shell
   pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
   ```
2. OpenMMLab pakcages. Thanks to the rich model zoo of OpenMMLab, I can use them to build this project. In this Project, I've managed to use [RTMDet](https://github.com/open-mmlab/mmdetection/tree/main/configs/rtmdet) and [RTMPose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose) with simple pip install.
   ```shell
   pip install -U openmim
   mim install -r requirements.txt
   ```
## Model Checkpoints & Configs
Because of the messy configs of all kinds of OpenMMLab projects, it is not easy to get the model config by the project. The good news is that OpenMMLab has a great management tool: OpenMIM to solve this situation. It is not perfect, and some links are broken, but it gives you an overview of the model zoo.
I've built a simple Manager to get the model config I need, just checkout `tools/manager.py`

I organize the model_zoo as follow
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

## Highlight

🥳 Manipulate openmmlab models with easy apis
😎 Model can run on CPU with acceptable speed
🔥 Build a FastVisualizer with skimage, which more beatiful than OpenCV
😀 Can be used to all kinds of pose comparison scenes. In my case, tennis pose analysis!