import torch
from mmengine.registry import MODELS
from mmengine.dataset import Compose, pseudo_collate
from mmengine.model.utils import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmengine.config import Config

from mmdeploy.utils import get_input_shape, load_config
from mmdeploy.apis.utils import build_task_processor

def build_model(cfg, checkpoint=None, device='cpu'):
    """ Build model from config and load checkpoint
    checkpoint_meta usually contains dataset classes information
    """
    if isinstance(cfg, str):
        cfg = Config.fromfile(cfg)
    # scope of model, e.g. mmdet, mmseg, mmpose...
    init_default_scope(cfg.default_scope)
    model = MODELS.build(cfg.model)
    model = revert_sync_batchnorm(model)
    if checkpoint is not None:
        ckpt = load_checkpoint(model, checkpoint,
                            map_location='cpu')
        checkpoint_meta = ckpt.get('meta', {})
        # usually classes and pallate are in checkpoint_meta
        model.checkpoint_meta = checkpoint_meta
    model.to(device)
    model.eval()
    return model

def inference(model, cfg, img):
    """ Given model, config and image, return inference results.
    Models in mmlab does not share the same inference api. So this
    function is just a memo for me...
    """
    if isinstance(cfg, str):
        cfg = Config.fromfile(cfg)
    # process pipline
    test_pipeline = cfg.test_dataloader.dataset.pipeline
    # Use 'LoadImage' to handle both cases of img and img_path
    # This is specially designed for mmdet config, which uses 'LoadImageFromFile'
    for pipeline in test_pipeline:
        if 'LoadImage' in pipeline['type']:
            pipeline['type'] = 'mmpose.LoadImage'

    init_default_scope(cfg.default_scope)
    pipeline = Compose(test_pipeline)

    if isinstance(img, str):
        # img_id is useless...but to be compatible with mmdet
        data_info = dict(img_path=img, img_id=0)
    else:
        data_info = dict(img=img, img_id=0)

    data = pipeline(data_info)
    batch = pseudo_collate([data])
    
    with torch.no_grad():
        results = model.test_step(batch)

    return results

def build_onnx_model_and_task_processor(model_cfg, deploy_cfg, backend_files, device):

    deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

    task_processor = build_task_processor(model_cfg, deploy_cfg, device)

    model = task_processor.build_backend_model(
        backend_files, task_processor.update_data_preprocessor)

    return model, task_processor

def inference_onnx_model(model, task_processor, deploy_cfg, img):
    input_shape = get_input_shape(deploy_cfg)
    model_inputs, _ = task_processor.create_input(img, input_shape)
    
    with torch.no_grad():
        result = model.test_step(model_inputs)

    return result
    
if __name__ == '__main__':
    config = '/github/Tennis.ai/model_zoo/rtmpose/rtmpose-t_8xb256-420e_aic-coco-256x192/rtmpose-t_8xb256-420e_aic-coco-256x192.py'
    ckpt = '/github/Tennis.ai/model_zoo/rtmpose/rtmpose-t_8xb256-420e_aic-coco-256x192/rtmpose-tiny_simcc-aic-coco_pt-aic-coco_420e-256x192-cfc8f33d_20230126.pth'
    img = '/github/Tennis.ai/assets/000000197388.jpg'

    detector = build_model(config, checkpoint=ckpt)
    result = inference(detector, config, img)