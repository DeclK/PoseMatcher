from mmengine.registry import count_registered_modules, init_default_scope
import mmdet.models

from mmengine.registry import MODELS

def build_model(cfg):
    # scope of model, e.g. mmdet, mmseg, mmpose...
    scope = cfg.default_scope
    init_default_scope(scope)
    model = MODELS.build(cfg)
    return model
    