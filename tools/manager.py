import mim
from pathlib import Path
from mim.utils import get_installed_path, echo_success
from mmengine.config import Config

class Manager:

    def __init__(self, path=None) -> None:

        if path: # root path of projects
            self.path = Path(path)
        else:
            self.path = Path(__file__).parents[1]   
        self.keys = ['weight', 'config', 'model', 'training_data']

    def get_model_infos(self, package_name, keyword: str=None):
        """ because mim search is too strict,
        I want to search by keyword, not a strict match
        """
        model_infos = mim.get_model_info(package_name)
        model_names = model_infos.index
        info_keys = model_infos.columns.tolist()
        keys = self.intersect_keys(info_keys,
                                   self.keys)
        if keyword is None:
            return model_infos[:, keys]
        # get valid names, which contains the keyword
        valid_names = [name for name in model_names
                                 if keyword in name]
        filter_infos = model_infos.loc[valid_names, keys]
        return filter_infos

    def intersect_keys(self, keys1 , keys2):
        return list(set(keys1) & set(keys2))

    def download(self, package, model, config_only=False):
        """ Use model names to download checkpoints and configs.
        Args:
            - package: package name, e.g. mmdet
            - model: model name, e.g. faster_rcnn or faster_rcnn_r50_fpn_1x_coco
            - config_only: only download configs, which is helpful when you 
                            already download checkpoints fast through other ways.
        """
        infos = self.get_model_infos(package, model)
        
        for model, info in infos.iterrows():
            # get destination path
            hyper_name = info['model']
            dst_path = self.path / 'model_zoo' / hyper_name / model
            dst_path.mkdir(parents=True, exist_ok=True)

            if config_only:
                # get config path of the package
                installed_path = Path(get_installed_path(package))
                config_path = info['config']
                config_path = installed_path / '.mim' / config_path
                # build and dump config
                config_obj = Config.fromfile(config_path)
                saved_config_path = dst_path / f'{model}.py'
                config_obj.dump(saved_config_path)
                echo_success(
                    f'Successfully dumped {model}.py to {dst_path}')
            else:
                mim.download(package, [model], dest_root=dst_path)

if __name__ == '__main__':
    m = Manager()
    print(m.get_model_infos('mmdet', 'rtmdet'))
    m.download('mmdet', 'rtmdet_tiny_8xb32-300e_coco', config_only=True)
