import importlib
from typing import Dict

from omegaconf import OmegaConf

from .yaml_cfg import YamlCfgParser
import shutil
import os


class PythonCfgParser(YamlCfgParser):
    def __init__(self):
        super().__init__()
        self.cfg_dict = {}

    def load_cfg(self, path: str):
        # record for save
        if len(self.cfg_dict) == 0:
            self.cfg_dict['cfg.py'] = path
        else:
            self.cfg_dict[path] = path

        if path.endswith('.py'):  # import_module do not need .py suffix
            path = path[:-3]
        path = path.replace('/', '.').replace('\\', '.')
        module = importlib.import_module(path)
        # return OmegaConf.create({k: v for k, v in module.__dict__.items() if not k.startswith('__')})
        return OmegaConf.create(module.config, flags={"allow_objects": True})

    def merge_configs(self, source: Dict, overrides):
        """
        Update a nested dictionary or similar mapping.
        Modify `source` in place.
        """
        for key, value in overrides.items():
            if isinstance(value, dict) and value:
                returned = self.merge_configs(source.get(key, {}), value)
                source[key] = returned
            else:
                source[key] = overrides[key]
        return source

    def load_config(self, path, remove_undefined=True):
        """
        Load the .py format config file to OmegaConf.
        """
        cfg = self.load_cfg(path)
        if '_base_' in cfg:
            for base in cfg['_base_']:
                cfg = OmegaConf.merge(self.load_config(base, remove_undefined=False), cfg)
            del cfg['_base_']
        if remove_undefined:
            cfg = self.remove_config_undefined(cfg)
        return cfg

    def save_configs(self, cfg, path):
        for dst, src in self.cfg_dict.items():
            path_dst = os.path.join(path, dst)
            os.makedirs(os.path.dirname(path_dst), exist_ok=True)
            shutil.copy2(src, path_dst)