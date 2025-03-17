from typing import Iterable, Any, Dict

from omegaconf import OmegaConf, ListConfig, DictConfig
import os


class YamlCfgParser:

    def load_cfg(self, path: str):
        return OmegaConf.load(path)

    def merge_configs(self, source: Dict, overrides: Dict):
        self.remove_replace(source, overrides)
        return OmegaConf.merge(source, overrides)

    def remove_config_undefined(self, cfg):
        itr: Iterable[Any] = range(len(cfg)) if isinstance(cfg, ListConfig) else cfg

        undefined_keys = []
        for key in itr:
            if cfg._get_child(key) == '---':
                undefined_keys.append(key)
            elif OmegaConf.is_config(cfg[key]):
                self.remove_config_undefined(cfg[key])
        for key in undefined_keys:
            del cfg[key]
        return cfg

    def remove_replace(self, base_cfg, cfg):
        if isinstance(cfg, ListConfig) or base_cfg is None:
            return base_cfg

        replace_keys = []
        for key in cfg:
            child = cfg._get_child(key)
            if isinstance(child, DictConfig):
                if child._get_child('_replace_'):
                    if key in base_cfg:
                        replace_keys.append(key)
                    continue
                else:
                    if key in base_cfg:
                        self.remove_replace(base_cfg[key], child)
                    else:
                        continue
        for key in replace_keys:
            del base_cfg[key]
            del cfg[key]['_replace_']
        return base_cfg

    def load_config(self, path, remove_undefined=True):
        cfg = self.load_cfg(path)
        if '_base_' in cfg:
            for base in cfg['_base_']:
                cfg = self.merge_configs(self.load_config(base, remove_undefined=False), cfg)
            del cfg['_base_']
        if remove_undefined:
            cfg = self.remove_config_undefined(cfg)
        return cfg

    def load_config_with_cli(self, path, args_list=None, remove_undefined=True):
        cfg = self.load_config(path, remove_undefined=False)
        cfg_cli = OmegaConf.from_cli(args_list)
        cfg = OmegaConf.merge(cfg, cfg_cli)
        if remove_undefined:
            cfg = self.remove_config_undefined(cfg)
        return cfg

    def save_configs(self, cfg, path, name='cfg'):
        OmegaConf.save(cfg, os.path.join(path, f'{name}.yaml'))