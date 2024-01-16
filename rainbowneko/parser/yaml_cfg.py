from typing import Iterable, Any, Dict

from omegaconf import OmegaConf, ListConfig
import os


class YamlCfgParser:

    def load_cfg(self, path: str):
        return OmegaConf.load(path)

    def merge_configs(self, source: Dict, overrides: Dict):
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

    def save_configs(self, cfg, path):
        OmegaConf.save(cfg, os.path.join(path, 'cfg.yaml'))