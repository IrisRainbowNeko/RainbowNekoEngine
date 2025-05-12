from typing import Dict, Any

from torch import nn

from rainbowneko.models.plugin import PluginGroup
from rainbowneko.utils import KeyMapper
from .format import CkptFormat, SafeTensorFormat
from .source import LocalCkptSource

LAYERS_ALL = 'all'
LAYERS_TRAINABLE = 'trainable'


class NekoLoader:
    def __init__(self, format: CkptFormat = None, source: LocalCkptSource = None, layers='all', key_map=None):
        if format is None:
            format = SafeTensorFormat()
        if source is None:
            source = LocalCkptSource()

        self.format = format
        self.source = source
        self.layers = layers
        self.key_mapper = KeyMapper(key_map=key_map)

    def load(self, path, ext=None, **kwargs):
        return self.source.get(path, self.format, **kwargs)

    def load_to(self, name, **kwargs):
        _, info = self.key_mapper(**kwargs)
        return self._load_to(name, **info)

    def _load_to(self, name, model):
        raise NotImplementedError()

    @staticmethod
    def load_all(cfg: Dict[str, "NekoLoader"], **kwargs):
        '''
        :param cfg:
        :param kwargs:
            model: nn.Module
            plugin_groups: Dict[str, PluginGroup]
            optimizer: Optimizer
            ...
        :return:
        '''

        for name, loader in cfg.items():
            loader.load_to(name, **kwargs)


class NekoSaver:
    def __init__(self, format: CkptFormat = None, source: LocalCkptSource = None, layers='all', state_prefix='', key_map=None):
        if format is None:
            format = SafeTensorFormat()
        if source is None:
            source = LocalCkptSource()
        self.format = format
        self.source = source
        self.layers = layers
        self.state_prefix = state_prefix
        self.key_mapper = KeyMapper(key_map=key_map)

    def clean_prefix(self, state_dict: Dict[str, Any]):
        return {k.removeprefix(self.state_prefix): v for k, v in state_dict.items() if k.startswith(self.state_prefix)}

    def save(self, state_dict: Dict[str, Any], name, prefix=None):
        self.source.put(f"{name}.{self.format.EXT}", state_dict, self.format, prefix=prefix)

    def save_to(self, name, **kwargs):
        _, info = self.key_mapper(**kwargs)
        return self._save_to(name, **info)

    def _save_to(self, name, model: nn.Module, exclude_key=None, name_template=None):
        raise NotImplementedError

    @staticmethod
    def save_all(cfg: Dict[str, "NekoSaver"], exclude_key=None, name_template=None, **kwargs):
        '''
        :param cfg:
        :param exclude_key:
        :param name_template:
        :param kwargs:
            model: nn.Module
            plugin_groups: Dict[str, PluginGroup]
            model_ema: ModelEMA
            optimizer: Optimizer
        :return:
        '''
        for name, loader in cfg.items():
            loader.save_to(name, exclude_key=exclude_key, name_template=name_template, **kwargs)
