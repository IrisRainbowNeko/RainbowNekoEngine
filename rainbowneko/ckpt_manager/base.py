from typing import Dict, Any

from torch import nn

from rainbowneko.models.plugin import PluginGroup
from .format import CkptFormat, SafeTensorFormat
from .source import LocalCkptSource

LAYERS_ALL = 'all'
LAYERS_TRAINABLE = 'trainable'


class NekoLoader:
    def __init__(self, format: CkptFormat = None, source: LocalCkptSource = None, layers='all'):
        if format is None:
            format = SafeTensorFormat()
        if source is None:
            source = LocalCkptSource()

        self.format = format
        self.source = source
        self.layers = layers

    def load(self, path, ext=None, **kwargs):
        return self.source.get(path, self.format, **kwargs)

    def load_to(self, name, model):
        raise NotImplementedError()

    @staticmethod
    def load_all(model: nn.Module, cfg: Dict[str, "NekoLoader"]):
        for name, loader in cfg.items():
            loader.load_to(name, model)


class NekoSaver:
    def __init__(self, format: CkptFormat = None, source: LocalCkptSource = None, layers='all', state_prefix=''):
        if format is None:
            format = SafeTensorFormat()
        if source is None:
            source = LocalCkptSource()
        self.format = format
        self.source = source
        self.layers = layers
        self.state_prefix = state_prefix

    def clean_prefix(self, state_dict: Dict[str, Any]):
        return {k.removeprefix(self.state_prefix): v for k, v in state_dict.items() if k.startswith(self.state_prefix)}

    def save(self, state_dict: Dict[str, Any], name, prefix=None):
        self.source.put(f"{name}.{self.format.EXT}", state_dict, self.format, prefix=prefix)

    def save_to(self, name, model: nn.Module, plugin_groups: Dict[str, PluginGroup], model_ema=None, exclude_key=None,
                name_template=None):
        raise NotImplementedError

    @staticmethod
    def save_all(model: nn.Module, plugin_groups: Dict[str, PluginGroup], cfg: Dict[str, "NekoSaver"], model_ema=None,
                 exclude_key=None, name_template=None):
        for name, loader in cfg.items():
            loader.save_to(name, model, plugin_groups, model_ema=model_ema, exclude_key=exclude_key, name_template=name_template)
