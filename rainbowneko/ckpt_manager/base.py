from typing import Dict

from rainbowneko.models.plugin import PluginGroup
from torch import nn

from .format import CkptFormat
from .source import LocalCkptSource


class CkptManagerBase:
    def __init__(self, format: CkptFormat, source: LocalCkptSource, plugin_from_raw=False,
                 saved_model=({'model': '', 'trainable': True},), **kwargs):
        self.plugin_from_raw = plugin_from_raw
        self.saved_model = saved_model

        self.format = format
        self.source = source

    def save_step(self, model: nn.Module, name, step, prefix=None, model_ema=None, exclude_key=None):
        self.save(model, f"{name}-{step}", prefix, model_ema, exclude_key)

    def save(self, model: nn.Module, name, prefix=None, model_ema=None, exclude_key=None):
        raise NotImplementedError

    def load(cls, model_f, **kwargs):
        raise NotImplementedError

    def save_plugins(self, host_model: nn.Module, plugins: Dict[str, PluginGroup], name: str, prefix=None, model_ema=None):
        raise NotImplementedError
    
    def save_plugins_step(self, host_model: nn.Module, plugins: Dict[str, PluginGroup], name: str, step: int, prefix=None, model_ema=None):
        self.save_plugins(host_model, plugins, f"{name}-{step}", prefix, model_ema)

    def __repr__(self):
        return f"{self.__class__.__name__}(format={self.format}, source={self.source})"