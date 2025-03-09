"""
ckpt_pkl.py
====================
    :Name:        save model with torch
    :Author:      Dong Ziyi
    :Affiliation: HCP Lab, SYSU
    :Created:     8/04/2023
    :Licence:     MIT
"""

from typing import Dict

from rainbowneko.models.plugin import PluginGroup
from rainbowneko.models.ema import ModelEMA
from torch import nn

from .base import CkptManagerBase


class ModelManager(CkptManagerBase):

    def _get_modules(self, model):
        if len(self.saved_model) == 1:
            item = self.saved_model[0]
            base_model = model if item['model'] == '' else eval(f"model.{item['model']}")
        else:
            base_model = {}
            for item in self.saved_model:
                base_model[item['model']] = model if item['model'] == '' else eval(f"model.{item['model']}")
        return base_model

    def save(self, model: nn.Module, name, prefix=None, model_ema:ModelEMA=None, exclude_key=None):
        sd_model = {"base": self._get_modules(model)}
        if model_ema is not None:
            sd_model["base_ema"] = self._get_modules(model_ema.model)

        self.source.put(f"{name}.{self.format.EXT}", sd_model, self.format, prefix=prefix)

    def save_plugins(self, host_model: nn.Module, plugins: Dict[str, PluginGroup], name: str, step: int, model_ema:ModelEMA=None):
        if len(plugins) > 0:
            for plugin_name, plugin in plugins.items():
                sd_plugin = {"plugin": plugin}
                if model_ema is not None:
                    sd_plugin["plugin_ema"] = plugin.from_model(model_ema.model)
                self.source.put(f"{name}-{plugin_name}-{step}.{self.format.EXT}", sd_plugin, self.format)

    def load(self, name, ext=None, **kwargs) -> nn.Module:
        return self.source.get(name, self.format, **kwargs)
