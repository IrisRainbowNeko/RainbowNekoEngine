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

from rainbowneko.models.ema import ModelEMA
from rainbowneko.models.plugin import PluginGroup
from torch import nn

from .ckpt import NekoModelSaver, NekoPluginSaver


class NekoModelModuleSaver(NekoModelSaver):
    def _get_modules(self, model):
        assert len(self.target_module) == 1, "Only one target module is supported for NekoModelModuleSaver"

        item = self.target_module[0]
        base_model = model if item == '' else eval(f"model.{item}")
        return base_model

    def save_to(self, name, model: nn.Module, plugin_groups: Dict[str, PluginGroup], model_ema: ModelEMA = None, exclude_key=None,
                name_template=None):
        sd_model = {"base": self._get_modules(model)}
        if model_ema is not None:
            sd_model["base_ema"] = self._get_modules(model_ema.model)

        if name_template is not None:
            name = name_template.format(name)
        self.save(sd_model, name, prefix=self.prefix)


class NekoPluginModuleSaver(NekoPluginSaver):

    def save_to(self, name, model: nn.Module, plugin_groups: Dict[str, PluginGroup], model_ema: ModelEMA = None, exclude_key=None,
                name_template=None):
        plugin_name = self.target_plugin[0]
        plugin = plugin_groups[plugin_name]
        sd_plugin = {"plugin": plugin}
        if model_ema is not None:
            sd_plugin["plugin_ema"] = plugin.from_model(model_ema.model)

        if name_template is not None:
            name = name_template.format(name)
        self.save(sd_plugin, name, prefix=self.prefix)
