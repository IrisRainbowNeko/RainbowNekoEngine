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
from torch import nn

from .ckpt import NekoModelSaver, NekoPluginSaver


class NekoModelModuleSaver(NekoModelSaver):
    def _get_modules(self, model):
        assert len(self.target_module) == 1, "Only one target module is supported for NekoModelModuleSaver"

        item = self.target_module[0]
        base_model = model if item == '' else eval(f"model.{item}")
        return base_model

    def _save_to(self, name, model: nn.Module, model_ema=None, exclude_key=None, name_template=None):
        sd_model = self._get_modules(model)
        if name_template is not None:
            name_base = name_template.format(name)
        else:
            name_base = name
        self.save(sd_model, name_base, prefix=self.prefix)

        if model_ema is not None:
            sd_ema = self._get_modules(model_ema.model)
            if name_template is not None:
                name_ema = name_template.format(f'{name}-ema')
            else:
                name_ema = name
            self.save(sd_ema, name_ema, prefix=self.prefix)


class NekoPluginModuleSaver(NekoPluginSaver):

    def _save_to(self, name, host_model, plugin_groups: Dict[str, PluginGroup], model_ema=None, exclude_key=None,
                 name_template=None):
        plugin_name = self.target_plugin[0]
        plugin = plugin_groups[plugin_name]
        sd_plugin = {"plugin": plugin}
        if name_template is not None:
            name_base = name_template.format(name)
        else:
            name_base = name
        self.save(sd_plugin, name_base, prefix=self.prefix)

        if model_ema is not None:
            sd_ema = plugin.from_model(model_ema.model)
            if name_template is not None:
                name_ema = name_template.format(f'{name}-ema')
            else:
                name_ema = name
            self.save(sd_ema, name_ema, prefix=self.prefix)
