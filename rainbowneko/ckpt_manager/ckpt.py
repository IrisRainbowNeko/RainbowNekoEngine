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

from rainbowneko.models.plugin import PluginGroup, BasePluginBlock
from torch import nn

from .base import CkptManagerBase


class CkptManager(CkptManagerBase):

    def exclude_state(self, state, key):
        if key is None:
            return state
        else:
            return {k: v for k, v in state.items() if key not in k}

    def save(self, model: nn.Module, name, prefix=None, model_ema=None, exclude_key=None):
        sd_base = {}
        for item in self.saved_model:
            block = model if item['model'] == '' else eval(f"model.{item['model']}")
            sd_base.update(self.exclude_state(
                BasePluginBlock.extract_state_without_plugin(block, trainable=item['trainable']), exclude_key
            ))

        sd_model = {"base": sd_base}
        if model_ema is not None:
            sd_ema = model_ema.state_dict()
            sd_ema = {k: sd_ema[k] for k in sd_model["base"].keys()}
            sd_model["base_ema"] = self.exclude_state(sd_ema, exclude_key)
        self.source.put(f"{name}.{self.format.EXT}", sd_model, self.format, prefix=prefix)

    def save_plugins(self, host_model: nn.Module, plugins: Dict[str, PluginGroup], name: str, step: int, model_ema=None):
        if len(plugins) > 0:
            sd_plugin = {}
            for plugin_name, plugin in plugins.items():
                sd_plugin["plugin"] = plugin.state_dict(host_model if self.plugin_from_raw else None)
                if model_ema is not None:
                    sd_plugin["plugin_ema"] = plugin.state_dict(model_ema)
                self.source.put(f"{name}-{plugin_name}-{step}.{self.format.EXT}", sd_plugin, self.format)

    def load(self, name, ext=None, **kwargs):
        return self.source.get(name, self.format, **kwargs)
        # if len(postfix := name.rsplit(".", 1)) == 2 and postfix[1].isalpha():
        #     return self.source.get(name, self.format, **kwargs)
        # else:
        #     ext = ext or self.format.EXT
        #     return self.source.get(f"{name}.{ext}", self.format, **kwargs)
