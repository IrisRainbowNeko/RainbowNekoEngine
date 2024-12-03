"""
ckpt_pkl.py
====================
    :Name:        save model with torch
    :Author:      Dong Ziyi
    :Affiliation: HCP Lab, SYSU
    :Created:     8/04/2023
    :Licence:     MIT
"""

import os
from typing import Dict, Union

import torch
from torch import nn

from rainbowneko.models.plugin import PluginGroup, BasePluginBlock
from rainbowneko.models.wrapper import BaseWrapper
from .base import CkptManagerBase


class CkptManagerPKL(CkptManagerBase):
    def __init__(self, plugin_from_raw=False, saved_model=({'model':'', 'trainable':True},), **kwargs):
        self.plugin_from_raw = plugin_from_raw
        self.saved_model = saved_model

    def set_save_dir(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

    def exclude_state(self, state, key):
        if key is None:
            return state
        else:
            return {k: v for k, v in state.items() if key not in k}

    def save_model(self, model: nn.Module, name, step, model_ema=None, exclude_key=None):
        sd_base = {}
        for item in self.saved_model:
            block = model if item['model']=='' else eval(f"model.{item['model']}")
            sd_base.update(self.exclude_state(
                BasePluginBlock.extract_state_without_plugin(block, trainable=item['trainable']), exclude_key
            ))

        sd_model = {"base": sd_base}
        if model_ema is not None:
            sd_ema = model_ema.state_dict()
            sd_ema = {k: sd_ema[k] for k in sd_model["base"].keys()}
            sd_model["base_ema"] = self.exclude_state(sd_ema, exclude_key)
        self._save_ckpt(sd_model, name, step)

    def save_plugins(
            self, host_model: nn.Module, plugins: Dict[str, PluginGroup], name: str, step: int, model_ema=None
    ):
        # TODO: save plugins with saved_model
        if len(plugins) > 0:
            sd_plugin = {}
            for plugin_name, plugin in plugins.items():
                sd_plugin["plugin"] = plugin.state_dict(host_model if self.plugin_from_raw else None)
                if model_ema is not None:
                    sd_plugin["plugin_ema"] = plugin.state_dict(model_ema)
                self._save_ckpt(sd_plugin, f"{name}-{plugin_name}", step)

    def _save_ckpt(self, sd_model, name=None, step=None, save_path=None):
        if save_path is None:
            save_path = os.path.join(self.save_dir, f"{name}-{step}.ckpt")
        torch.save(sd_model, save_path)

    def load_ckpt(self, ckpt_path, map_location="cpu"):
        return torch.load(ckpt_path, map_location=map_location)

    def load_ckpt_to_model(self, model: nn.Module, ckpt_path, model_ema=None):
        sd = self.load_ckpt(ckpt_path)
        if "base" in sd:
            for item in self.saved_model:
                block = model if item['model'] == '' else eval(f"model.{item['model']}")
                missing_keys, unexpected_keys = block.load_state_dict(sd["base"], strict=False)
                if len(missing_keys)>0:
                    print(f'missing_keys of model.{item["model"]}: {missing_keys}')
        if "plugin" in sd:
            model.load_state_dict(sd["plugin"], strict=False)

        if model_ema is not None:
            if "base" in sd:
                for item in self.saved_model:
                    prefix = None if item['model'] == '' else item['model']
                    model_ema.load_state_dict(sd["base_ema"], prefix=prefix)
            if "plugin" in sd:
                model_ema.load_state_dict(sd["plugin_ema"])

    def save(self, name, step, model, all_plugin, ema={}, **kwargs):
        """

        :param step: current model train step
        :param model: Model Wrapper for training
        :param all_plugin: all plugins
        :param ema: model or plugins ema {'model':..., 'plugin':...}
        :return:
        """
        if len(kwargs) > 0:
            print(f'Unused kwargs in save model: {", ".join(kwargs.keys())}')

        self.save_model(model, model_ema=getattr(ema, "model", None), name=name, step=step)
        self.save_plugins(model, all_plugin, name=name, step=step, model_ema=getattr(self, "model", None))
