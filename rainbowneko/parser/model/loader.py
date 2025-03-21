import torch
from torch import nn
from rainbowneko.ckpt_manager import CkptManagerBase, auto_manager
from typing import Dict, Any

from .locator import get_match_layers


class NekoLoader:

    def load_to(self, name, model):
        raise NotImplementedError()

    @staticmethod
    def load_all(model: nn.Module, cfg:Dict[str, "NekoLoader"]):
        for name, loader in cfg.items():
            loader.load_to(name, model)


class NekoModelLoader(NekoLoader):
    def __init__(self, path: str, ckpt_manager: CkptManagerBase = None, layers='all', module_to_load='', state_prefix=None,
                 base_model_alpha=0.0, alpha=1.0, load_ema=False):
        self.path = path
        self.ckpt_manager = ckpt_manager or auto_manager(path)
        self.layers = layers
        self.module_to_load = module_to_load
        self.state_prefix = state_prefix
        self.base_model_alpha = base_model_alpha
        self.alpha = alpha
        self.load_ema = load_ema

    def load_to(self, name, model):
        model = model if self.module_to_load == '' else eval(f"model.{self.module_to_load}")
        named_modules = {k: v for k, v in model.named_modules()}
        named_params = {k: v for k, v in model.named_parameters()}
        named_params.update({k: v for k, v in model.named_buffers()})

        part_state = self.ckpt_manager.load(self.path, map_location='cpu')
        if self.load_ema:
            part_state = part_state['base_ema']
        else:
            if 'base' in part_state:
                part_state = part_state['base']

        if self.state_prefix:
            state_prefix_len = len(self.state_prefix)
            part_state = {k[state_prefix_len:]: v for k, v in part_state.items() if k.startswith(self.state_prefix)}

        if self.layers == 'all':
            for k, v in part_state.items():
                named_params[k].data = self.base_model_alpha * named_params[k].data + self.alpha * v.to(named_params[k].data.device)
        else:
            match_blocks = get_match_layers(self.layers, named_modules)
            state_add = {k: v for blk in match_blocks for k, v in part_state.items() if k.startswith(blk)}
            for k, v in state_add.items():
                named_params[k].data = self.base_model_alpha * named_params[k].data + self.alpha * v.to(named_params[k].data.device)

class NekoPluginLoader(NekoLoader):
    def __init__(self, path: str, ckpt_manager: CkptManagerBase = None, layers='all', module_to_load='', state_prefix=None,
                 base_model_alpha=0.0, load_ema=False, **plugin_kwargs):
        self.path = path
        self.ckpt_manager = ckpt_manager or auto_manager(path)
        self.layers = layers
        self.module_to_load = module_to_load
        self.state_prefix = state_prefix
        self.base_model_alpha = base_model_alpha
        self.load_ema = load_ema

        self.plugin_kwargs = plugin_kwargs

    def load_to(self, name, model):
        # get model to load plugin and its named_modules
        model = model if self.module_to_load == '' else eval(f"model.{self.module_to_load}")
        named_modules = {k: v for k, v in model.named_modules()}

        plugin_state = self.ckpt_manager.load(self.path, map_location='cpu')['plugin_ema' if self.load_ema else 'plugin']

        if self.state_prefix:
            state_prefix_len = len(self.state_prefix)
            plugin_state = {k[state_prefix_len:]: v for k, v in plugin_state.items() if k.startswith(self.state_prefix)}

        # filter layers to load
        if self.layers != 'all':
            match_blocks = get_match_layers(self.layers, named_modules)
            plugin_state = {k: v for blk in match_blocks for k, v in plugin_state.items() if k.startswith(blk)}

        # state to plugin_block_state
        plugin_block_state = {}
        for pname, p in plugin_state.items():
            prefix, block_name = pname.split('.___.', 1)
            plugin_block_state.setdefault(f'{prefix}.{name}', {})[block_name] = p

        # Load state to plugin
        plugin_state = {k.replace('___', name): v for k, v in
                        plugin_state.items()}  # replace placeholder to target plugin name
        load_info = model.load_state_dict(plugin_state, strict=False)
        if len(load_info.unexpected_keys) > 0:
            print(name, 'unexpected_keys', load_info.unexpected_keys)

        # set plugin hyper params or build plugin
        if hasattr(model, name):  # MultiPluginBlock
            getattr(model, name).set_hyper_params(**self.plugin_kwargs)
        else:
            for plugin_key in plugin_block_state.keys():
                named_modules[plugin_key].set_hyper_params(**self.plugin_kwargs)
