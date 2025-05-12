from typing import Dict, Union, List

from rainbowneko.models.plugin import PluginGroup, BasePluginBlock
from torch import nn
from torch.optim import Optimizer

from .base import NekoLoader, NekoSaver, LAYERS_ALL, LAYERS_TRAINABLE
from .format import CkptFormat
from .locator import get_match_layers
from .source import LocalCkptSource


class NekoModelLoader(NekoLoader):
    def __init__(self, format: CkptFormat = None, source: LocalCkptSource = None, path: str = None, layers='all',
                 target_module='', state_prefix=None, base_model_alpha=0.0, alpha=1.0, load_ema=False, key_map=None):
        key_map = key_map or ('name -> name', 'model -> model')
        super().__init__(format=format, source=source, layers=layers, key_map=key_map)
        self.path = path

        self.target_module = target_module
        self.base_model_alpha = base_model_alpha
        self.alpha = alpha
        self.load_ema = load_ema
        self.state_prefix = state_prefix

    def _load_to(self, name, model):
        model = model if self.target_module == '' else eval(f"model.{self.target_module}")
        states = model.state_dict()

        part_state = self.load(self.path, map_location='cpu')
        if self.load_ema and 'base_ema' in part_state:
            part_state = part_state['base_ema']
        else:
            if 'base' in part_state:
                part_state = part_state['base']

        if self.state_prefix:
            state_prefix_len = len(self.state_prefix)
            part_state = {k[state_prefix_len:]: v for k, v in part_state.items() if k.startswith(self.state_prefix)}

        if self.layers == LAYERS_ALL:
            sd_data = {k: self.base_model_alpha * states[k] + self.alpha * v.to(states[k].device) for k, v in part_state.items()}
            model.load_state_dict(sd_data, strict=False)
        else:
            named_modules = {k: v for k, v in model.named_modules()}
            match_blocks = get_match_layers(self.layers, named_modules)
            state_add = {k: v for blk in match_blocks for k, v in part_state.items() if k.startswith(blk)}
            sd_data = {k: self.base_model_alpha * states[k] + self.alpha * v.to(states[k].device) for k, v in state_add.items()}
            model.load_state_dict(sd_data, strict=False)


class NekoModelSaver(NekoSaver):
    def __init__(self, format: CkptFormat = None, source: LocalCkptSource = None, layers='all', state_prefix='',
                 target_module: Union[str, List[str]] = '', prefix=None, key_map=None):
        key_map = key_map or ('name -> name', 'model -> model', 'model_ema -> model_ema', 'exclude_key -> exclude_key',
                              'name_template -> name_template')
        super().__init__(format=format, source=source, layers=layers, state_prefix=state_prefix, key_map=key_map)
        self.prefix = prefix
        if isinstance(target_module, str):
            target_module = [target_module]
        self.target_module = target_module

    def exclude_state(self, state, key):
        if key is None:
            return state
        else:
            return {k: v for k, v in state.items() if key not in k}

    def _save_to(self, name, model: nn.Module, model_ema=None, exclude_key=None, name_template=None):
        sd_base = {}
        for item in self.target_module:
            block = model if item == '' else eval(f"model.{item}")
            sd_item = self.exclude_state(
                BasePluginBlock.extract_state_without_plugin(block, trainable=self.layers == LAYERS_TRAINABLE), exclude_key
            )

            # filter layers
            if not isinstance(self.layers, str):
                named_modules = {k: v for k, v in model.named_modules()}
                match_blocks = get_match_layers(self.layers, named_modules)
                sd_item = {k: v for blk in match_blocks for k, v in sd_item.items() if k.startswith(blk)}

            sd_base.update(sd_item)

        if model_ema is not None:
            sd_ema = model_ema.state_dict()
            sd_ema = {k: sd_ema[k] for k in sd_base.keys()}
            sd_ema = self.exclude_state(sd_ema, exclude_key)
            sd_ema = self.clean_prefix(sd_ema)

            if name_template is not None:
                name_ema = name_template.format(f'{name}-ema')
            else:
                name_ema = name
            self.save(sd_ema, name_ema, prefix=self.prefix)

        if len(sd_base) > 0:
            sd_base = self.clean_prefix(sd_base)
            if name_template is not None:
                name_base = name_template.format(name)
            else:
                name_base = name
            self.save(sd_base, name_base, prefix=self.prefix)


class NekoPluginLoader(NekoLoader):
    def __init__(self, format: CkptFormat = None, source: LocalCkptSource = None, path: str = None, layers='all',
                 target_plugin=None, state_prefix=None, base_model_alpha=0.0, load_ema=False, key_map=None, **plugin_kwargs):
        key_map = key_map or ('name -> name', 'plugin_groups -> plugin_groups')
        super().__init__(format=format, source=source, layers=layers, key_map=key_map)
        self.path = path

        self.target_plugin = target_plugin
        self.state_prefix = state_prefix
        self.base_model_alpha = base_model_alpha
        self.load_ema = load_ema

        self.plugin_kwargs = plugin_kwargs

    def _load_to(self, name: str, plugin_groups: Dict[str, PluginGroup]):
        # get plugin_group to load
        plugin_group = plugin_groups[self.target_plugin or name]
        state_dict = self.load(self.path, map_location='cpu')
        if 'base' in state_dict or 'base_ema' in state_dict:
            plugin_state = state_dict['base_ema' if self.load_ema else 'base']
        elif 'plugin' in state_dict or 'plugin_ema' in state_dict:
            plugin_state = state_dict['plugin_ema' if self.load_ema else 'plugin']
        else:
            plugin_state = state_dict

        if self.state_prefix:
            state_prefix_len = len(self.state_prefix)
            plugin_state = {k[state_prefix_len:]: v for k, v in plugin_state.items() if k.startswith(self.state_prefix)}

        # filter layers to load
        plugin_dict = plugin_group.plugin_dict
        if self.layers != LAYERS_ALL:
            plugin_dict = get_match_layers(self.layers, plugin_dict)

        # state to plugin_block_state
        plugin_block_state = {}
        for pname, p in plugin_state.items():
            prefix, block_name = pname.split('.___.', 1)
            plugin_block_state.setdefault(prefix, {})[block_name] = p

        # Load state to plugin
        missing_keys = []
        unexpected_keys = []
        for key, plugin in plugin_dict.items():
            if key not in plugin_block_state:
                missing_keys.append(key)
            else:
                load_info = plugin.load_state_dict(plugin_block_state.pop(key), strict=False)
                plugin.set_hyper_params(**self.plugin_kwargs)
                missing_keys.extend(load_info.missing_keys)
                unexpected_keys.extend(load_info.unexpected_keys)


class NekoPluginSaver(NekoSaver):
    def __init__(self, format: CkptFormat = None, source: LocalCkptSource = None, layers='all', state_prefix='',
                 target_plugin: Union[str, List[str]] = '', prefix=None, plugin_from_raw=False, key_map=None):
        key_map = key_map or ('name -> name', 'model -> host_model', 'plugin_groups -> plugin_groups', 'model_ema -> model_ema',
                              'exclude_key -> exclude_key', 'name_template -> name_template')
        super().__init__(format=format, source=source, layers=layers, state_prefix=state_prefix, key_map=key_map)
        self.prefix = prefix
        if isinstance(target_plugin, str):
            target_plugin = [target_plugin]
        self.target_plugin = target_plugin
        self.plugin_from_raw = plugin_from_raw

    def _save_to(self, name, host_model, plugin_groups: Dict[str, PluginGroup], model_ema=None, exclude_key=None,
                 name_template=None):
        sd_base = {}
        sd_ema = {}
        for plugin_name in self.target_plugin:
            plugin = plugin_groups[plugin_name]
            sd_item = plugin.state_dict(host_model if self.plugin_from_raw else None)

            # filter layers
            if not isinstance(self.layers, str):
                match_blocks = get_match_layers(self.layers, plugin.plugin_dict)
                sd_item = {k: v for blk in match_blocks for k, v in sd_item.items() if k.startswith(blk)}

            sd_base.update(sd_item)

            if model_ema is not None:
                sd_ema_item = plugin.state_dict(model_ema)
                sd_ema_item = {k: sd_ema_item[k] for k in sd_item.keys()}
                sd_ema.update(sd_ema_item)

        if len(sd_base) > 0:
            sd_base = self.clean_prefix(sd_base)
            if name_template is not None:
                name_base = name_template.format(name)
            else:
                name_base = name
            self.save(sd_base, name_base, prefix=self.prefix)

        if model_ema is not None:
            if name_template is not None:
                name_ema = name_template.format(f'{name}-ema')
            else:
                name_ema = name
            sd_ema = self.clean_prefix(sd_ema)
            self.save(sd_ema, name_ema, prefix=self.prefix)


class NekoEasySaver(NekoSaver):
    def __init__(self, format: CkptFormat = None, source: LocalCkptSource = None, layers='all', state_prefix='',
                 prefix=None, plugin_from_raw=False):
        super().__init__(format=format, source=source, layers=layers, state_prefix=state_prefix)

        self.prefix = prefix
        self.plugin_from_raw = plugin_from_raw

    def exclude_state(self, state, key):
        if key is None:
            return state
        else:
            return {k: v for k, v in state.items() if key not in k}

    def model_save_to(self, name, model: nn.Module, model_ema=None, exclude_key=None, name_template=None):
        sd_base = {}
        sd_item = self.exclude_state(
            BasePluginBlock.extract_state_without_plugin(model, trainable=self.layers == LAYERS_TRAINABLE), exclude_key
        )

        # filter layers
        if not isinstance(self.layers, str):
            named_modules = {k: v for k, v in model.named_modules()}
            match_blocks = get_match_layers(self.layers, named_modules)
            sd_item = {k: v for blk in match_blocks for k, v in sd_item.items() if k.startswith(blk)}

        sd_base.update(sd_item)

        if model_ema is not None:
            sd_ema = model_ema.state_dict()
            sd_ema = {k: sd_ema[k] for k in sd_base.keys()}
            sd_ema = self.exclude_state(sd_ema, exclude_key)
            sd_ema = self.clean_prefix(sd_ema)

            if name_template is not None:
                name_ema = name_template.format(f'{name}-ema')
            else:
                name_ema = name
            self.save(sd_ema, name_ema, prefix=self.prefix)

        if len(sd_base) > 0:
            sd_base = self.clean_prefix(sd_base)
            if name_template is not None:
                name_base = name_template.format(name)
            else:
                name_base = name
            self.save(sd_base, name_base, prefix=self.prefix)

    def plugin_save_to(self, name, host_model, plugin_groups: Dict[str, PluginGroup], model_ema=None, exclude_key=None,
                       name_template=None):
        for plugin_name, plugin in plugin_groups:
            sd_base = plugin.state_dict(host_model if self.plugin_from_raw else None)

            # filter layers
            # TODO: filter trainable
            if not isinstance(self.layers, str):
                match_blocks = get_match_layers(self.layers, plugin.plugin_dict)
                sd_base = {k: v for blk in match_blocks for k, v in sd_base.items() if k.startswith(blk)}

            if model_ema is not None:
                sd_ema = plugin.state_dict(model_ema)
                sd_ema = {k: sd_ema[k] for k in sd_base.keys()}

            if len(sd_base) > 0:
                sd_base = self.clean_prefix(sd_base)
                if name_template is not None:
                    name_base = name_template.format(name)
                else:
                    name_base = name
                self.save(sd_base, name_base, prefix=self.prefix)

            if model_ema is not None:
                if name_template is not None:
                    name_ema = name_template.format(f'{name}-ema')
                else:
                    name_ema = name
                sd_ema = self.clean_prefix(sd_ema)
                self.save(sd_ema, name_ema, prefix=self.prefix)

    def _save_to(self, name, model, plugin_groups: Dict[str, PluginGroup], model_ema=None, exclude_key=None, name_template=None):
        self.model_save_to(name, model, model_ema, exclude_key, name_template)
        self.plugin_save_to(name, model, plugin_groups, model_ema, exclude_key, name_template)


class NekoOptimizerSaver(NekoSaver):
    def __init__(self, format: CkptFormat = None, source: LocalCkptSource = None, state_prefix='',
                 target_module: Union[str, List[str]] = '', prefix=None, key_map=None):
        key_map = key_map or ('name -> name', 'optimizer -> optimizer', 'name_template -> name_template')
        super().__init__(format=format, source=source, state_prefix=state_prefix, key_map=key_map)
        self.prefix = prefix
        if isinstance(target_module, str):
            target_module = [target_module]
        self.target_module = target_module

    def _save_to(self, name, optimizer: Optimizer, name_template=None):
        sd_base = {}
        for item in self.target_module:
            block = optimizer if item == '' else eval(f"optimizer.{item}")
            sd_item = block.state_dict()
            sd_base.update(sd_item)

        if len(sd_base) > 0:
            sd_base = self.clean_prefix(sd_base)
            if name_template is not None:
                name_base = name_template.format(name)
            else:
                name_base = name
            self.save(sd_base, name_base, prefix=self.prefix)


class NekoOptimizerLoader(NekoLoader):
    def __init__(self, format: CkptFormat = None, source: LocalCkptSource = None, path: str = None,
                 target_module='', state_prefix=None, base_model_alpha=0.0, alpha=1.0, key_map=None):
        key_map = key_map or ('name -> name', 'optimizer -> optimizer')
        super().__init__(format=format, source=source, key_map=key_map)
        self.path = path

        self.target_module = target_module
        self.base_model_alpha = base_model_alpha
        self.alpha = alpha
        self.state_prefix = state_prefix

    def _load_to(self, name, optimizer: Optimizer):
        optimizer = optimizer if self.target_module == '' else eval(f"optimizer.{self.target_module}")
        states = optimizer.state_dict()

        part_state = self.load(self.path, map_location='cpu')
        if 'base' in part_state:
            part_state = part_state['base']

        if self.state_prefix:
            state_prefix_len = len(self.state_prefix)
            part_state = {k[state_prefix_len:]: v for k, v in part_state.items() if k.startswith(self.state_prefix)}

        sd_data = {k: self.base_model_alpha * states[k] + self.alpha * v.to(states[k].device) for k, v in part_state.items()}
        optimizer.load_state_dict(sd_data, strict=True)
