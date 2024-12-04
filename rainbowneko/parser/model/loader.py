import torch
from rainbowneko.ckpt_manager import auto_manager

from .locator import get_match_layers


class NekoModelLoader:
    def __init__(self, host):
        self.host = host

    @torch.no_grad()
    def load_part(self, cfg, base_model_alpha=0.0, load_ema=False):
        if cfg is None:
            return
        for item in cfg:
            module_to_load = item.get('module_to_load', '')
            model = self.host if module_to_load == '' else eval(f"self.host.{module_to_load}")
            named_modules = {k: v for k, v in model.named_modules()}
            named_params = {k: v for k, v in model.named_parameters()}
            named_params.update({k: v for k, v in model.named_buffers()})

            if isinstance(item.path, str):
                part_state = auto_manager(item.path).load_ckpt(item.path, map_location='cpu')['base_ema' if load_ema else 'base']
            else:
                part_state = item.path.load() # model loader
            layers = item.get('layers', 'all')
            if layers == 'all':
                for k, v in part_state.items():
                    named_params[k].data = base_model_alpha*named_params[k].data+item.alpha*v
            else:
                match_blocks = get_match_layers(layers, named_modules)
                state_add = {k:v for blk in match_blocks for k, v in part_state.items() if k.startswith(blk)}
                for k, v in state_add.items():
                    named_params[k].data = base_model_alpha*named_params[k].data+item.alpha*v

    @torch.no_grad()
    def load_plugin(self, cfg, load_ema=False):
        if cfg is None:
            return

        for name, item in cfg.items():
            module_to_load = item.get('module_to_load', '')
            model = self.host if module_to_load == '' else eval(f"self.host.{module_to_load}")
            named_modules = {k: v for k, v in model.named_modules()}

            if isinstance(item.path, str):
                plugin_state = auto_manager(item.path).load_ckpt(item.path, map_location='cpu')['plugin_ema' if load_ema else 'plugin']
            else:
                plugin_state = item.path.load(named_modules=named_modules) # model loader

            # filter layers to load
            layers = item.get('layers', 'all')
            if layers != 'all':
                match_blocks = get_match_layers(layers, named_modules)
                plugin_state = {k:v for blk in match_blocks for k, v in plugin_state.items() if k.startswith(blk)}

            if 'layers' in item:
                del item.layers
            del item.path

            # state to plugin_block_state
            plugin_block_state = {}
            for pname, p in plugin_state.items():
                prefix, block_name = pname.split('.___.', 1)
                plugin_block_state.setdefault(f'{prefix}.{name}', {})[block_name] = p

            # set plugin hyper params or build plugin
            plugin_type = item.get('type', 'auto') # TODO: auto build plugin
            if hasattr(self.host, name):  # MultiPluginBlock
                getattr(self.host, name).set_hyper_params(**item)
            else:
                for plugin_key in plugin_block_state.keys():
                    named_modules[plugin_key].set_hyper_params(**item)

            # Load state to plugin
            plugin_state = {k.replace('___', name): v for k, v in plugin_state.items()}  # replace placeholder to target plugin name
            load_info = self.host.load_state_dict(plugin_state, strict=False)
            if len(load_info.unexpected_keys)>0:
                print(name, 'unexpected_keys', load_info.unexpected_keys)

    def load_all(self, cfg_merge, load_ema=False):
        self.load_part(cfg_merge.get('part', []), base_model_alpha=cfg_merge.get('base_model_alpha', 0.0), load_ema=load_ema)
        self.load_plugin(cfg_merge.get('plugin', {}), load_ema=load_ema)
