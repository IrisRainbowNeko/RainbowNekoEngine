import functools
from typing import Dict, List, Tuple

from rainbowneko.models.plugin import SinglePluginBlock, MultiPluginBlock, PluginBlock, PluginGroup, PatchPluginBlock
from rainbowneko.utils import net_path_join, split_module_name
from torch import nn

from rainbowneko.ckpt_manager.locator import get_match_layers


class CfgPluginParser:
    def __init__(self, cfg_plugin: Dict, lr=1e-5, weight_decay=0):
        self.cfg_plugin = cfg_plugin
        self.lr = lr
        self.weight_decay = weight_decay

    def get_params_multi(self, builder, named_modules, plugin_name, model, train_plugin):
        blocks = {}
        params_list = []

        from_layers = [
            {**item, "layer": named_modules[item["layer"]]}
            for item in get_match_layers(builder.keywords.pop("from_layers"), named_modules, return_metas=True)
        ]
        to_layers = [
            {**item, "layer": named_modules[item["layer"]]}
            for item in get_match_layers(builder.keywords.pop("to_layers"), named_modules, return_metas=True)
        ]

        layer = builder(name=plugin_name, host_model=model, from_layers=from_layers, to_layers=to_layers)
        if train_plugin:
            layer.train()
            params = layer.get_trainable_parameters()
            for p in params:
                p.requires_grad_(True)
                params_list.append(p)
        else:
            layer.requires_grad_(False)
            layer.eval()
        blocks[""] = layer
        return blocks, params_list

    def get_params_single(self, builder, named_modules, plugin_name, model, train_plugin):
        blocks = {}
        params_list = []

        layers_name = builder.keywords.pop("layers")
        for layer_name in get_match_layers(layers_name, named_modules):
            blocks = builder(name=plugin_name, host_model=model, host=named_modules[layer_name])
            if not isinstance(blocks, dict):
                blocks = {"": blocks}

            for k, v in blocks.items():
                blocks[net_path_join(layer_name, k)] = v
                if train_plugin:
                    v.train()
                    params = v.get_trainable_parameters()
                    for p in params:
                        p.requires_grad_(True)
                        params_list.append(p)
                else:
                    v.requires_grad_(False)
                    v.eval()

        return blocks, params_list

    def get_params_plugin(self, builder, named_modules, plugin_name, model, train_plugin):
        blocks = {}
        params_list = []

        from_layer = get_match_layers(builder.keywords.pop("from_layer"), named_modules, return_metas=True)
        to_layer = get_match_layers(builder.keywords.pop("to_layer"), named_modules, return_metas=True)

        for from_layer_meta, to_layer_meta in zip(from_layer, to_layer):
            from_layer_name = from_layer_meta["layer"]
            from_layer_meta["layer"] = named_modules[from_layer_name]
            to_layer_meta["layer"] = named_modules[to_layer_meta["layer"]]
            layer = builder(name=plugin_name, host_model=model, from_layer=from_layer_meta, to_layer=to_layer_meta)
            if train_plugin:
                layer.train()
                params = layer.get_trainable_parameters()
                for p in params:
                    p.requires_grad_(True)
                    params_list.append(p)
            else:
                layer.requires_grad_(False)
                layer.eval()
            blocks[from_layer_name] = layer

        return blocks, params_list

    def get_params_patch(self, builder, named_modules, plugin_name, model, train_plugin):
        blocks = {}
        params_list = []

        layers_name = builder.keywords.pop("layers")
        for layer_name in get_match_layers(layers_name, named_modules):
            parent_name, host_name = split_module_name(layer_name)
            layers = builder(
                name=plugin_name,
                host_model=model,
                host=named_modules[layer_name],
                parent_block=named_modules[parent_name],
                host_name=host_name,
            )
            if not isinstance(layers, dict):
                layers = {"": layers}

            for k, v in layers.items():
                blocks[net_path_join(layer_name, k)] = v
                if train_plugin:
                    v.train()
                    params = v.get_trainable_parameters()
                    for p in params:
                        p.requires_grad_(True)
                        params_list.append(p)
                else:
                    v.requires_grad_(False)
                    v.eval()

        return blocks, params_list

    def get_params_group(self, model) -> Tuple[List, Dict[str, PluginGroup]]:
        train_params = []
        all_plugin_group = {}

        if self.cfg_plugin is None:
            return train_params, all_plugin_group

        named_modules = {k: v for k, v in model.named_modules()}

        for plugin_name, builder in self.cfg_plugin.items():
            builder: functools.partial

            lr = builder.keywords.pop("lr", self.lr)
            train_plugin = builder.keywords.pop("train", True)
            reload = builder.keywords.pop("_reload_", False)
            plugin_class = getattr(builder.func, "__self__", builder.func)  # support static or class method

            if reload:
                named_modules_ = {k: v for k, v in model.named_modules()}
            else:
                named_modules_ = named_modules

            if issubclass(plugin_class, MultiPluginBlock):
                blocks, params_list = self.get_params_multi(builder, named_modules_, plugin_name, model, train_plugin)
            elif issubclass(plugin_class, SinglePluginBlock):
                blocks, params_list = self.get_params_single(builder, named_modules_, plugin_name, model, train_plugin)
            elif issubclass(plugin_class, PluginBlock):
                blocks, params_list = self.get_params_plugin(builder, named_modules_, plugin_name, model, train_plugin)
            elif issubclass(plugin_class, PatchPluginBlock):
                blocks, params_list = self.get_params_patch(builder, named_modules_, plugin_name, model, train_plugin)
            else:
                raise NotImplementedError(f"Unknown plugin: {plugin_class}")

            if train_plugin:
                train_params.append({"params": params_list, "lr": lr})
            all_plugin_group[plugin_name] = PluginGroup(blocks)
        return train_params, all_plugin_group

class CfgWDPluginParser(CfgPluginParser):
    def get_params_group(self, model) -> Tuple[List, Dict[str, PluginGroup]]:
        train_params = []
        all_plugin_group = {}

        if self.cfg_plugin is None:
            return train_params, all_plugin_group

        named_modules = {k: v for k, v in model.named_modules()}

        for plugin_name, builder in self.cfg_plugin.items():
            builder: functools.partial

            lr = builder.keywords.pop("lr", self.lr)
            weight_decay = builder.keywords.pop("weight_decay", self.weight_decay)
            train_plugin = builder.keywords.pop("train", True)
            reload = builder.keywords.pop("_reload_", False)
            plugin_class = getattr(builder.func, "__self__", builder.func)  # support static or class method

            if reload:
                named_modules_ = {k: v for k, v in model.named_modules()}
            else:
                named_modules_ = named_modules

            if issubclass(plugin_class, MultiPluginBlock):
                blocks, params_list = self.get_params_multi(builder, named_modules_, plugin_name, model, train_plugin)
            elif issubclass(plugin_class, SinglePluginBlock):
                blocks, params_list = self.get_params_single(builder, named_modules_, plugin_name, model, train_plugin)
            elif issubclass(plugin_class, PluginBlock):
                blocks, params_list = self.get_params_plugin(builder, named_modules_, plugin_name, model, train_plugin)
            elif issubclass(plugin_class, PatchPluginBlock):
                blocks, params_list = self.get_params_patch(builder, named_modules_, plugin_name, model, train_plugin)
            else:
                raise NotImplementedError(f"Unknown plugin: {plugin_class}")

            if train_plugin:
                params_nowd, params_wd = self.get_wd(blocks, params_list)
                if len(params_nowd)>0:
                    train_params.append({"params": params_nowd, "lr": lr, "weight_decay": 0})
                if len(params_wd) > 0:
                    train_params.append({"params": params_wd, "lr": lr, "weight_decay": weight_decay})
            all_plugin_group[plugin_name] = PluginGroup(blocks)
        return train_params, all_plugin_group

    def get_wd(self, blocks: Dict[str, nn.Module], params_list: List[nn.Parameter]):
        params_nowd = []
        params_wd = []

        wd_params = set()
        for block in blocks.values():
            for m in block.modules():
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    wd_params.add(m.weight)

        for p in params_list:
            if p in wd_params:
                params_wd.append(p)
            else:
                params_nowd.append(p)
        return params_nowd, params_wd