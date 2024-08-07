"""
cfg_net_tools.py
====================
    :Name:        creat model and plugin from config
    :Author:      Dong Ziyi
    :Affiliation: HCP Lab, SYSU
    :Created:     10/03/2023
    :Licence:     Apache-2.0
"""

import re
from typing import Dict, List, Tuple, Union, Any

import torch
from torch import nn
import functools

from rainbowneko.models.plugin import SinglePluginBlock, MultiPluginBlock, PluginBlock, PluginGroup, PatchPluginBlock
from rainbowneko.ckpt_manager import CkptManagerPKL, CkptManagerSafe
from ..utils import net_path_join
from ..utils import split_module_name


def get_class_match_layer(class_name, block: nn.Module):
    if type(block).__name__ == class_name:
        return [""]
    else:
        return ["." + name for name, layer in block.named_modules() if type(layer).__name__ == class_name]


def get_match_layers(layers, all_layers, return_metas=False) -> Union[List[str], List[Dict[str, Any]]]:
    res = []
    for name in layers:
        metas = name.split(":")

        use_re = False
        pre_hook = False
        cls_filter = None
        for meta in metas[:-1]:
            if meta == "re":
                use_re = True
            elif meta == "pre_hook":
                pre_hook = True
            elif meta.startswith("cls("):
                cls_filter = meta[4:-1]

        name = metas[-1]
        if use_re:
            pattern = re.compile(name)
            match_layers = filter(lambda x: pattern.match(x) != None, all_layers.keys())
        else:
            match_layers = [name]

        if cls_filter is not None:
            match_layers_new = []
            for layer in match_layers:
                match_layers_new.extend([layer + x for x in get_class_match_layer(name[1], all_layers[layer])])
            match_layers = match_layers_new

        for layer in match_layers:
            if return_metas:
                res.append({"layer": layer, "pre_hook": pre_hook})
            else:
                res.append(layer)

    # Remove duplicates and keep the original order
    if return_metas:
        layer_set = set()
        res_unique = []
        for item in res:
            if item["layer"] not in layer_set:
                layer_set.add(item["layer"])
                res_unique.append(item)
        return res_unique
    else:
        return sorted(set(res), key=res.index)


class CfgModelParser:
    def __init__(self, cfg_model, lr=1e-5, weight_decay=0):
        self.cfg_model = cfg_model
        self.lr = lr
        self.weight_decay = weight_decay

    def get_params_group(self, model):
        named_modules = {k: v for k, v in model.named_modules()}

        train_params = []
        train_layers = []

        if self.cfg_model is not None:
            for item in self.cfg_model:
                params_group = []
                for layer_name in get_match_layers(item.layers, named_modules):
                    layer = named_modules[layer_name]
                    layer.requires_grad_(True)
                    layer.train()
                    train_layers.append(layer)
                    params_group.extend(layer.parameters())
                train_params.append({"params": list(dict.fromkeys(params_group)), "lr": getattr(item, "lr", self.lr),
                                    "weight_decay": getattr(item, "weight_decay", self.weight_decay)})

        return train_params, train_layers

class CfgWDModelParser(CfgModelParser):
    def get_params_group(self, model):
        named_modules = {k: v for k, v in model.named_modules()}

        train_params = []
        train_layers = []

        wd_params = set()
        for m in model.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                wd_params.add(m.weight)

        if self.cfg_model is not None:
            for item in self.cfg_model:
                params_group = []
                params_group_wd = []
                for layer_name in get_match_layers(item.layers, named_modules):
                    layer = named_modules[layer_name]
                    layer.requires_grad_(True)
                    layer.train()
                    train_layers.append(layer)
                    for p in layer.parameters():
                        if p in wd_params:
                            params_group_wd.append(p)
                        else:
                            params_group.append(p)
                if len(params_group)>0:
                    train_params.append({"params": list(dict.fromkeys(params_group)), "lr": getattr(item, "lr", self.lr),
                                        "weight_decay": 0})
                if len(params_group_wd) > 0:
                    train_params.append({"params": list(dict.fromkeys(params_group_wd)), "lr": getattr(item, "lr", self.lr),
                                         "weight_decay": getattr(item, "weight_decay", self.weight_decay)})


        return train_params, train_layers


class CustomModelParser:
    def __init__(self, params_loader, **kwargs):
        self.params_loader = params_loader
        self.kwargs = kwargs

    def get_params_group(self, model):
        train_params, train_layers = self.params_loader(model, **self.kwargs)
        return train_params, train_layers

def parse_plugin_cfg(model, cfg_plugin, default_lr=1e-5) -> Tuple[List, Dict[str, PluginGroup]]:
    train_params = []
    all_plugin_group = {}

    if cfg_plugin is None:
        return train_params, all_plugin_group

    named_modules = {k: v for k, v in model.named_modules()}

    for plugin_name, builder in cfg_plugin.items():
        builder: functools.partial
        all_plugin_blocks = {}

        lr = builder.keywords.pop("lr") if "lr" in builder.keywords else default_lr
        train_plugin = builder.keywords.pop("train") if "train" in builder.keywords else True
        plugin_class = getattr(builder.func, "__self__", builder.func)  # support static or class method

        params_group = []
        if issubclass(plugin_class, MultiPluginBlock):
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
                    params_group.append(p)
            else:
                layer.requires_grad_(False)
                layer.eval()
            all_plugin_blocks[""] = layer
        elif issubclass(plugin_class, SinglePluginBlock):
            layers_name = builder.keywords.pop("layers")
            for layer_name in get_match_layers(layers_name, named_modules):
                blocks = builder(name=plugin_name, host_model=model, host=named_modules[layer_name])
                if not isinstance(blocks, dict):
                    blocks = {"": blocks}

                for k, v in blocks.items():
                    all_plugin_blocks[net_path_join(layer_name, k)] = v
                    if train_plugin:
                        v.train()
                        params = v.get_trainable_parameters()
                        for p in params:
                            p.requires_grad_(True)
                            params_group.append(p)
                    else:
                        v.requires_grad_(False)
                        v.eval()
        elif issubclass(plugin_class, PluginBlock):
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
                        params_group.append(p)
                else:
                    layer.requires_grad_(False)
                    layer.eval()
                all_plugin_blocks[from_layer_name] = layer
        elif issubclass(plugin_class, PatchPluginBlock):
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
                    all_plugin_blocks[net_path_join(layer_name, k)] = v
                    if train_plugin:
                        v.train()
                        params = v.get_trainable_parameters()
                        for p in params:
                            p.requires_grad_(True)
                            params_group.append(p)
                    else:
                        v.requires_grad_(False)
                        v.eval()
        else:
            raise NotImplementedError(f"Unknown plugin {plugin_class}")
        if train_plugin:
            train_params.append({"params": params_group, "lr": lr})
        all_plugin_group[plugin_name] = PluginGroup(all_plugin_blocks)
    return train_params, all_plugin_group


@torch.no_grad()
def load_models(model: nn.Module, cfg_merge):
    named_modules = {k: v for k, v in model.named_modules()}
    named_params = {k: v for k, v in model.named_parameters()}
    all_lora_blocks = {}

    ckpt_manager_torch = CkptManagerPKL()
    ckpt_manager_safe = CkptManagerSafe()

    def get_ckpt_manager(path: str):
        return ckpt_manager_safe if path.endswith(".safetensors") else ckpt_manager_torch

    if getattr(cfg_merge, "lora", None) is not None:
        for lora_id, item in enumerate(cfg_merge.lora):
            lora_state = get_ckpt_manager(item.path).load_ckpt(item.path, map_location="cpu")["lora"]
            lora_block_state = {}
            # get all layers in the lora_state
            for name, p in lora_state.items():
                # lora_block. is the old format
                prefix, block_name = name.split(".___." if name.rfind("lora_block.") == -1 else ".lora_block.", 1)
                if prefix not in lora_block_state:
                    lora_block_state[prefix] = {}
                lora_block_state[prefix][block_name] = p
            # get selected layers
            if item.layers != "all":
                match_blocks = get_match_layers(item.layers, named_modules)
                lora_state_new = {}
                for k, v in lora_block_state.items():
                    for mk in match_blocks:
                        if k.startswith(mk):
                            lora_state_new[k] = v
                            break
                lora_block_state = lora_state_new
            # add lora to host and load weights
            for host_name, lora_state in lora_block_state.items():
                lora_layer_cls, rank, rank_groups = get_lora_rank_and_cls(lora_state)
                if "alpha" in lora_state:
                    del lora_state["alpha"]
                if "scale" in lora_state:  # old format
                    del lora_state["scale"]

                lora_block = lora_layer_cls.wrap_layer(
                    lora_id,
                    named_modules[host_name],
                    rank=rank,
                    dropout=getattr(item, "dropout", 0.0),
                    alpha=getattr(item, "alpha", 1.0),
                    bias="layer.lora_up.bias" in lora_state,
                    rank_groups=rank_groups,
                    alpha_auto_scale=getattr(item, "alpha_auto_scale", True),
                )
                all_lora_blocks[f"{host_name}.{lora_block.name}"] = lora_block
                lora_block.load_state_dict(lora_state, strict=False)
                lora_block.set_mask(getattr(item, "mask", None))
                lora_block.to(model.device)

    if getattr(cfg_merge, "part", None) is not None:
        for item in cfg_merge.part:
            part_state = get_ckpt_manager(item.path).load_ckpt(item.path, map_location="cpu")["base"]
            if item.layers == "all":
                for k, v in part_state.items():
                    named_params[k].data = cfg_merge.base_model_alpha * named_params[k].data + item.alpha * v
            else:
                match_blocks = get_match_layers(item.layers, named_modules)
                state_add = {k: v for blk in match_blocks for k, v in part_state.items() if k.startswith(blk)}
                for k, v in state_add.items():
                    named_params[k].data = cfg_merge.base_model_alpha * named_params[k].data + item.alpha * v

    if getattr(cfg_merge, "plugin", None) is not None:
        for name, item in cfg_merge.plugin.items():
            plugin_state = get_ckpt_manager(item.path).load_ckpt(item.path, map_location="cpu")["plugin"]
            if item.layers != "all":
                match_blocks = get_match_layers(item.layers, named_modules)
                plugin_state = {k: v for blk in match_blocks for k, v in plugin_state.items() if k.startswith(blk)}
            plugin_key_set = set([k.split("___", 1)[0] + name for k in plugin_state.keys()])
            plugin_state = {
                k.replace("___", name): v for k, v in plugin_state.items()
            }  # replace placeholder to target plugin name
            model.load_state_dict(plugin_state, strict=False)
            del item.layers
            del item.path
            if hasattr(model, name):  # MultiPluginBlock
                getattr(model, name).set_hyper_params(**item)
            else:
                for plugin_key in plugin_key_set:
                    named_modules[plugin_key].set_hyper_params(**item)

    return LoraGroup(all_lora_blocks)
