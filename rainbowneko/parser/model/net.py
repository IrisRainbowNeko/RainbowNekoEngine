from typing import Dict, List
from rainbowneko.ckpt_manager.locator import get_match_layers
from torch import nn

class CfgModelParser:
    def __init__(self, cfg_model: List[Dict], lr=1e-5, weight_decay=0):
        self.cfg_model = cfg_model
        self.lr = lr
        self.weight_decay = weight_decay

    def get_params(self, layers, named_modules, named_parameters):
        params = []
        train_layers = []
        for layer_name in get_match_layers(layers, named_modules):
            layer = named_modules[layer_name]
            layer.requires_grad_(True)
            layer.train()
            train_layers.append(layer)
            params.extend(layer.parameters())

        for param_name in get_match_layers(layers, named_parameters):
            if param_name in named_parameters:
                param: nn.Parameter = named_parameters[param_name]
                param.requires_grad_(True)
                params.append(param)
        return train_layers, list(dict.fromkeys(params))  # remove duplicates and keep order

    def get_params_group(self, model: nn.Module):
        named_modules = {k: v for k, v in model.named_modules()}
        named_parameters = {k: v for k, v in model.named_parameters()}

        params_group = []
        train_layers = []

        if self.cfg_model is not None:
            for item in self.cfg_model:
                layers, params = self.get_params(item.layers, named_modules, named_parameters)
                params_group.append({"params": params, "lr": getattr(item, "lr", self.lr),
                                    "weight_decay": getattr(item, "weight_decay", self.weight_decay)})
                train_layers.extend(layers)

        return params_group, train_layers

class CfgWDModelParser(CfgModelParser):
    def get_params_group(self, model):
        named_modules = {k: v for k, v in model.named_modules()}
        named_parameters = {k: v for k, v in model.named_parameters()}

        params_group = []
        train_layers = []

        wd_params = set()
        for m in model.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                wd_params.add(m.weight)

        if self.cfg_model is not None:
            for item in self.cfg_model:
                layers, params = self.get_params(item.layers, named_modules, named_parameters)
                train_layers.extend(layers)

                params_nowd = []
                params_wd = []
                for p in params:
                    if p in wd_params:
                        params_wd.append(p)
                    else:
                        params_nowd.append(p)
                if len(params_nowd)>0:
                    params_group.append({"params": params_nowd, "lr": getattr(item, "lr", self.lr),
                                        "weight_decay": 0})
                if len(params_wd) > 0:
                    params_group.append({"params": params_wd, "lr": getattr(item, "lr", self.lr),
                                         "weight_decay": getattr(item, "weight_decay", self.weight_decay)})

        return params_group, train_layers

class CustomModelParser:
    def __init__(self, params_loader, **kwargs):
        self.params_loader = params_loader
        self.kwargs = kwargs

    def get_params_group(self, model):
        train_params, train_layers = self.params_loader(model, **self.kwargs)
        return train_params, train_layers