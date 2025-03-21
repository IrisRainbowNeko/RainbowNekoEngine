from typing import Dict, Any, Callable

import torch
from rainbowneko.parser.model import CfgPluginParser

from .base import BasicAction


class PrepareAction(BasicAction):
    def __init__(self, device='cuda', dtype=torch.float32, key_map_in=None, key_map_out=None):
        super().__init__(key_map_in=key_map_in, key_map_out=key_map_out)
        self.device = torch.device(device)
        self.dtype = dtype

    def forward(self, in_preview=False, **states):
        if not in_preview:
            return {'device': self.device, 'dtype': self.dtype}


class BuildModelAction(BasicAction):
    def __init__(self, model_builder: Callable, key_map_in=None, key_map_out=None):
        super().__init__(key_map_in=key_map_in, key_map_out=key_map_out)
        self.model_builder = model_builder

    def forward(self, device, in_preview=False, **states):
        if not in_preview:
            model = self.model_builder().to(device)
            model.eval()
            return {'model': model}


class BuildPluginAction(BasicAction):
    def __init__(self, parser: CfgPluginParser, key_map_in=None, key_map_out=None):
        super().__init__(key_map_in=key_map_in, key_map_out=key_map_out)
        self.parser = parser

    def forward(self, model, device, in_preview=False, **states):
        if not in_preview:
            train_params, all_plugin_group = self.parser.get_params_group(model)
            model.eval()
            return {'all_plugin_group': all_plugin_group}


class ForwardAction(BasicAction):
    def to_dev(self, x, device, dtype):
        if isinstance(x, torch.Tensor):
            if torch.is_floating_point(x):
                return x.to(device, dtype=dtype)
            else:
                return x.to(device)
        else:
            return x

    def forward(self, input: Dict[str, Any], model, device, dtype, **states):
        with torch.no_grad(), torch.amp.autocast(device.type, dtype=dtype):
            input = {k:self.to_dev(v, device, dtype) for k,v in input.items()}
            output: Dict[str, Any] = model(**input)
        return {'output': output}


class VisPredAction(BasicAction):
    def forward(self, output: Dict[str, Any], **states):
        for k, v in output.items():
            if isinstance(v, torch.Tensor):
                print(f'{k} = {v.cpu()}')
            else:
                print(f'{k} = {v}')
