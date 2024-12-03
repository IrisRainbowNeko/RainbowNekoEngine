from typing import Dict, Any, Callable

import torch

from .base import BasicAction


class PrepareAction(BasicAction):
    def __init__(self, device='cuda', dtype=torch.float32, key_map_in=None, key_map_out=None):
        super().__init__(key_map_in=key_map_in, key_map_out=key_map_out)
        self.device = device
        self.dtype = dtype

    def forward(self, **states):
        return {'device': self.device, 'dtype': self.dtype}


class BuildModelAction(BasicAction):
    def __init__(self, model_builder: Callable, key_map_in=None, key_map_out=None):
        super().__init__(key_map_in=key_map_in, key_map_out=key_map_out)
        self.model_builder = model_builder

    def forward(self, device, **states):
        model = self.model_builder().to(device)
        model.eval()
        return {'model': model}


class ForwardAction(BasicAction):

    def forward(self, input: Dict[str, Any], model, **states):
        with torch.inference_mode():
            output: Dict[str, Any] = model(**input)
        return {'output': output}


class VisPredAction(BasicAction):
    def forward(self, output: Dict[str, Any], **states):
        for k, v in output.items():
            if isinstance(v, torch.Tensor):
                print(f'{k} = {v.cpu()}')
            else:
                print(f'{k} = {v}')
