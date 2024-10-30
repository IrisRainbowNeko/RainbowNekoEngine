from typing import Iterable, Tuple, Dict

import torch
from torch import nn
from copy import deepcopy


class ModelEMA:
    def __init__(self, model: nn.Module, decay_max=0.997, decay_factor=(1, 5),
                 optimization_step=0, interval=1):
        self.train_params = {name: p.clone().detach() for name, p in model.state_dict().items()}
        self.model = deepcopy(model)
        model.load_state_dict(self.train_params)

        self.decay_max = decay_max
        self.decay_factor = decay_factor
        self.optimization_step = optimization_step
        self.interval = interval

    @torch.no_grad()
    def step(self, model: nn.Module, step: int = 0):
        self.optimization_step += 1

        if step % self.interval == 0:
            # Compute the decay factor for the exponential moving average.
            if self.decay_factor[0] == self.decay_factor[1]:
                one_minus_decay = 1 - self.decay_max
            else:
                value = (self.decay_factor[0] + self.optimization_step) / (self.decay_factor[1] + self.optimization_step)
                one_minus_decay = 1 - min(self.decay_max, value)

            for name, param in model.state_dict().items():
                if name in self.train_params:
                    s_param = self.train_params[name]
                    if torch.is_floating_point(s_param):
                        s_param.sub_(one_minus_decay * (s_param - param.to(s_param.device)))

        # torch.cuda.empty_cache()

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def copy_to(self, parameters: Iterable[Tuple[str, torch.nn.Parameter]]) -> None:
        for name, param in parameters:
            if name in self.train_params:
                param.data.copy_(self.train_params[name])

    def to(self, device=None, dtype=None):
        # .to() on the tensors handles None correctly
        self.train_params = {
            name: (p.to(device=device, dtype=dtype) if p.is_floating_point() else p.to(device=device))
            for name, p in self.train_params
        }
        return self

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.train_params

    def load_state_dict(self, state: Dict[str, torch.Tensor], prefix=None):
        for k, v in state:
            if prefix is None:
                if k in self.train_params:
                    self.train_params[k] = v
            else:
                if k in self.train_params and k.startswith(prefix):
                    self.train_params[k] = v
