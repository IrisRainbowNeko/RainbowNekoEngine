from copy import deepcopy
from typing import Dict, Iterable, Union, Tuple

import torch
from rainbowneko.utils.lr_scheduler import SchedulerType
from torch import nn

ModelParamsType = Union[Dict[str, nn.Parameter], Iterable[Tuple[str, nn.Parameter]], nn.Module]


class ModelEMA:
    def __init__(self, model: nn.Module, scheduler: SchedulerType, start_step=0, interval=1):
        with torch.no_grad():
            self.model = deepcopy(model)
            self.train_params = {name: p for name, p in model.named_parameters()}

        self.scheduler = scheduler
        self.optimization_step = start_step
        self.interval = interval

        self.foreach = hasattr(torch, '_foreach_copy_')

    @torch.no_grad()
    def step(self, model: nn.Module, step: int = 0):
        self.optimization_step += 1

        if step % self.interval == 0:
            # Compute the decay factor for the exponential moving average.
            one_minus_decay = 1 - self.scheduler(step)

            if self.foreach:
                params_grad = [
                    param.to(s_param.device) for s_param, param in zip(self.model.parameters(), model.parameters())
                    if param.requires_grad
                ]
                s_params_grad = [
                    s_param for s_param, param in zip(self.model.parameters(), model.parameters()) if param.requires_grad
                ]

                torch._foreach_sub_(s_params_grad, torch._foreach_sub(s_params_grad, params_grad), alpha=one_minus_decay)

            else:
                for s_param, param in zip(self.model.parameters(), model.parameters()):
                    s_param.sub_(one_minus_decay * (s_param - param.to(s_param.device)))

        # torch.cuda.empty_cache()

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def copy_to(self, model: nn.Module) -> None:
        if self.foreach:
            parameters = list(model.parameters())
            torch._foreach_copy_(
                [param.data for param in parameters],
                [s_param.to(param.device).data for s_param, param in zip(self.model.parameters(), parameters)],
            )
        else:
            for s_param, param in zip(self.model.parameters(), model.parameters()):
                param.data.copy_(s_param.to(param.device).data)

    def to(self, device=None, dtype=None):
        with torch.no_grad():
            self.model.to(device, dtype)
            self.train_params = {name: p for name, p in self.model.named_parameters()}
        return self

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.model.state_dict()

    def load_state_dict(self, state: Dict[str, torch.Tensor], prefix=None):
        for k, v in state:
            if prefix is None:
                if k in self.train_params:
                    self.train_params[k].data = v
            else:
                if k in self.train_params and k.startswith(prefix):
                    self.train_params[k].data = v


class ParameterEMA:
    def __init__(self, params: ModelParamsType, scheduler: SchedulerType, start_step=0, interval=1):
        with torch.no_grad():
            self.params = self.get_params(params)

        self.scheduler = scheduler
        self.optimization_step = start_step
        self.interval = interval

        self.foreach = hasattr(torch, '_foreach_copy_')

    def get_params(self, model_or_params: ModelParamsType, grad_only=True):
        if isinstance(model_or_params, nn.Module):
            return {name: p for name, p in model_or_params.named_parameters() if not grad_only or p.requires_grad}
        elif isinstance(model_or_params, dict):
            return {name: p for name, p in model_or_params.items() if not grad_only or p.requires_grad}
        else:
            return {name: p for name, p in model_or_params if not grad_only or p.requires_grad}

    @torch.no_grad()
    def step(self, params: ModelParamsType, step: int = 0):
        self.optimization_step += 1
        params = self.get_params(params)

        if step % self.interval == 0:
            # Compute the decay factor for the exponential moving average.
            one_minus_decay = 1 - self.scheduler(step)

            if self.foreach:
                params_grad = [param.to(self.params[name].device) for name, param in params.items()]
                s_params_grad = [self.params[name] for name in params.keys()]

                torch._foreach_sub_(s_params_grad, torch._foreach_sub(s_params_grad, params_grad), alpha=one_minus_decay)

            else:
                for name, param in params.items():
                    s_param = self.params[name]
                    s_param.sub_(one_minus_decay * (s_param - param.to(s_param.device)))

    def copy_to(self, params: ModelParamsType) -> None:
        params = self.get_params(params)
        if self.foreach:
            torch._foreach_copy_(
                [param.data for param in params.values()],
                [self.params[name].to(param.device).data for name, param in params.items()],
            )
        else:
            for name, param in params.items():
                s_param = self.params[name]
                param.data.copy_(s_param.to(param.device).data)

    def to(self, device=None, dtype=None):
        with torch.no_grad():
            self.params = {name: p.to(device, dtype) for name, p in self.params.items()}
        return self

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {name: p.data for name, p in self.params.items()}

    def load_state_dict(self, state: Dict[str, torch.Tensor], prefix=None):
        for k, v in state:
            if prefix is None:
                if k in self.params:
                    self.params[k].data = v
            else:
                if k in self.params and k.startswith(prefix):
                    self.params[k].data = v
