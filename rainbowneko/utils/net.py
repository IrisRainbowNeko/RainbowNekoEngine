from functools import partial
from typing import Optional, Union

import torch
from .lr_scheduler import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION, Optimizer
from torch import nn
from torch.optim import lr_scheduler

import random
import numpy as np
import torch

class RandomContext:
    def __init__(self, cuda=False):
        self.py_state = random.getstate()
        self.np_state = np.random.get_state()
        self.torch_state = torch.get_rng_state()
        if cuda:
            self.cuda_state = torch.cuda.get_rng_state()

        self.py_state_save = None
        self.np_state_save = None
        self.torch_state_save = None
        if cuda:
            self.cuda_state_save = None

        self.cuda = cuda

    def __enter__(self):
        random.setstate(self.py_state)
        np.random.set_state(self.np_state)
        torch.set_rng_state(self.torch_state)
        if self.cuda and torch.cuda.is_available():
            torch.cuda.set_rng_state(self.cuda_state)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Save Python random state
        if self.py_state_save is None:
            self.py_state_save = random.getstate()

        # Save NumPy random state
        if self.np_state_save is None:
            self.np_state_save = np.random.get_state()

        # Save PyTorch random state
        if self.torch_state_save is None:
            self.torch_state_save = torch.get_rng_state()

        # Restore Python random state
        random.setstate(self.py_state_save)

        # Restore NumPy random state
        np.random.set_state(self.np_state_save)

        # Restore PyTorch random state
        torch.set_rng_state(self.torch_state_save)

        # Restore CUDA random state if available
        if self.cuda and torch.cuda.is_available() and self.cuda_state is not None:
            if self.cuda_state_save is None:
                self.cuda_state_save = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.cuda_state_save)


def get_scheduler(cfg, optimizer, num_training_steps):
    if cfg is None:
        return None
    elif isinstance(cfg, partial):
        return cfg(optimizer=optimizer)
    else:
        return get_scheduler_with_name(optimizer=optimizer, num_training_steps=num_training_steps, **cfg)

def get_scheduler_with_name(
    name: Union[str, SchedulerType],
    optimizer: Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    **kwargs
):
    """
    Unified API to get any scheduler from its name.

    Args:
        name (`str` or `SchedulerType`):
            The name of the scheduler to use.
        optimizer (`torch.optim.Optimizer`):
            The optimizer that will be used during training.
        num_warmup_steps (`int`, *optional*):
            The number of warmup steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_training_steps (`int``, *optional*):
            The number of training steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_cycles (`int`, *optional*):
            The number of hard restarts used in `COSINE_WITH_RESTARTS` scheduler.
        power (`float`, *optional*, defaults to 1.0):
            Power factor. See `POLYNOMIAL` scheduler
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    """
    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if isinstance(num_warmup_steps, float): # warmup ratio
        num_warmup_steps = int(num_warmup_steps * num_training_steps)

    # One Cycle for super convergence
    if name == 'one_cycle':
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=[x['lr'] for x in optimizer.state_dict()['param_groups']],
                                            steps_per_epoch=num_training_steps, epochs=1,
                                            pct_start=num_warmup_steps/num_training_steps, **kwargs)
        return scheduler

    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
    if name == SchedulerType.CONSTANT:
        return schedule_func(optimizer, **kwargs)

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, **kwargs)

    if name == SchedulerType.COSINE_WITH_RESTARTS:
        return schedule_func(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, **kwargs
        )

    if name == SchedulerType.POLYNOMIAL:
        return schedule_func(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, **kwargs
        )

    return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, **kwargs)

def remove_all_hooks(model: nn.Module) -> None:
    for name, child in model.named_modules():
        child._forward_hooks.clear()
        child._forward_pre_hooks.clear()
        child._backward_hooks.clear()

def remove_layers(model: nn.Module, layer_class):
    named_modules = {k:v for k, v in model.named_modules()}
    for k, v in named_modules.items():
        if isinstance(v, layer_class):
            parent, name = named_modules[k.rsplit('.', 1)]
            delattr(parent, name)
            del v

def hook_compile(model):
    named_modules = {k:v for k, v in model.named_modules()}

    for name, block in named_modules.items():
        if len(block._forward_hooks)>0:
            for hook in block._forward_hooks.values():  # 从前往后执行
                old_forward = block.forward

                def new_forward(*args, **kwargs):
                    result = old_forward(*args, **kwargs)
                    hook_result = hook(block, args, result)
                    if hook_result is not None:
                        result = hook_result
                    return result

                block.forward = new_forward

        if len(block._forward_pre_hooks)>0:
            for hook in list(block._forward_pre_hooks.values())[::-1]:  # 从前往后执行
                old_forward = block.forward

                def new_forward(*args, **kwargs):
                    result = hook(block, args)
                    if result is not None:
                        if not isinstance(result, tuple):
                            result = (result,)
                    else:
                        result = args
                    return old_forward(*result, **kwargs)

                block.forward = new_forward
    remove_all_hooks(model)

def _convert_cpu(t):
    return t.to('cpu') if t.device.type == 'cuda' else t

def _convert_cuda(t):
    return t.to('cuda') if t.device.type == 'cpu' else t

def to_cpu(model):
    model._apply(_convert_cpu)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def to_cuda(model):
    model._apply(_convert_cuda)

def split_module_name(layer_name):
    name_split = layer_name.rsplit('.', 1)
    if len(name_split) == 1:
        parent_name, host_name = '', name_split[0]
    else:
        parent_name, host_name = name_split
    return parent_name, host_name
