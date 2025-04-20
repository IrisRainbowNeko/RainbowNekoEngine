from functools import partial
from typing import Union, Optional

from torch.optim import Optimizer
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LambdaLR

from .base import SchedulerName, TYPE_TO_SCHEDULER_FUNCTION, constant_schedule, constant_schedule_with_warmup, \
    cosine_schedule_with_warmup, cosine_with_hard_restarts_schedule_with_warmup, polynomial_decay_schedule_with_warmup, \
    piecewise_constant_schedule


def get_lr_scheduler_with_name(
        name: Union[str, SchedulerName],
        optimizer: Optimizer,
        warmup_steps: Optional[int] = None,
        training_steps: Optional[int] = None,
        **kwargs
):
    """
    Unified API to get any scheduler from its name.

    Args:
        name (`str` or `SchedulerName`):
            The name of the scheduler to use.
        optimizer (`torch.optim.Optimizer`):
            The optimizer that will be used during training.
        warmup_steps (`int`, *optional*):
            The number of warmup steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        training_steps (`int``, *optional*):
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
    if training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    # All other schedulers require `num_warmup_steps`
    if warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if isinstance(warmup_steps, float):  # warmup ratio
        warmup_steps = int(warmup_steps * training_steps)

    # One Cycle for super convergence
    if name == 'one_cycle':
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=[x['lr'] for x in optimizer.state_dict()['param_groups']],
                                            steps_per_epoch=training_steps, epochs=1,
                                            pct_start=warmup_steps / training_steps, **kwargs)
        return scheduler

    name = SchedulerName(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
    if name == SchedulerName.CONSTANT:
        scheduler = schedule_func()
    elif name == SchedulerName.CONSTANT_WITH_WARMUP:
        scheduler = schedule_func(num_warmup_steps=warmup_steps)
    elif name == SchedulerName.COSINE_WITH_RESTARTS:
        scheduler = schedule_func(num_warmup_steps=warmup_steps, num_training_steps=training_steps, **kwargs)
    elif name == SchedulerName.POLYNOMIAL:
        scheduler = schedule_func(num_warmup_steps=warmup_steps, num_training_steps=training_steps, **kwargs)
    else:
        scheduler = schedule_func(num_warmup_steps=warmup_steps, num_training_steps=training_steps, **kwargs)
    return LambdaLR(optimizer, scheduler)


def get_lr_scheduler(cfg, optimizer, num_training_steps):
    if cfg is None:
        return None
    elif isinstance(cfg, partial):
        try:
            return cfg(optimizer=optimizer, training_steps=num_training_steps)
        except:
            return cfg(optimizer=optimizer)
    else:
        return get_lr_scheduler_with_name(optimizer=optimizer, training_steps=num_training_steps, **cfg)


def ConstantLR(optimizer: Optimizer, warmup_steps: int = None, last_epoch: int = -1):
    if warmup_steps is None:
        scheduler = constant_schedule()
    else:
        scheduler = constant_schedule_with_warmup(warmup_steps)
    return LambdaLR(optimizer, scheduler, last_epoch=last_epoch)


def MultiStepLR(optimizer: Optimizer, step_rules: str, last_epoch: int = -1):
    scheduler = piecewise_constant_schedule(step_rules)
    return LambdaLR(optimizer, scheduler, last_epoch=last_epoch)


def CosineLR(optimizer: Optimizer, training_steps: int, warmup_steps: int = 0, num_cycles: float = 0.5, min_scale: float = 0.0,
             last_epoch: int = -1):
    scheduler = cosine_schedule_with_warmup(warmup_steps, training_steps, num_cycles, final_scale=min_scale)
    return LambdaLR(optimizer, scheduler, last_epoch=last_epoch)


def CosineRestartLR(optimizer: Optimizer, training_steps: int, warmup_steps: int = 0, num_cycles: int = 1, min_scale: float = 0.0,
                    last_epoch: int = -1):
    scheduler = cosine_with_hard_restarts_schedule_with_warmup(warmup_steps, training_steps, num_cycles, final_scale=min_scale)
    return LambdaLR(optimizer, scheduler, last_epoch=last_epoch)


def PolynomialLR(optimizer, training_steps: int, warmup_steps: int = 0, lr_end=1e-7, power=1.0, last_epoch=-1):
    lr_init = optimizer.defaults["lr"]
    scheduler = polynomial_decay_schedule_with_warmup(warmup_steps, training_steps, lr_init, lr_end, power)
    return LambdaLR(optimizer, scheduler, last_epoch=last_epoch)


def OneCycleLR(optimizer: Optimizer, training_steps: int, warmup_steps: int = None, **kwargs):
    return lr_scheduler.OneCycleLR(optimizer, max_lr=[x['lr'] for x in optimizer.state_dict()['param_groups']],
                                   steps_per_epoch=training_steps, epochs=1, pct_start=warmup_steps / training_steps, **kwargs)
