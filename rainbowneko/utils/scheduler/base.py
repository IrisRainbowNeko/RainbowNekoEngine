# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""LR Scheduler copy from diffusers."""

import math
from enum import Enum
from typing import Optional, Union, Callable

SchedulerType = Callable[[int], float]


class SchedulerName(Enum):
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
    PIECEWISE_CONSTANT = "piecewise_constant"


def constant_schedule() -> SchedulerType:
    """
    Create a schedule with a constant learning rate, using the learning rate set in optimizer.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    return lambda _: 1


def constant_schedule_with_warmup(num_warmup_steps: int, warmup_start=0.0, warmup_final=1.0) -> SchedulerType:
    """
    Create a schedule with a constant learning rate preceded by a warmup period during which the learning rate
    increases linearly between 0 and the initial lr set in the optimizer.

    Args:
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            v = float(current_step) / float(max(1.0, num_warmup_steps))
            return v*(warmup_final - warmup_start) + warmup_start
        else:
            v = warmup_final
        return v

    return lr_lambda


def piecewise_constant_schedule(step_rules: str) -> SchedulerType:
    """
    Create a schedule with a constant learning rate, using the learning rate set in optimizer.

    Args:
        step_rules (`string`):
            The rules for the learning rate. ex: rule_steps="1:10,0.1:20,0.01:30,0.005" it means that the learning rate
            if multiple 1 for the first 10 steps, mutiple 0.1 for the next 20 steps, multiple 0.01 for the next 30
            steps and multiple 0.005 for the other steps.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    rules_dict = {}
    rule_list = step_rules.split(",")
    for rule_str in rule_list[:-1]:
        value_str, steps_str = rule_str.split(":")
        steps = int(steps_str)
        value = float(value_str)
        rules_dict[steps] = value
    last_lr_multiple = float(rule_list[-1])

    def create_rules_function(rules_dict, last_lr_multiple):
        def rule_func(steps: int) -> float:
            sorted_steps = sorted(rules_dict.keys())
            for i, sorted_step in enumerate(sorted_steps):
                if steps < sorted_step:
                    return rules_dict[sorted_steps[i]]
            return last_lr_multiple

        return rule_func

    rules_func = create_rules_function(rules_dict, last_lr_multiple)

    return rules_func


def linear_schedule_with_warmup(num_warmup_steps, num_training_steps, start_scale=1.0, final_scale=0.0) -> SchedulerType:
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        scale = float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, scale * (start_scale - final_scale) + final_scale)

    return lr_lambda


def cosine_schedule_with_warmup(num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, start_scale=1.0, final_scale=0.0):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of periods of the cosine function in a schedule (the default is to just decrease from the max
            value to 0 following a half-cosine).

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        scale = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        return max(0.0, scale * (start_scale - final_scale) + final_scale)

    return lr_lambda


def cosine_with_hard_restarts_schedule_with_warmup(num_warmup_steps: int, num_training_steps: int, num_cycles: int = 1, start_scale=1.0, final_scale=0.0):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, with several hard restarts, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer.

    Args:
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`int`, *optional*, defaults to 1):
            The number of hard restarts to use.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        if progress >= 1.0:
            return final_scale
        scale = 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0)))
        return max(0.0, scale * (start_scale - final_scale) + final_scale)

    return lr_lambda


def polynomial_decay_schedule_with_warmup(num_warmup_steps, num_training_steps, v_start, v_end=1e-7, power=1.0):
    """
    Create a schedule with a learning rate that decreases as a polynomial decay from the initial lr set in the
    optimizer to end lr defined by *lr_end*, after a warmup period during which it increases linearly from 0 to the
    initial lr set in the optimizer.

    Args:
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        lr_end (`float`, *optional*, defaults to 1e-7):
            The end LR.
        power (`float`, *optional*, defaults to 1.0):
            Power factor.

    Note: *power* defaults to 1.0 as in the fairseq implementation, which in turn is based on the original BERT
    implementation at
    https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/optimization.py#L37

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.

    """

    if not (v_start > v_end):
        raise ValueError(f"v_end ({v_end}) must be be smaller than v_start ({v_start})")

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step > num_training_steps:
            return v_end / v_start  # as LambdaLR multiplies by lr_init
        else:
            lr_range = v_start - v_end
            decay_steps = num_training_steps - num_warmup_steps
            pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
            decay = lr_range * pct_remaining ** power + v_end
            return decay / v_start  # as LambdaLR multiplies by lr_init

    return lr_lambda


def fractional_warmup_schedule(decay_max=0.997, decay_factor=(1, 10)):
    def lr_lambda(current_step: int):
        value = (decay_factor[0] + current_step) / (decay_factor[1] + current_step)
        return min(decay_max, value)

    return lr_lambda


def polynomial_EMA_schedule(decay_max=0.997, inv_gamma=1.0, power=3 / 4):
    def lr_lambda(current_step: int):
        value = 1 - (1 + current_step / inv_gamma) ** -power
        return min(decay_max, value)

    return lr_lambda


TYPE_TO_SCHEDULER_FUNCTION = {
    SchedulerName.LINEAR: linear_schedule_with_warmup,
    SchedulerName.COSINE: cosine_schedule_with_warmup,
    SchedulerName.COSINE_WITH_RESTARTS: cosine_with_hard_restarts_schedule_with_warmup,
    SchedulerName.POLYNOMIAL: polynomial_decay_schedule_with_warmup,
    SchedulerName.CONSTANT: constant_schedule,
    SchedulerName.CONSTANT_WITH_WARMUP: constant_schedule_with_warmup,
    SchedulerName.PIECEWISE_CONSTANT: piecewise_constant_schedule,
}


def get_scheduler_with_name(
        name: Union[str, SchedulerName],
        warmup_steps: Optional[int] = None,
        training_steps: Optional[int] = None,
        **kwargs
):
    """
    Unified API to get any scheduler from its name.

    Args:
        name (`str` or `SchedulerName`):
            The name of the scheduler to use.
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
    return scheduler