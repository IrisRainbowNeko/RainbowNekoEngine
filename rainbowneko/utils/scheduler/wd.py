import types
from functools import partial
from typing import Union, Optional

from torch.optim import Optimizer

from .base import SchedulerName, TYPE_TO_SCHEDULER_FUNCTION, constant_schedule, constant_schedule_with_warmup, \
    cosine_schedule_with_warmup, cosine_with_hard_restarts_schedule_with_warmup, polynomial_decay_schedule_with_warmup, \
    piecewise_constant_schedule


class WDScheduler:

    def __init__(self, optimizer, last_epoch=-1):
        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f'{type(optimizer).__name__} is not an Optimizer')
        self.optimizer = optimizer

        # Initialize epoch and base learning rates
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_wd', group['weight_decay'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_wd' not in group:
                    raise KeyError("param 'initial_wd' is not specified "
                                   f"in param_groups[{i}] when resuming an optimizer")
        self.base_wds = [group['initial_wd'] for group in optimizer.param_groups]
        self.last_epoch = last_epoch

        self._initial_step()

    def _initial_step(self):
        """Initialize step counts and performs a step"""
        self.step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_wd(self):
        """ Return last computed learning rate by current scheduler.
        """
        return self._last_wd

    def get_wd(self):
        # Compute learning rate using chainable form of the scheduler
        raise NotImplementedError

    def print_wd(self, is_verbose, group, wd, epoch=None):
        """Display the current weight_decay.
        """
        if is_verbose:
            if epoch is None:
                print(f'Adjusting weight_decay of group {group} to {wd:.4e}.')
            else:
                epoch_str = ("%.2f" if isinstance(epoch, float) else
                             "%.5d") % epoch
                print(f'Epoch {epoch_str}: adjusting weight_decay of group {group} to {wd:.4e}.')

    def step(self, epoch=None):
        if epoch is None:
            self.last_epoch += 1
            values = self.get_wd()
        else:
            self.last_epoch = epoch
            if hasattr(self, "_get_closed_form_wd"):
                values = self._get_closed_form_wd()
            else:
                values = self.get_wd()

        for i, data in enumerate(zip(self.optimizer.param_groups, values)):
            param_group, wd = data
            param_group['weight_decay'] = wd

        self._last_wd = [group['weight_decay'] for group in self.optimizer.param_groups]


class LambdaWD(WDScheduler):
    """Sets the learning rate of each parameter group to the initial lr
    times a given function. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        wd_lambda (function or list): A function which computes a multiplicative
            factor given an integer parameter epoch, or a list of such
            functions, one for each group in optimizer.param_groups.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

            .. deprecated:: 2.2
                ``verbose`` is deprecated. Please use ``get_last_lr()`` to access the
                learning rate.

    Example:
        >>> # xdoctest: +SKIP
        >>> # Assuming optimizer has two groups.
        >>> lambda1 = lambda epoch: epoch // 30
        >>> lambda2 = lambda epoch: 0.95 ** epoch
        >>> scheduler = LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, wd_lambda, last_epoch=-1):
        self.optimizer = optimizer

        if not isinstance(wd_lambda, list) and not isinstance(wd_lambda, tuple):
            self.wd_lambdas = [wd_lambda] * len(optimizer.param_groups)
        else:
            if len(wd_lambda) != len(optimizer.param_groups):
                raise ValueError(f"Expected {len(optimizer.param_groups)} wd_lambdas, but got {len(wd_lambda)}")
            self.wd_lambdas = list(wd_lambda)
        super().__init__(optimizer, last_epoch)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        The learning rate lambda functions will only be saved if they are callable objects
        and not if they are functions or lambdas.

        When saving or loading the scheduler, please make sure to also save or load the state of the optimizer.
        """

        state_dict = {key: value for key, value in self.__dict__.items() if key not in ('optimizer', 'wd_lambdas')}
        state_dict['wd_lambdas'] = [None] * len(self.wd_lambdas)

        for idx, fn in enumerate(self.wd_lambdas):
            if not isinstance(fn, types.FunctionType):
                state_dict['wd_lambdas'][idx] = fn.__dict__.copy()

        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        When saving or loading the scheduler, please make sure to also save or load the state of the optimizer.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """

        wd_lambdas = state_dict.pop('wd_lambdas')
        self.__dict__.update(state_dict)
        # Restore state_dict keys in order to prevent side effects
        # https://github.com/pytorch/pytorch/issues/32756
        state_dict['wd_lambdas'] = wd_lambdas

        for idx, fn in enumerate(wd_lambdas):
            if fn is not None:
                self.wd_lambdas[idx].__dict__.update(fn)

    def get_wd(self):
        return [base_wd * lmbda(self.last_epoch) for lmbda, base_wd in zip(self.wd_lambdas, self.base_wds)]


class ConstantWD(WDScheduler):
    def __init__(self, optimizer: Optimizer, last_epoch: int = -1, warmup_steps: int = None):
        if warmup_steps is None:
            self.scheduler = constant_schedule()
        else:
            self.scheduler = constant_schedule_with_warmup(warmup_steps)
        super().__init__(optimizer, last_epoch)

    def get_wd(self):
        scale = self.scheduler(self.last_epoch)
        return [base_wd * scale for base_wd in self.base_wds]


class MultiStepWD(WDScheduler):
    def __init__(self, optimizer: Optimizer, step_rules: str, last_epoch: int = -1):
        self.scheduler = piecewise_constant_schedule(step_rules)
        super().__init__(optimizer, last_epoch)

    def get_wd(self):
        scale = self.scheduler(self.last_epoch)
        return [base_wd * scale for base_wd in self.base_wds]


class CosineWD(WDScheduler):
    def __init__(self, optimizer: Optimizer, training_steps: int, warmup_steps: int = 0, num_cycles: float = 0.5,
                 min_scale: float = 0.0, last_epoch: int = -1):
        self.scheduler = cosine_schedule_with_warmup(warmup_steps, training_steps, num_cycles, final_scale=min_scale)
        super().__init__(optimizer, last_epoch)

    def get_wd(self):
        scale = self.scheduler(self.last_epoch)
        return [base_wd * scale for base_wd in self.base_wds]


class CosineRestartWD(WDScheduler):
    def __init__(self, optimizer: Optimizer, training_steps: int, warmup_steps: int = 0, num_cycles: int = 1,
                 min_scale: float = 0.0, last_epoch: int = -1):
        self.scheduler = cosine_with_hard_restarts_schedule_with_warmup(warmup_steps, training_steps, num_cycles,
                                                                        final_scale=min_scale)
        super().__init__(optimizer, last_epoch)

    def get_wd(self):
        scale = self.scheduler(self.last_epoch)
        return [base_wd * scale for base_wd in self.base_wds]


class PolynomialWD(WDScheduler):
    def __init__(self, optimizer: Optimizer, training_steps: int, warmup_steps: int = 0, wd_end=1e-7, power=1.0, last_epoch=-1):
        wd_init = optimizer.defaults["weight_decay"]
        self.scheduler = polynomial_decay_schedule_with_warmup(warmup_steps, training_steps, wd_init, wd_end, power)
        super().__init__(optimizer, last_epoch)

    def get_wd(self):
        scale = self.scheduler(self.last_epoch)
        return [base_wd * scale for base_wd in self.base_wds]


def get_wd_scheduler_with_name(
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
        raise NotImplementedError("one_cycle wd scheduler not implemented.")

    name = SchedulerName(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
    if name == SchedulerName.CONSTANT:
        scheduler = schedule_func()
    elif name == SchedulerName.CONSTANT_WITH_WARMUP:
        scheduler = schedule_func(num_warmup_steps=warmup_steps)
    else:
        scheduler = schedule_func(num_warmup_steps=warmup_steps, num_training_steps=training_steps, **kwargs)
    return LambdaWD(optimizer, scheduler)


def get_wd_scheduler(cfg, optimizer, num_training_steps):
    if cfg is None:
        return None
    elif isinstance(cfg, partial):
        try:
            return cfg(optimizer=optimizer, training_steps=num_training_steps)
        except:
            return cfg(optimizer=optimizer)
    else:
        return get_wd_scheduler_with_name(optimizer=optimizer, training_steps=num_training_steps, **cfg)
