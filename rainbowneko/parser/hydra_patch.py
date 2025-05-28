import functools
from functools import partial
from typing import Union, Callable, Any, Tuple, Dict

from hydra._internal.instantiate import _instantiate2
from omegaconf import OmegaConf

from .recursive_partial import RecursivePartial


class InitAttr:
    def __init__(self, target: Callable, ex_attrs: Dict[str, Any]):
        self.target = target
        self.ex_attrs = ex_attrs

    def __call__(self, *args, **kwargs):
        obj = self.target(*args, **kwargs)
        for k, v in self.ex_attrs.items():
            setattr(obj, k, v)
        return obj


# >>>>>>>>>> patch hydra _resolve_target and _call_target >>>>>>>>>>
target_stack = []


def getattr_call(obj: RecursivePartial, attr: str):
    if type(obj) is partial:
        obj = RecursivePartial(obj)
        obj.add_sub = True
        return obj(getattr, attr)
    elif type(obj) is RecursivePartial:
        obj.add_sub = True
        return obj(getattr, attr)
    else:
        return getattr(obj, attr)


def _resolve_target(
        target: Union[str, type, Callable[..., Any]], full_key: str
) -> Union[type, Callable[..., Any]]:
    """Resolve target string, type or callable into type or callable."""
    if isinstance(target, str):
        try:
            target = _instantiate2._locate(target)
        except Exception as e:
            msg = f"Error locating target '{target}', set env var HYDRA_FULL_ERROR=1 to see chained exception."
            if full_key:
                msg += f"\nfull_key: {full_key}"
            raise _instantiate2.InstantiationException(msg) from e
    elif _instantiate2._is_target(target):  # Recursive target resolve
        target = _instantiate2.instantiate_node(target)
    if not callable(target):
        msg = f"Expected a callable target, got '{target}' of type '{type(target).__name__}'"
        if full_key:
            msg += f"\nfull_key: {full_key}"
        raise _instantiate2.InstantiationException(msg)

    if type(target) is partial:
        target = RecursivePartial(target)
        target.add_sub = True
    elif target is getattr:
        target = getattr_call
    elif type(target) is RecursivePartial:
        target.add_sub = True

    return target


def _call_target(
        _target_: Callable[..., Any],
        _partial_: bool,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        full_key: str,
) -> Any:
    """Call target (type) with args and kwargs."""
    try:
        args, kwargs = _instantiate2._extract_pos_args(args, kwargs)
        # detaching configs from parent.
        # At this time, everything is resolved and the parent link can cause
        # issues when serializing objects in some scenarios.
        for arg in args:
            if OmegaConf.is_config(arg):
                arg._set_parent(None)
        for v in kwargs.values():
            if OmegaConf.is_config(v):
                v._set_parent(None)
    except Exception as e:
        msg = (
                f"Error in collecting args and kwargs for '{_instantiate2._convert_target_to_string(_target_)}':"
                + f"\n{repr(e)}"
        )
        if full_key:
            msg += f"\nfull_key: {full_key}"

        raise _instantiate2.InstantiationException(msg) from e

    if '_ex_attrs_' in kwargs:
        _target_ = InitAttr(_target_, kwargs.pop('_ex_attrs_'))

    if _partial_:
        try:
            return functools.partial(_target_, *args, **kwargs)
        except Exception as e:
            msg = (
                    f"Error in creating partial({_instantiate2._convert_target_to_string(_target_)}, ...) object:"
                    + f"\n{repr(e)}"
            )
            if full_key:
                msg += f"\nfull_key: {full_key}"
            raise _instantiate2.InstantiationException(msg) from e
    else:
        try:
            return _target_(*args, **kwargs)
        except Exception as e:
            msg = f"Error in call to target '{_instantiate2._convert_target_to_string(_target_)}':\n{repr(e)}"
            if full_key:
                msg += f"\nfull_key: {full_key}"
            raise _instantiate2.InstantiationException(msg) from e


_instantiate2._resolve_target = _resolve_target
_instantiate2._call_target = _call_target
# <<<<<<<<<< patch hydra _resolve_target and _call_target <<<<<<<<<<
