from functools import partial
from typing import Union, Callable, Any
from types import ModuleType

from hydra._internal.instantiate import _instantiate2

from .recursive_partial import RecursivePartial

####### patch hydra resolve target #######
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

_instantiate2._resolve_target = _resolve_target
####### patch hydra resolve target #######

from .model import CfgModelParser, CustomModelParser, CfgWDModelParser, CfgPluginParser, CfgWDPluginParser
from .python_cfg import PythonCfgParser, get_rel_path
from .yaml_cfg import YamlCfgParser


def load_config(path: Union[str, ModuleType], remove_undefined=True):
    if isinstance(path, ModuleType):
        parser = PythonCfgParser()
    elif path.lower().endswith('.yaml'):
        parser = YamlCfgParser()
    elif path.lower().endswith('.py'):
        parser = PythonCfgParser()
    else:
        raise ValueError('Unsupported config file format: {}'.format(path))
    return parser, parser.load_config(path, remove_undefined)


def load_config_with_cli(path: Union[str, ModuleType], args_list=None, remove_undefined=True):
    if isinstance(path, ModuleType):
        parser = PythonCfgParser()
    elif path.lower().endswith('.yaml'):
        parser = YamlCfgParser()
    elif path.lower().endswith('.py'):
        parser = PythonCfgParser()
    else:
        raise ValueError('Unsupported config file format: {}'.format(path))
    return parser, parser.load_config_with_cli(path, args_list, remove_undefined)

def load_config_instant(path: str, remove_undefined=True):
    import hydra
    if path.lower().endswith('.yaml'):
        parser = YamlCfgParser()
    elif path.lower().endswith('.py'):
        parser = PythonCfgParser()
    else:
        raise ValueError('Unsupported config file format: {}'.format(path))
    cfg = parser.load_config(path, remove_undefined)
    return hydra.utils.instantiate(cfg)