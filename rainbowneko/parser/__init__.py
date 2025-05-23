from types import ModuleType

from . import hydra_patch  # patch hydra _resolve_target and _call_target
from .cfg2py import ConfigCodeReconstructor
from .model import CfgModelParser, CustomModelParser, CfgWDModelParser, CfgPluginParser, CfgWDPluginParser
from .python_cfg import PythonCfgParser, get_rel_path, disable_neko_cfg
from .recursive_partial import RecursivePartial
from .yaml_cfg import YamlCfgParser


def neko_cfg(func):
    parser = PythonCfgParser()
    return parser.compile_cfg(func)


def load_config(path: str | ModuleType, remove_undefined=True):
    if isinstance(path, ModuleType):
        parser = PythonCfgParser()
    elif path.lower().endswith('.yaml'):
        parser = YamlCfgParser()
    elif path.lower().endswith('.py'):
        parser = PythonCfgParser()
    else:
        raise ValueError('Unsupported config file format: {}'.format(path))
    return parser, parser.load_config(path, remove_undefined)


def load_config_with_cli(path: str | ModuleType, args_list=None, remove_undefined=True):
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
