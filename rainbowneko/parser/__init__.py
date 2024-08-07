import os
from .cfg_net_parser import parse_plugin_cfg, CfgModelParser, CustomModelParser, CfgWDModelParser
from .yaml_cfg import YamlCfgParser
from .python_cfg import PythonCfgParser

def load_config(path: str, remove_undefined=True):
    if path.lower().endswith('.yaml'):
        parser = YamlCfgParser()
    elif path.lower().endswith('.py'):
        parser = PythonCfgParser()
    else:
        raise ValueError('Unsupported config file format: {}'.format(path))
    return parser, parser.load_config(path, remove_undefined)

def load_config_with_cli(path: str, args_list=None, remove_undefined=True):
    if path.lower().endswith('.yaml'):
        parser = YamlCfgParser()
    elif path.lower().endswith('.py'):
        parser = PythonCfgParser()
    else:
        raise ValueError('Unsupported config file format: {}'.format(path))
    return parser, parser.load_config_with_cli(path, args_list, remove_undefined)

def get_rel_path(path):
    if not isinstance(path, str):
        path = path.__file__
    current_dir = os.getcwd()
    return os.path.relpath(path, current_dir)

def make_base(*base):
    return [get_rel_path(x) for x in base]