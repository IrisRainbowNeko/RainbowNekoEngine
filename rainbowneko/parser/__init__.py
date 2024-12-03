from .model import CfgModelParser, CustomModelParser, CfgWDModelParser
from .python_cfg import PythonCfgParser
from .yaml_cfg import YamlCfgParser


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
