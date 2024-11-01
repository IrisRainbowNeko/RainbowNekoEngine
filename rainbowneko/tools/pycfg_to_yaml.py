import argparse
import re

import yaml
from omegaconf import OmegaConf
from omegaconf.omegaconf import get_omega_conf_dumper, _ensure_container
from rainbowneko.parser import PythonCfgParser


class NoAliasDumper(get_omega_conf_dumper()):
    def ignore_aliases(self, data):
        return True


def to_yaml(cfg, *, resolve: bool = False, sort_keys: bool = False, alias: bool = False) -> str:
    """
    returns a yaml dump of this config object.

    :param cfg: Config object, Structured Config type or instance
    :param resolve: if True, will return a string with the interpolations resolved, otherwise
        interpolations are preserved
    :param sort_keys: If True, will print dict keys in sorted order. default False.
    :return: A string containing the yaml representation.
    """
    cfg = _ensure_container(cfg)
    container = OmegaConf.to_container(cfg, resolve=resolve, enum_to_str=True)
    return yaml.dump(  # type: ignore
        container,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=sort_keys,
        Dumper=get_omega_conf_dumper() if alias else NoAliasDumper,
    )


def remove_wrapping(text, prefix='!!python/name:', suffix=" ''"):
    pattern = prefix + r"([^'\n]+)" + suffix
    result = re.sub(pattern, r'\1', text)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Python cfg to yaml cfg")
    parser.add_argument("cfg", type=str, default=None)
    parser.add_argument('--alias', action='store_true')
    parser.add_argument('--save', type=str, default=None)
    args = parser.parse_args()

    parser = PythonCfgParser()
    py_cfg = parser.load_cfg(args.cfg)
    yaml_cfg = to_yaml(py_cfg)
    yaml_cfg = remove_wrapping(yaml_cfg)
    #yaml_cfg = remove_wrapping(yaml_cfg, prefix='!!python/object:', suffix='')

    if args.save is None:
        print(yaml_cfg)
    else:
        with open(args.save, 'w') as f:
            f.write(yaml_cfg)
