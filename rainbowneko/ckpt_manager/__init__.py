from typing import Union, List

from .base import NekoLoader, NekoSaver, LAYERS_ALL, LAYERS_TRAINABLE
from .ckpt import NekoModelSaver, NekoPluginSaver, NekoModelLoader, NekoPluginLoader, NekoEasySaver, NekoOptimizerSaver, \
    NekoOptimizerLoader
from .format import PKLFormat, SafeTensorFormat, CkptFormat
from .model import NekoPluginModuleSaver, NekoModelModuleSaver
from .resume import NekoResumer
from .source import LocalCkptSource


def ckpt_saver(ckpt_type='safetensors', layers=LAYERS_ALL, state_prefix='', target_module: Union[str, List[str]] = '',
               prefix=None):
    if ckpt_type == 'pkl':
        format = PKLFormat()
    elif ckpt_type == 'safetensors':
        format = SafeTensorFormat()
    else:
        raise ValueError(f'Unknown ckpt_type: {ckpt_type}')

    return NekoModelSaver(format, LocalCkptSource(), layers=layers, state_prefix=state_prefix, target_module=target_module,
                          prefix=prefix)


def plugin_saver(ckpt_type='safetensors', layers=LAYERS_ALL, state_prefix='', target_plugin: Union[str, List[str]] = '',
                 prefix=None, plugin_from_raw=False):
    if ckpt_type == 'pkl':
        format = PKLFormat()
    elif ckpt_type == 'safetensors':
        format = SafeTensorFormat()
    else:
        raise ValueError(f'Unknown ckpt_type: {ckpt_type}')

    return NekoPluginSaver(format, LocalCkptSource(), layers=layers, state_prefix=state_prefix, target_plugin=target_plugin,
                           prefix=prefix, plugin_from_raw=plugin_from_raw)


def ckpt_easy_saver(ckpt_type='safetensors', layers=LAYERS_ALL, state_prefix='', prefix=None):
    if ckpt_type == 'pkl':
        format = PKLFormat()
    elif ckpt_type == 'safetensors':
        format = SafeTensorFormat()
    else:
        raise ValueError(f'Unknown ckpt_type: {ckpt_type}')

    return NekoEasySaver(format, LocalCkptSource(), layers=layers, state_prefix=state_prefix, prefix=prefix)


def auto_ckpt_loader(path: str, layers=LAYERS_ALL, target_module='',
                     state_prefix=None, base_model_alpha=0.0, alpha=1.0, load_ema=False):
    if path.endswith('.safetensors'):
        format = SafeTensorFormat()
    else:
        format = PKLFormat()

    return NekoModelLoader(format, LocalCkptSource(), path=path, layers=layers, target_module=target_module,
                           state_prefix=state_prefix, base_model_alpha=base_model_alpha, alpha=alpha, load_ema=load_ema)


def auto_plugin_loader(path: str, layers=LAYERS_ALL, target_plugin='', state_prefix=None, base_model_alpha=0.0, load_ema=False,
                       **plugin_kwargs):
    if path.endswith('.safetensors'):
        format = SafeTensorFormat()
    else:
        format = PKLFormat()

    return NekoPluginLoader(format, LocalCkptSource(), path=path, layers=layers, target_plugin=target_plugin,
                            state_prefix=state_prefix, base_model_alpha=base_model_alpha, load_ema=load_ema, **plugin_kwargs)


def auto_load_ckpt(path: str, **kwargs):
    loader = auto_ckpt_loader(path=path, **kwargs)
    return loader.load(path)
