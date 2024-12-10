from .base import CkptManagerBase
from .ckpt import CkptManager
from .format import PKLFormat, SafeTensorFormat
from .model import ModelManager
from .source import LocalCkptSource

def ckpt_manager(ckpt_type='pkl', plugin_from_raw=False, saved_model=({'model': '', 'trainable': True},)):
    if ckpt_type == 'pkl':
        format = PKLFormat()
    elif ckpt_type == 'safetensors':
        format = SafeTensorFormat()
    else:
        raise ValueError(f'Unknown ckpt_type: {ckpt_type}')

    return CkptManager(format, LocalCkptSource(), plugin_from_raw=plugin_from_raw, saved_model=saved_model)


def auto_manager(ckpt_path: str, **kwargs):
    ckpt_type = 'safetensors' if ckpt_path.endswith('.safetensors') else 'pkl'
    return ckpt_manager(ckpt_type=ckpt_type, **kwargs)
