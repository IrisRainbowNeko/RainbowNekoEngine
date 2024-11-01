import time
import warnings
from omegaconf import OmegaConf
import importlib

def times(a,b):
    warnings.warn(f"${{times:{a},{b}}} is deprecated and will be removed in the future. Please use ${{hcp.eval:{a}*{b}}} instead.", DeprecationWarning)
    return a*b

OmegaConf.register_new_resolver("times", times)

OmegaConf.register_new_resolver("neko.eval", lambda exp: eval(exp))
OmegaConf.register_new_resolver("neko.time", lambda format="%Y-%m-%d-%H-%M-%S": time.strftime(format))

def get(name):
    module = name.split('.', 1)[0]
    exec(f'import {module}')
    return eval(name)

OmegaConf.register_new_resolver("neko.get", get) # get python object

#OmegaConf.register_new_resolver("neko.dtype", lambda dtype: dtype_dict.get(dtype, torch.float32))