import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import xformers

    xformers_available = True
except:
    xformers_available = False

def remove_all_hooks(model: nn.Module) -> None:
    for name, child in model.named_modules():
        child._forward_hooks.clear()
        child._forward_pre_hooks.clear()
        child._backward_hooks.clear()

def remove_layers(model: nn.Module, layer_class):
    named_modules = {k:v for k, v in model.named_modules()}
    for k, v in named_modules.items():
        if isinstance(v, layer_class):
            parent, name = named_modules[k.rsplit('.', 1)]
            delattr(parent, name)
            del v

def hook_compile(model):
    named_modules = {k:v for k, v in model.named_modules()}

    for name, block in named_modules.items():
        if len(block._forward_hooks)>0:
            for hook in block._forward_hooks.values():  # 从前往后执行
                old_forward = block.forward

                def new_forward(*args, **kwargs):
                    result = old_forward(*args, **kwargs)
                    hook_result = hook(block, args, result)
                    if hook_result is not None:
                        result = hook_result
                    return result

                block.forward = new_forward

        if len(block._forward_pre_hooks)>0:
            for hook in list(block._forward_pre_hooks.values())[::-1]:  # 从前往后执行
                old_forward = block.forward

                def new_forward(*args, **kwargs):
                    result = hook(block, args)
                    if result is not None:
                        if not isinstance(result, tuple):
                            result = (result,)
                    else:
                        result = args
                    return old_forward(*result, **kwargs)

                block.forward = new_forward
    remove_all_hooks(model)

def _convert_cpu(t):
    return t.to('cpu') if t.device.type == 'cuda' else t

def _convert_cuda(t):
    return t.to('cuda') if t.device.type == 'cpu' else t

def to_cpu(model):
    model._apply(_convert_cpu)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def to_cuda(model):
    model._apply(_convert_cuda)

def split_module_name(layer_name):
    name_split = layer_name.rsplit('.', 1)
    if len(name_split) == 1:
        parent_name, host_name = '', name_split[0]
    else:
        parent_name, host_name = name_split
    return parent_name, host_name

def maybe_DDP(model):
    if isinstance(model, DDP):
        return model.module
    else:
        return model

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module