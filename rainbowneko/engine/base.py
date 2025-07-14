"""
base.py
====================
    :Name:        train with accelerate
    :Author:      Dong Ziyi
    :Affiliation: HCP Lab, SYSU
    :Created:     10/03/2023
    :Licence:     Apache-2.0
"""

import hydra
import torch
import torch.distributed as dist
import torch.utils.checkpoint
import torch.utils.data
from omegaconf import OmegaConf, DictConfig
from typing import Any, List, TYPE_CHECKING


class NekoEngineMixin:
    if TYPE_CHECKING:
        device: torch.device # Model running device
        weight_dtype: torch.dtype # Model running dtype
        world_size: int # Number of GPUs
        local_rank: int # Process id (GPU id)

    def __init__(self, parser, cfgs_raw, **cfgs):
        torch.backends.cudnn.benchmark = True
        if len(cfgs) == 0:
            cfgs = hydra.utils.instantiate(cfgs_raw)
        else:
            cfgs = OmegaConf.create(cfgs, flags={"allow_objects": True})
        self.cfgs_raw = cfgs_raw
        self.cfgs: DictConfig = cfgs
        self.parser = parser

    def to_dev(self, x):
        if isinstance(x, torch.Tensor):
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.weight_dtype)
            else:
                return x.to(self.device)
        if isinstance(x, dict):
            return {k: self.to_dev(v) for k, v in x.items()}
        else:
            return x

    def boardcast_main(self, data):
        obj = [data]
        dist.broadcast_object_list(obj, src=0)
        return obj[0]

    def gather_to_main(self, data: Any) -> List[Any]:
        if not hasattr(self, 'gloo_group'):  # Transfer data on cpu
            self.gloo_group = dist.new_group(backend='gloo')
        if self.is_local_main_process:
            gathered_objects = [None for _ in range(self.world_size)]
        else:
            gathered_objects = None
        dist.gather_object(data, gathered_objects, dst=0, group=self.gloo_group)
        return gathered_objects

    def all_gather(self, data: Any) -> List[Any]:
        if not hasattr(self, 'gloo_group'):  # Transfer data on cpu
            self.gloo_group = dist.new_group(backend='gloo')
        gathered_objects = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered_objects, data, group=self.gloo_group)
        return gathered_objects

    def cpu_gather(self, tensor):
        if self.world_size > 1:
            if not hasattr(self, 'gloo_group'):  # Transfer data on cpu
                self.gloo_group = dist.new_group(backend='gloo')

            world_size = dist.get_world_size()
            gathered_tensors = [torch.empty_like(tensor) for _ in range(world_size)]
            dist.all_gather(gathered_tensors, tensor, group=self.gloo_group)
            return torch.cat(gathered_tensors, dim=0)
        else:
            return tensor
