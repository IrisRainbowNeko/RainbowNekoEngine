from typing import Dict, Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from rainbowneko.utils import FILE_LIKE
from .base import CkptFormat


class SafeTensorFormat(CkptFormat):
    EXT = 'safetensors'

    def save_ckpt(self, sd_model: Dict[str, Any], save_f: FILE_LIKE):
        sd_unfold = self.unfold_dict(sd_model)
        sd_unfold = self.type_check(sd_unfold)
        save_file(sd_unfold, save_f)

    def load_ckpt(self, ckpt_f: FILE_LIKE, map_location='cpu'):
        with safe_open(ckpt_f, framework="pt", device=map_location) as f:
            sd_fold = self.fold_dict(f)
        return sd_fold

    @staticmethod
    def type_check(sd_unfold: Dict[str, Any]):
        sd_pruned = {}
        for k, v in sd_unfold.items():
            if isinstance(v, (float, int)):
                sd_pruned[k] = torch.tensor(v)
            elif isinstance(v, torch.Tensor):
                sd_pruned[k] = v
            else:
                pass
        return sd_pruned

    @staticmethod
    def unfold_dict(data, split_key=':'):
        dict_unfold={}

        def unfold(prefix, dict_fold):
            for k,v in dict_fold.items():
                k_new = k if prefix=='' else f'{prefix}{split_key}{k}'
                if isinstance(v, dict):
                    unfold(k_new, v)
                elif isinstance(v, list) or isinstance(v, tuple):
                    unfold(k_new, {i:d for i,d in enumerate(v)})
                else:
                    dict_unfold[k_new]=v

        unfold('', data)
        return dict_unfold

    @staticmethod
    def fold_dict(safe_f, split_key=':'):
        dict_fold = {}

        for k in safe_f.keys():
            k_list = k.split(split_key)
            dict_last = dict_fold
            for item in k_list[:-1]:
                if item not in dict_last:
                    dict_last[item] = {}
                dict_last = dict_last[item]
            dict_last[k_list[-1]]=safe_f.get_tensor(k)

        return dict_fold