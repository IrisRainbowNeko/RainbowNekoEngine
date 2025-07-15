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
        save_file(sd_unfold, save_f)

    def load_ckpt(self, ckpt_f: FILE_LIKE, map_location='cpu'):
        with safe_open(ckpt_f, framework="pt", device=map_location) as f:
            sd_fold = self.fold_dict(f)
        return sd_fold

    @staticmethod
    def unfold_dict(data, split_key=':', meta_key='$'):
        dict_unfold={}

        def unfold(prefix, dict_fold):
            for k,v in dict_fold.items():
                k_new = k if prefix=='' else f'{prefix}{split_key}{k}'
                if isinstance(v, dict):
                    unfold(k_new, v)
                elif isinstance(v, (list, tuple)):
                    cls_name = type(v).__name__
                    unfold(f'{k_new}{meta_key}{cls_name}', {i:d for i,d in enumerate(v)})
                elif isinstance(v, (float, int, bool)):
                    cls_name = type(v).__name__
                    dict_unfold[f'{k_new}{meta_key}{cls_name}'] = torch.tensor(v)
                elif torch.is_tensor(v):
                    dict_unfold[k_new]=v
                else:
                    print(f'{k_new} with type {type(v)} not supported by SafeTensorFormat!')

        unfold('', data)
        return dict_unfold

    @staticmethod
    def fold_dict(safe_f, split_key=':', meta_key='$'):
        dict_fold = {}

        for k in safe_f.keys():
            k_list = k.split(split_key)
            dict_last = dict_fold
            for item in k_list[:-1]:
                if item not in dict_last:
                    dict_last[item] = {}
                dict_last = dict_last[item]
            dict_last[k_list[-1]]=safe_f.get_tensor(k)

        def type_recover(key, data):
            metas = key.split(meta_key)
            if len(metas) == 1:
                if isinstance(data, dict):
                    return {k.split(meta_key)[0]:type_recover(k, v) for k, v in data.items()}
                else:
                    return data
            else:
                if metas[1]=='list':
                    return [type_recover(k, v) for k, v in data.items()]
                elif metas[1]=='tuple':
                    return tuple(type_recover(k, v) for k, v in data.items())
                elif metas[1]=='int' or metas[1]=='float' or metas[1]=='bool':
                    return data.item()
                else:
                    raise ValueError(f'Unknown meta type {metas[1]}')

        return type_recover('', dict_fold)