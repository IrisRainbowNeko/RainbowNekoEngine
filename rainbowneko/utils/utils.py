import hashlib
import keyword
import math
import random
import re
from itertools import cycle, islice
from pathlib import Path
from typing import Tuple, List, Any, Dict, Union

import torch
from omegaconf import OmegaConf

from .img_size_tool import types_support

size_mul = {'K': 1 << 10, 'M': 1 << 20, 'G': 1 << 30, 'T': 1 << 40}
size_key = 'TGMK'


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def low_rank_approximate(weight, rank, clamp_quantile=0.99):
    if len(weight.shape) == 4:  # conv
        weight = weight.flatten(1)
        out_ch, in_ch, k1, k2 = weight.shape

    U, S, Vh = torch.linalg.svd(weight)
    U = U[:, :rank]
    S = S[:rank]
    U = U @ torch.diag(S)

    Vh = Vh[:rank, :]

    dist = torch.cat([U.flatten(), Vh.flatten()])
    hi_val = torch.quantile(dist, clamp_quantile)
    low_val = -hi_val

    U = U.clamp(low_val, hi_val)
    Vh = Vh.clamp(low_val, hi_val)

    if len(weight.shape) == 4:
        # U is (out_channels, rank) with 1x1 conv.
        U = U.reshape(U.shape[0], U.shape[1], 1, 1)
        # V is (rank, in_channels * kernel_size1 * kernel_size2)
        Vh = Vh.reshape(Vh.shape[0], in_ch, k1, k2)
    return U, Vh


def get_cfg_range(cfg_text: str):
    dy_cfg_f = 'ln'
    if cfg_text.find(':') != -1:
        cfg_text, dy_cfg_f = cfg_text.split(':')

    if cfg_text.find('-') != -1:
        l, h = cfg_text.split('-')
        return float(l), float(h), dy_cfg_f
    else:
        return float(cfg_text), float(cfg_text), dy_cfg_f


def to_validate_file(name):
    rstr = r"[\/\\\:\*\?\"\<\>\|]"  # '/ \ : * ? " < > |'
    new_title = re.sub(rstr, "_", name)  # 替换为下划线
    return new_title[:50]


def make_mask(start, end, length):
    mask = torch.zeros(length)
    mask[int(length * start):int(length * end)] = 1
    return mask.bool()


def is_image_file(file: Union[str, Path]):
    return Path(file).suffix[1:].lower() in types_support


def factorization(dimension: int, factor: int = -1) -> Tuple[int, int]:
    find_one = lambda x: len(x) - (x.rfind('1') + 1)
    dim_bin = bin(dimension)
    num = find_one(dim_bin)
    f_max = (len(dim_bin) - 3) >> 1 if factor < 0 else find_one(bin(factor))
    num = min(num, f_max)
    return dimension >> num, 1 << num


def isinstance_list(obj, cls_list):
    for cls in cls_list:
        if isinstance(obj, cls):
            return True
    return False


def net_path_join(*args):
    return '.'.join(args).strip('.').replace('..', '.')


def mgcd(*args):
    g = args[0]
    for s in args[1:]:
        g = math.gcd(g, s)
    return g


def size_to_int(size):
    return int(size[:-3]) * size_mul[size[-3]]


def int_to_size(size):
    for i, k in zip(range(40, 0, -10), size_key):
        if size >= 1 << i:
            return f'{size >> i}{k}iB'


def prepare_seed(seeds: List[int], device='cuda'):
    return [torch.Generator(device=device).manual_seed(s or random.randint(0, 1 << 30)) for s in seeds]


def hash_str(data):
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def format_number(num):
    if num >= 1e9:
        return f'{num / 1e9:.1f}B'
    elif num >= 1e6:
        return f'{num / 1e6:.1f}M'
    elif num >= 1e3:
        return f'{num / 1e3:.1f}K'
    else:
        return str(num)


def is_list(v):
    return OmegaConf.is_list(v) or isinstance(v, list)


def is_dict(v):
    return OmegaConf.is_dict(v) or isinstance(v, dict)


def addto_dictlist(dict_list: Dict[str, List], data: Dict[str, Any], v_proc=None):
    for k, v in data.items():
        if k not in dict_list:
            dict_list[k] = []
        if v_proc is not None:
            dict_list[k].append(v_proc(v))
        else:
            dict_list[k].append(v)
    return dict_list


def set_list_value(lst, index, value, default=None):
    if index >= len(lst):
        lst.extend([default] * (index - len(lst) + 1))
    lst[index] = value


def dict_parse_list(data: Dict[str, Any]):
    if isinstance(data, dict):
        if len(data) > 0 and isinstance(next(iter(data)), int):
            return [dict_parse_list(data[i]) for i in range(len(data))]
        else:
            return {k: dict_parse_list(v) for k, v in data.items()}
    else:
        return data

def dict_list_copy(obj):
    if isinstance(obj, dict):
        return {key: dict_list_copy(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [dict_list_copy(item) for item in obj]
    else:
        return obj

def is_empty(v):
    return isinstance(v, (dict, list)) and len(v) == 0

def remove_empty_dict_list(data):
    if isinstance(data, dict):
        cleaned = {k: remove_empty_dict_list(v) for k, v in data.items()}
        return {k: v for k, v in cleaned.items() if not is_empty(v)}
    elif isinstance(data, list):
        cleaned = [remove_empty_dict_list(v) for v in data]
        return [v for v in cleaned if not is_empty(v)]
    return data

def dict_merge(dict_base, dict_override):
    """
    recursive merge dicts
    """
    merged = dict_base.copy()
    for key, value in dict_override.items():
        if key in merged:
            if isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = dict_merge(merged[key], value)
            else:
                merged[key] = value
        else:
            merged[key] = value
    return merged


def repeat_list(lst, length):
    return list(islice(cycle(lst), length))

def is_valid_variable_name(s):
    return s.isidentifier() and not keyword.iskeyword(s)
