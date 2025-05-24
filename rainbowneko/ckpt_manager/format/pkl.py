from typing import Dict, Any, Union

import torch
from rainbowneko.utils import FILE_LIKE
from .base import CkptFormat


class PKLFormat(CkptFormat):
    EXT = 'ckpt'

    def save_ckpt(self, sd_model: Union[torch.nn.Module, Dict[str, Any]], save_f: FILE_LIKE):
        torch.save(sd_model, save_f)

    def load_ckpt(self, ckpt_f: FILE_LIKE, map_location="cpu"):
        return torch.load(ckpt_f, map_location=map_location)