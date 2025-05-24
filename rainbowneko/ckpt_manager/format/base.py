from typing import Dict, Any, Union

import torch
from rainbowneko.utils import FILE_LIKE


class CkptFormat:
    EXT = ''

    def save_ckpt(self, sd_model: Union[torch.nn.Module, Dict[str, Any]], save_f: FILE_LIKE):
        raise NotImplementedError

    def load_ckpt(self, ckpt_f: FILE_LIKE, map_location="cpu"):
        raise NotImplementedError

    def __repr__(self):
        return f'{self.__class__.__name__}(EXT={self.EXT})'
