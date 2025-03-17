from .base import BasicAction
import torch
from typing import List

class VisClassAction(BasicAction):
    def __init__(self, class_map: List[str], key_map_in=None, key_map_out=None):
        super().__init__(key_map_in=key_map_in, key_map_out=key_map_out)
        self.class_map = class_map

    def forward(self, pred: torch.Tensor, **states):
        for cls in pred.argmax(dim=-1):
            print(self.class_map[cls.item()])