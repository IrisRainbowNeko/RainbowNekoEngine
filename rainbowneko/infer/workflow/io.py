from typing import Dict, Any, List, Union
import torch

from PIL import Image
from .base import BasicAction
from rainbowneko.ckpt_manager import NekoLoader

class LoadImageAction(BasicAction):
    def __init__(self, image_paths:Union[str, List[str]], image_transforms=None, key_map_in=None, key_map_out=None):
        super().__init__(key_map_in=key_map_in, key_map_out=key_map_out)
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        self.image_paths = image_paths
        self.image_transforms = image_transforms

    def load_one(self, path:str):
        img = Image.open(path).convert('RGB')
        if self.image_transforms:
            img = self.image_transforms(img)
        return img

    def forward(self, device, **states):
        input = torch.stack([self.load_one(path) for path in self.image_paths]).to(device)
        input: Dict[str, Any] = {'x':input}
        return {'input':input}

class LoadModelAction(BasicAction):
    def __init__(self, cfg: Dict[str, Any], key_map_in=None, key_map_out=None):
        super().__init__(key_map_in=key_map_in, key_map_out=key_map_out)
        self.cfg=cfg

    def forward(self, model, all_plugin_group=None, in_preview=False, **states):
        if not in_preview:
            NekoLoader.load_all(self.cfg, model=model, plugin_groups=all_plugin_group)

class FeedAction(BasicAction):
    def __init__(self, key_map_in=None, key_map_out=None, **datas):
        super().__init__(key_map_in=key_map_in, key_map_out=key_map_out)
        self.datas = datas

    def forward(self, **states):
        return self.datas
