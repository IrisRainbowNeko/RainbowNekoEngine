from typing import Dict, Any, List, Union
import torch

from PIL import Image
from .base import BasicAction, feedback_input, MemoryMixin
from rainbowneko.ckpt_manager import auto_manager

class LoadImageAction(BasicAction):
    def __init__(self, image_paths:Union[str, List[str]], image_transforms=None):
        super().__init__()
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        self.image_paths = image_paths
        self.image_transforms = image_transforms

    def load_one(self, path:str):
        img = Image.open(path).convert('RGB')
        if self.image_transforms:
            img = self.image_transforms(img)
        return img

    @feedback_input
    def forward(self, device, **states):
        input = torch.stack([self.load_one(path) for path in self.image_paths]).to(device)
        input: Dict[str, Any] = {'x':input}
        return {'input':input}

class LoadModelAction(BasicAction, MemoryMixin):
    def __init__(self, part_paths: Union[str, Dict[str, str]]):
        super().__init__()
        self.part_paths=part_paths

    @feedback_input
    def forward(self, memory, **states):
        if isinstance(self.part_paths, str):
            manager = auto_manager(self.part_paths)
        else:
            manager = auto_manager(next(iter(self.part_paths.values())))
        manager.load_to_model(memory.model, self.part_paths)