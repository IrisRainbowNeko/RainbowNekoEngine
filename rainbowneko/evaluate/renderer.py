from rainbowneko.utils import KeyMapper
from PIL import Image
from typing import List, Callable


class Renderer:
    def __init__(self, trans: Callable, key_map=KeyMapper.cls_map):
        self.trans = trans
        self.key_mapper = KeyMapper(None, key_map)

    def render(self, image, label) -> List[Image.Image]:
        return [self.trans(item) for item in image]

    def __call__(self, pred, target):
        args, kwargs = self.key_mapper(pred=pred, target=target)
        return self.render(*args, **kwargs)

    def to(self, device):
        pass