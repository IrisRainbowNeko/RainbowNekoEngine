from typing import List, Tuple, Union

from rainbowneko.utils import RandomContext

from .base import DataHandler
import torch


class FrameHandler(DataHandler):
    def __init__(self, handler: DataHandler, cat=False, key_map_in=('frames -> frames',), key_map_out=('frames -> frames',)):
        super().__init__(key_map_in, key_map_out)
        self.handler = handler
        self.cat = cat

    def handle(self, frames: Union[List, Tuple]):
        assert isinstance(frames, list) or isinstance(frames, tuple), "FrameHandler: frames should be a list or tuple"

        random_context = RandomContext()
        frames_new = []
        for frame in frames:
            with random_context:
                frames_new.append(self.handler(frame))
        if self.cat:
            frames_new = torch.cat(frames_new, 0)
        return {'frames': frames_new}


class FrameCatHandler(DataHandler):
    def __init__(self, key_map_in=('frames -> frames',), key_map_out=('frames -> frames',)):
        super().__init__(key_map_in, key_map_out)

    def handle(self, frames: Union[List, Tuple]):
        return {'frames': torch.stack(frames, 0)} # [T, ...]