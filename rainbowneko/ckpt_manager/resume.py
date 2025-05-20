from typing import Dict

from .base import NekoLoader


class NekoResumer:
    def __init__(self, loader: Dict[str, NekoLoader], start_step: int):
        self.loader = loader
        self.start_step = start_step

    def load_to(self, **kwargs):
        NekoLoader.load_all(cfg=self.loader, **kwargs)
