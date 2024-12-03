from .base import BasicAction
from rainbowneko.train.data.handler import DataHandler
from functools import partial

class HandlerAction(BasicAction):
    def __init__(self, handler: DataHandler, key_map_in=None, key_map_out=None):
        super().__init__(key_map_in=key_map_in, key_map_out=key_map_out)
        self.handler = handler

    def forward(self, **states):
        return self.handler(states)

class DataLoaderAction(BasicAction):
    def __init__(self, dataset: partial, batch_size: int = 1, key_map_in=None, key_map_out=None):
        super().__init__(key_map_in=key_map_in, key_map_out=key_map_out)
        self.dataset = dataset()
