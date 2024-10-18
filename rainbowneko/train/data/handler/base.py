from typing import Dict, Any, List
from rainbowneko.utils import KeyMapper, RandomContext

class DataHandler:
    def __init__(self, key_map_in=('image -> image',), key_map_out=('image -> image',)):
        self.key_mapper_in = KeyMapper(key_map=key_map_in)
        self.key_mapper_out = KeyMapper(key_map=key_map_out)

    def handle(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, data) -> Dict[str, Any]:
        data_proc = self.handle(**self.key_mapper_in.map_data(data))
        return self.key_mapper_out.map_data(data_proc)

class HandlerGroup(DataHandler):
    def __init__(self, handlers: Dict[str, DataHandler]):
        self.handlers = handlers

    def __call__(self, data) -> Dict[str, Any]:
        data_new = {}
        for name, handler in self.handlers.items():
            data_new.update(handler(data))
        return data_new

class HandlerChain(DataHandler):
    def __init__(self, handlers: Dict[str, DataHandler]):
        self.handlers = handlers

    def __call__(self, data) -> Dict[str, Any]:
        for name, handler in self.handlers.items():
            data.update(handler(data))
        return data

class SyncHandler(DataHandler):
    def __init__(self, handlers: Dict[str, DataHandler]):
        self.handlers = handlers

    def __call__(self, data) -> Dict[str, Any]:
        random_context = RandomContext()

        data_new = {}
        for name, handler in self.handlers.items():
            with random_context:
                data_new.update(handler(data))
        return data_new