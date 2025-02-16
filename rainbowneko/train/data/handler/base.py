from typing import Dict, Any, List
from rainbowneko.utils import KeyMapper, RandomContext

class DataHandler:
    def __init__(self, key_map_in=('image -> image',), key_map_out=('image -> image',)):
        self.key_mapper_in = KeyMapper(key_map=key_map_in)
        self.key_mapper_out = KeyMapper(key_map=key_map_out)

    def handle(self, *args, **kwargs):
        return kwargs

    def __call__(self, data) -> Dict[str, Any]:
        data_proc = self.handle(**self.key_mapper_in.map_data(data)[1])
        return self.key_mapper_out.map_data(data_proc)[1]

class HandlerGroup(DataHandler):
    def __init__(self, key_map_in=None, key_map_out=None, **handlers: DataHandler):
        self.handlers = handlers
        self.key_mapper_in = KeyMapper(key_map=key_map_in) if key_map_in else None
        self.key_mapper_out = KeyMapper(key_map=key_map_out) if key_map_out else None

    def __call__(self, data:Dict[str, Any]) -> Dict[str, Any]:
        if self.key_mapper_in:
            data = self.key_mapper_in.map_data(data)[1]
        data_new = {}
        for name, handler in self.handlers.items():
            data_new.update(handler(data))
        data.update(data_new)
        if self.key_mapper_out:
            data = self.key_mapper_out.map_data(data)[1]
        return data

class HandlerChain(DataHandler):
    def __init__(self, key_map_in=None, key_map_out=None, **handlers: DataHandler):
        self.handlers = handlers
        self.key_mapper_in = KeyMapper(key_map=key_map_in) if key_map_in else None
        self.key_mapper_out = KeyMapper(key_map=key_map_out) if key_map_out else None

    def __call__(self, data:Dict[str, Any]) -> Dict[str, Any]:
        data = data.copy()
        if self.key_mapper_in:
            data = self.key_mapper_in.map_data(data)[1]
        for name, handler in self.handlers.items():
            tmp = handler(data)
            data.update(tmp)
        if self.key_mapper_out:
            data = self.key_mapper_out.map_data(data)[1]
        return data

class SyncHandler(DataHandler):
    def __init__(self, **handlers: DataHandler):
        self.handlers = handlers

    def __call__(self, data) -> Dict[str, Any]:
        random_context = RandomContext()

        data_new = {}
        for name, handler in self.handlers.items():
            with random_context:
                data_new.update(handler(data))
        data.update(data_new)
        return data