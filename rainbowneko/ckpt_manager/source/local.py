import os

from .base import CkptSource
from ..format import CkptFormat


class LocalCkptSource(CkptSource):

    def get(self, name: str, format: CkptFormat, prefix=None, **kwargs):
        path = name if prefix is None else os.path.join(prefix, name)
        return format.load_ckpt(path, **kwargs)

    def put(self, name: str, data, format: CkptFormat, prefix=None):
        path = name if prefix is None else os.path.join(prefix, name)
        return format.save_ckpt(data, path)
