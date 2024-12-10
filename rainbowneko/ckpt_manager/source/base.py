from ..format import CkptFormat


class CkptSource:

    def get(self, name: str, format: CkptFormat, prefix=None, **kwargs):
        raise NotImplementedError

    def put(self, name: str, data, format: CkptFormat, prefix=None):
        raise NotImplementedError

    def __repr__(self):
        return f'{self.__class__.__name__}()'
