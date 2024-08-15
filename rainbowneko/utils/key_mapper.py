from typing import Union, Iterable, Dict, Any
from rainbowneko.utils import is_dict


class KeyMapper:
    cls_map = {0: 'pred.pred', 1: 'target.label'} # 'pred.pred -> 0', 'target.label -> 1'
    image_map = {0: 'pred.pred', 1: 'target.target_image'}

    def __init__(self, host=None, key_map: Union[Iterable[str], Dict[Any, str]] = None):
        if key_map is None and host is not None:
            if hasattr(host, '_key_map'):
                self.key_map = self.parse_key_map(host._key_map)
            else:
                self.key_map = self.cls_map
        else:
            self.key_map = self.parse_key_map(key_map)

    def parse_key_map(self, key_map: Union[Iterable[str], Dict[Any, str]]):
        if is_dict(key_map):
            return key_map
        else:
            key_map_parse = {}
            for x in key_map:
                src, dst = x.split('->')
                src, dst = src.strip(), dst.strip()

                try:
                    src = int(src)
                except:
                    pass

                try:
                    dst = int(dst)
                except:
                    pass

                key_map_parse[dst] = src
            return key_map_parse

    def get_value(self, v, keys):
        for k in keys:
            if isinstance(v, (list, tuple)):
                try:
                    k = int(k)
                except ValueError:
                    raise ValueError(f'{k} in {keys} cannot index a list')
                v = v[k]
            else:
                v = v[k]
        return v

    def map_args(self, **src):
        args = []
        kwargs = {}
        for k_dst, k_src in self.key_map.items():
            keys = k_src.split('.')
            v = self.get_value(src, keys)
            if isinstance(k_dst, int):
                args.append(v)
            else:
                kwargs[k_dst] = v
        return args, kwargs

    def map_data(self, src):
        data = {}
        for k_dst, k_src in self.key_map.items():
            if isinstance(k_src, int):
                keys = [k_src]
            else:
                keys = k_src.split('.')
            v = self.get_value(src, keys)
            data[k_dst] = v
        return data

    def __call__(self, **src):
        return self.map_args(**src)
