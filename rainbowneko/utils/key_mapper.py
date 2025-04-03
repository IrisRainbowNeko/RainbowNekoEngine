from typing import Union, Iterable, Dict, Any, List

from addict import Dict as ADict

from .utils import is_dict, dict_parse_list, dict_list_copy, remove_empty_dict_list


class KeyMapper:
    cls_map = {0: 'pred.pred', 1: 'inputs.label'}  # 'pred.pred -> 0', 'inputs.label -> 1'
    image_map = {0: 'pred.pred', 1: 'inputs.target_image'}

    def __init__(self, host=None, key_map: Union[Iterable[str], Dict[Any, str], "KeyMapper"] = None, skip_missing=True,
                 move_mode=False):
        self.skip_missing = skip_missing
        self.move_mode = move_mode

        if key_map is None:
            if host is None:
                self.key_map = None
            else:
                if hasattr(host, '_key_map'):
                    self.key_map = self.parse_key_map(host._key_map)
                else:
                    self.key_map = self.cls_map
        elif isinstance(key_map, KeyMapper):
            self.key_map = key_map.key_map
            self.skip_missing = key_map.skip_missing
            self.move_mode = key_map.move_mode
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

    def get_value(self, v, keys: Union[str, List]):
        if isinstance(keys, str):
            keys = keys.split('.')
        elif isinstance(keys, int):
            keys = [keys]

        for i, k in enumerate(keys):
            if isinstance(v, (list, tuple)):
                try:
                    k = int(k)
                except ValueError:
                    raise ValueError(f'{k} in {keys} cannot index a list')
                if self.move_mode and i == len(keys) - 1:
                    v = v.pop(k)
                else:
                    v = v[k]
            else:
                if self.move_mode and i == len(keys) - 1:
                    v = v.pop(k)
                else:
                    v = v[k]
        return v

    def set_value(self, v, keys: Union[str, List], data: ADict):
        if isinstance(keys, str):
            keys = keys.split('.')
        elif isinstance(keys, int):
            keys = [keys]

        try:
            k = int(keys[0])
            data = data['args']
        except ValueError:
            data = data['kwargs']

        for k in keys[:-1]:
            try:
                k = int(k)
            except ValueError:
                pass
            data = data[k]

        try:
            k = int(keys[-1])
            data[k] = v
        except ValueError:
            data[keys[-1]] = v

    def map_data(self, src):
        if self.key_map is None:
            return [], src

        if self.move_mode:
            src = dict_list_copy(src)

        data = ADict({'args': {}, 'kwargs': {}})
        for k_dst, k_src in self.key_map.items():
            if self.skip_missing:
                try:
                    v = self.get_value(src, k_src)
                    self.set_value(v, k_dst, data)
                except KeyError:
                    pass
                except IndexError:
                    pass
            else:
                v = self.get_value(src, k_src)
                self.set_value(v, k_dst, data)
        if self.move_mode:
            src_ = ADict(src)
            src_.update(data['kwargs'])
            src_ = remove_empty_dict_list(src_)
            data['kwargs'] = src_
        data = dict_parse_list(data.to_dict())
        args, kwargs = data['args'], data['kwargs']

        return args, kwargs

    def __call__(self, **src):
        return self.map_data(src)


if __name__ == '__main__':
    mapper = KeyMapper(key_map=['0 -> input', '1 -> target.label'])
    data = ('t1', 't2')

    args, kwargs = mapper.map_data(data)
    print(args, kwargs)

    mapper = KeyMapper(key_map=['image -> input', 'target.label -> 0', 'target.image -> 1'])
    data = {'image': 'image_data', 'target': {'label': 'label_data'}}

    args, kwargs = mapper.map_data(data)
    print(args, kwargs)

    mapper = KeyMapper(key_map=['target.label -> label_1'], move_mode=True)
    args, kwargs = mapper.map_data(data)
    print(args, kwargs)  # {} {'image': 'image_data', 'label_1': 'label_data'}

    mapper = KeyMapper(key_map=['target.label -> label_1'], move_mode=True)
    data = {'image': 'image_data', 'target': {'label': 'label_data', 'cls': 'cls_data'}}
    args, kwargs = mapper.map_data(data)
    print(args, kwargs)  # {} {'image': 'image_data', 'target': {'cls': 'cls_data'}, 'label_1': 'label_data'}
