
class KeyMapper:
    cls_map = {0: 'pred.pred', 1: 'target.label'}
    image_map = {0: 'pred.pred', 1: 'target.target_image'}

    def __init__(self, host=None, key_map=None):
        if key_map is None and host is not None:
            if hasattr(host, '_key_map'):
                self.key_map = host._key_map
            else:
                self.key_map = self.cls_map
        else:
            self.key_map = key_map

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
            keys = k_src.split('.')
            v = self.get_value(src, keys)
            data[k_dst] = v
        return data

    def __call__(self, **src):
        return self.map_args(**src)