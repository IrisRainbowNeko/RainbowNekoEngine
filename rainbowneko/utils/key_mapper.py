
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

    def map_args(self, **src):
        args = []
        kwargs = {}
        for k_dst, k_src in self.key_map.items():
            keys = k_src.split('.')
            v = src[keys[0]]
            for k in keys[1:]:
                v = v[k]
            if isinstance(k_dst, int):
                args.append(v)
            else:
                kwargs[k_dst] = v
        return args, kwargs

    def __call__(self, **src):
        return self.map_args(**src)