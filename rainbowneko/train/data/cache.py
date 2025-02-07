from typing import Dict, Union, List

import torch

from .dataset import BaseDataset, BaseBucket, DataSource, DataHandler


class DataCache:
    def __init__(self, pre_build:str=None):
        self.cache = self.load(pre_build) if pre_build else {}

    def before_load(self, index):
        return None, False

    def before_handler(self, index, data):
        return data

    def on_finish(self, index, data):
        if index in self.cache:
            return self.cache[index]
        else:
            self.cache[index] = data
            return data

    def on_batch(self, batch):
        return batch

    def reset(self):
        self.cache.clear()

    def save(self, path):
        torch.save(self.cache, path)

    def load(self, path):
        self.cache = torch.load(path)

class DataCacheGroup(DataCache):
    def __init__(self, *caches):
        self.caches = caches

    def before_load(self, index):
        new_data = {}
        for cache in self.caches:
            data, full = cache.before_load(index)
            if full:
                raise ValueError('caches in DataCacheGroup should not be full cache!')
            new_data.update(data)
        return new_data, False

class CacheableDataset(BaseDataset):
    def __init__(self, bucket: BaseBucket = None, source: Dict[str, DataSource] = None, handler: DataHandler = None,
                 batch_handler: DataHandler = None, cache: DataCache = None, **kwargs):
        super().__init__(bucket=bucket, source=source, handler=handler, batch_handler=batch_handler, **kwargs)
        self.cache = cache

    def __getitem__(self, index):
        datas, full = self.cache.before_load(index)
        if full:
            return datas
        datas = datas or self.bucket[index]
        datas = self.cache.before_handler(index, datas)
        datas = self.handler(datas)
        datas = self.cache.on_finish(index, datas)
        return datas

    def batch_process(self, batch: Dict[str, Union[List, torch.Tensor]]):
        if self.batch_handler is not None:
            batch = self.batch_handler(batch)
        batch = self.cache.on_batch(batch)
        return batch

    def build_cache(self):
        self.cache.reset()
        for datas in self:
            pass
