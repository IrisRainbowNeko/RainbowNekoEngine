from .dataset import BaseDataset
from .bucket import BaseBucket, FixedBucket, FixedCropBucket, RatioBucket, SizeBucket, RatioSizeBucket, LongEdgeBucket
from .utils import CycleData
from .label_loader import JsonLabelLoader, YamlLabelLoader, TXTLabelLoader, auto_label_loader
from .sampler import DistributedCycleSampler, get_sampler
from typing import Dict, Any


class DataGroup:
    def __init__(self, loader_dict:Dict[str, Any], loss_weights:Dict[str, float], cycle=True):
        self.loader_dict = loader_dict
        self.loss_weights = loss_weights
        self.cycle = cycle

    def __iter__(self):
        if self.cycle:
            self.data_iter_dict = {name:iter(CycleData(loader)) for name, loader in self.loader_dict.items()}
        else:
            self.data_iter_dict = {}
            for name, loader in self.loader_dict.items():
                loader.dataset.bucket.rest(0) # rest bucket
                self.data_iter_dict[name] = iter(loader)
        return self

    def __next__(self):
        if self.cycle:
            return {name:next(data_iter) for name, data_iter in self.data_iter_dict.items()}
        else:
            data_dict = {}
            for name, data_iter in self.data_iter_dict.items():
                try:
                    data_dict[name] = next(data_iter)
                except StopIteration:
                    pass
            if len(data_dict) == 0:
                raise StopIteration()
            return data_dict

    def __len__(self):
        return max([len(loader) for loader in self.loader_dict.values()])

    def get_dataset(self, name):
        return self.loader_dict[name].dataset

    def get_loss_weights(self, name):
        return self.loss_weights[name]
    
    def first_loader(self):
        return next(iter(self.loader_dict.values()))