from .img_label_dataset import ImageLabelDataset
from .img_pair_dataset import ImagePairDataset
from .bucket import BaseBucket, FixedBucket, FixedCropBucket, RatioBucket, SizeBucket, RatioSizeBucket, LongEdgeBucket
from .utils import CycleData
from .label_loader import JsonLabelLoader, YamlLabelLoader, TXTLabelLoader, auto_label_loader
from .sampler import DistributedCycleSampler, get_sampler


class DataGroup:
    def __init__(self, loader_list, loss_weights, cycle=True):
        self.loader_list = loader_list
        self.loss_weights = loss_weights
        self.cycle = cycle

    def __iter__(self):
        if self.cycle:
            self.data_iter_list = [iter(CycleData(loader)) for loader in self.loader_list]
        else:
            self.data_iter_list = []
            for loader in self.loader_list:
                loader.dataset.bucket.rest(0) # rest bucket
                self.data_iter_list.append(iter(loader))
        return self

    def __next__(self):
        if self.cycle:
            return [next(data_iter) for data_iter in self.data_iter_list]
        else:
            data_list = []
            for data_iter in self.data_iter_list:
                try:
                    data_list.append(next(data_iter))
                except StopIteration:
                    pass
            if len(data_list) == 0:
                raise StopIteration()
            return data_list

    def __len__(self):
        return max([len(loader) for loader in self.loader_list])

    def get_dataset(self, idx):
        return self.loader_list[idx].dataset

    def get_loss_weights(self, idx):
        return self.loss_weights[idx]