from typing import Dict

from torch.utils.data import IterableDataset
from .dataset import BaseDataset, BaseBucket, DataHandler
from .source import WebDatasetSource, ComposeWebdsSource


class WebDataset(IterableDataset, BaseDataset):
    def __init__(self, bucket: BaseBucket = None, source: Dict[str, WebDatasetSource] = None, handler: DataHandler = None,
                 batch_handler: DataHandler = None, shuffle=True, **kwargs):
        assert all(isinstance(source, WebDatasetSource) for source in source.values()), 'WebDataset only accept WebDatasetSource.'
        self.shuffle = shuffle

        self.bucket: BaseBucket = bucket
        self.source = ComposeWebdsSource(list(source.values()))
        self.handler = handler
        self.batch_handler = batch_handler

        self.bucket.can_shuffle = False

    def __iter__(self):
        return self

    def __next__(self):
        datas = self.bucket.next_data(shuffle=self.shuffle)
        datas = self.handler(datas)
        return datas
