from typing import Dict

from .dataset import BaseDataset, BaseBucket, DataHandler
from .source import WebDatasetSource


class WebDataset(BaseDataset):
    def __init__(self, bucket: BaseBucket = None, source: Dict[str, WebDatasetSource] = None, handler: DataHandler = None,
                 batch_handler: DataHandler = None, size: int = 10000, shuffle=True, **kwargs):
        assert all(isinstance(source, WebDatasetSource) for source in source.values()), 'WebDataset only accept WebDatasetSource.'
        self.size = size
        self.shuffle = shuffle
        super().__init__(bucket=bucket, source=source, handler=handler, batch_handler=batch_handler, **kwargs)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        datas = self.bucket.next_data(shuffle=self.shuffle)
        datas = self.handler(datas)
        return datas
