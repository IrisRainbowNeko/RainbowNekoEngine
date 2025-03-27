from typing import Tuple, Union

from .base import BaseBucket
from ..handler import AutoSizeHandler


class FixedBucket(BaseBucket):
    handler = AutoSizeHandler()

    def __init__(self, target_size: Union[Tuple[int, int], int] = 512, **kwargs):
        self.target_size = (target_size, target_size) if isinstance(target_size, int) else target_size

    def __getitem__(self, idx) -> Tuple[Tuple[str, 'DataSource'], Tuple[int, int]]:
        datas = self.source[idx]
        datas['image_size'] = self.target_size
        return datas

    def __len__(self):
        return len(self.source)


class FixedCropBucket(FixedBucket):
    handler = AutoSizeHandler(mode='pad')
