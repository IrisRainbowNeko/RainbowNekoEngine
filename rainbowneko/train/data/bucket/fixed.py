from .base import BaseBucket
from ..utils import resize_crop_fix, pad_crop_fix
from typing import List, Tuple, Union
import cv2

class FixedBucket(BaseBucket):
    def __init__(self, target_size: Union[Tuple[int, int], int] = 512, **kwargs):
        self.target_size = (target_size, target_size) if isinstance(target_size, int) else target_size

    def build(self, bs: int, world_size: int, source: 'DataSource'):
        self.source = source

    def crop_resize(self, image, size, mask_interp=cv2.INTER_CUBIC):
        return resize_crop_fix(image, size, mask_interp=mask_interp)

    def __getitem__(self, idx) -> Tuple[Tuple[str, 'DataSource'], Tuple[int, int]]:
        return self.source[idx], self.target_size

    def __len__(self):
        return len(self.source)

class FixedCropBucket(BaseBucket):
    def __init__(self, target_size: Union[Tuple[int, int], int] = 512, **kwargs):
        self.target_size = (target_size, target_size) if isinstance(target_size, int) else target_size

    def build(self, bs: int, world_size: int, source: 'DataSource'):
        self.source = source

    def crop_resize(self, image, size, mask_interp=cv2.INTER_CUBIC):
        return pad_crop_fix(image, size)

    def __getitem__(self, idx) -> Tuple[Tuple[str, 'DataSource'], Tuple[int, int]]:
        return self.source[idx], self.target_size

    def __len__(self):
        return len(self.source)