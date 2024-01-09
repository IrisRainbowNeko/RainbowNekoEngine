from .base import BaseBucket
from ..utils import resize_crop_fix, pad_crop_fix
from typing import List, Tuple, Union
import cv2

class FixedBucket(BaseBucket):
    def __init__(self, target_size: Union[Tuple[int, int], int] = 512, **kwargs):
        self.target_size = (target_size, target_size) if isinstance(target_size, int) else target_size

    def build(self, bs: int, file_names: List[Tuple[str, 'DataSource']]):
        self.file_names = file_names

    def crop_resize(self, image, size, mask_interp=cv2.INTER_CUBIC):
        return resize_crop_fix(image, size, mask_interp=mask_interp)

    def __getitem__(self, idx) -> Tuple[Tuple[str, 'DataSource'], Tuple[int, int]]:
        return self.file_names[idx], self.target_size

    def __len__(self):
        return len(self.file_names)