from typing import Tuple, Union

import cv2
import numpy as np

from .base import BaseBucket
from ..handler import AutoSizeHandler


class PosNegBucket(BaseBucket):
    can_shuffle = False
    handler = AutoSizeHandler(mode='full')

    def __init__(self, target_size: Union[Tuple[int, int], int] = 512, pos_rate=0.5, **kwargs):
        self.target_size = (target_size, target_size) if isinstance(target_size, int) else target_size
        self.pos_rate = pos_rate

    def build(self, bs: int, world_size: int, source: 'DataSource'):
        self.bs = bs  # bs per device
        self.world_size = world_size
        self.source = source

        self.cls_group = {}  # {cls: idx}
        for i, (path, source) in enumerate(self.source):
            cls_name = source.get_class_name(path)
            if cls_name not in self.cls_group:
                self.cls_group[cls_name] = []
            self.cls_group[cls_name].append(i)
        self.cls_names = list(self.cls_group.keys())

    def rest(self, epoch):
        self.rs = np.random.RandomState(42 + epoch)

        # shuffle of batches
        img_idxs = np.arange(0, len(self.source)).astype(int)
        self.rs.shuffle(img_idxs)

        self.img_idxs = img_idxs

    def __getitem__(self, idx) -> Tuple[Tuple[str, 'DataSource'], Tuple[int, int]]:

        # world_size for DistributedSampler
        ws_bs = self.world_size * self.bs
        idx_bs = (idx // self.world_size) % self.bs
        idx_0 = idx // ws_bs * ws_bs + idx % self.world_size

        if idx_bs == 0:
            return self.source[self.img_idxs[idx]], self.target_size
        elif idx_bs < int(self.bs * self.pos_rate):  # pos
            path, source = self.source[self.img_idxs[idx_0]]
            idx_c = self.rs.choice(self.cls_group[source.get_class_name(path)])
            return self.source[idx_c], self.target_size
        else:  # neg
            path, source = self.source[self.img_idxs[idx_0]]
            cls_name = source.get_class_name(path)
            cls_name_c = self.rs.choice(self.cls_names)
            while cls_name == cls_name_c:
                cls_name_c = self.rs.choice(self.cls_names)
            idx_c = self.rs.choice(self.cls_group[cls_name_c])
            return self.source[idx_c], self.target_size

    def __len__(self):
        return len(self.source)


class TripletBucket(PosNegBucket):
    def __getitem__(self, idx) -> Tuple[Tuple[str, 'DataSource'], Tuple[int, int]]:
        assert self.bs % 3 == 0, 'batch size of TripletBucket must be a multiple of 3.'
        bs_1 = self.bs // 3

        # world_size for DistributedSampler
        ws_bs = self.world_size * self.bs
        idx_bs = (idx // self.world_size) % self.bs
        idx_0 = idx // ws_bs * ws_bs + idx % self.world_size + (idx_bs % bs_1) * self.world_size
        
        if idx_bs < bs_1:
            return self.source[self.img_idxs[idx]], self.target_size
        elif idx_bs < bs_1 * 2:  # pos
            path, source = self.source[self.img_idxs[idx_0]]
            idx_c = self.rs.choice(self.cls_group[source.get_class_name(path)])
            return self.source[idx_c], self.target_size
        else:  # neg
            path, source = self.source[self.img_idxs[idx_0]]
            cls_name = source.get_class_name(path)
            cls_name_c = self.rs.choice(self.cls_names)
            while cls_name == cls_name_c:
                cls_name_c = self.rs.choice(self.cls_names)
            idx_c = self.rs.choice(self.cls_group[cls_name_c])
            return self.source[idx_c], self.target_size

    def process_label(self, idx:int, label):
        bs_1 = self.bs // 3
        ws_bs = self.world_size * self.bs
        idx_bs = (idx // self.world_size) % self.bs
        idx_0 = idx // ws_bs * ws_bs + idx % self.world_size + (idx_bs % bs_1) * self.world_size
        
        if idx_bs < bs_1:
            label['label']['label'] = 0
        elif idx_bs < bs_1 * 2:  # pos
            label['label']['label'] = 1
        else:  # neg
            label['label']['label'] = 2
            
        return label