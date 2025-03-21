from typing import Tuple, Union

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
        for i, (data, source) in enumerate(self.source):
            cls_name = data['label']
            if cls_name not in self.cls_group:
                self.cls_group[cls_name] = []
            self.cls_group[cls_name].append(i)
        self.cls_group = {k: np.array(v) for k, v in self.cls_group.items()}
        self.cls_names = list(self.cls_group.keys())

    def rest(self, epoch):
        self.rs = np.random.RandomState(42 + epoch)

        # build bucket
        bucket = []
        pos_len = int(self.bs * self.pos_rate)
        neg_len = self.bs - pos_len
        for name, group in self.cls_group.items():
            rest = len(group) % pos_len
            if rest > 0:
                group = np.hstack([group, self.rs.choice(group, pos_len - rest)])

            group = group.reshape(-1, pos_len)
            neg = np.hstack([g_neg for name_neg, g_neg in self.cls_group.items() if name_neg != name])
            group_neg = self.rs.choice(neg, (group.shape[0], neg_len), replace=False)
            group = np.hstack([group, group_neg])
            bucket.append(group)

        bucket = np.vstack(bucket)  # [N,bs]
        rest = len(bucket) % self.world_size
        if rest > 0:
            idx = self.rs.random_integers(0, len(bucket) - 1, self.world_size - rest)
            bucket = np.vstack([bucket, bucket[idx]])
        bucket = bucket.reshape(-1, self.world_size, self.bs).transpose(0, 2, 1)

        self.img_idxs = bucket.flatten()
        self.data_len = len(self.img_idxs)

    def __getitem__(self, idx) -> Tuple[Tuple[str, 'DataSource'], Tuple[int, int]]:
        file_idx = self.img_idxs[idx]
        datas = self.source[file_idx]
        datas['image_size'] = self.target_size
        return datas

    def __len__(self):
        return self.data_len


class TripletBucket(PosNegBucket):
    def rest(self, epoch):
        assert self.bs % 3 == 0, 'batch size of TripletBucket must be a multiple of 3.'
        self.rs = np.random.RandomState(42 + epoch)

        # build bucket
        bucket = []
        pos_len = 2 * self.bs // 3
        neg_len = self.bs - pos_len
        for name, group in self.cls_group.items():
            rest = len(group) % pos_len
            if rest > 0:
                group = np.hstack([group, self.rs.choice(group, pos_len - rest)])

            group = group.reshape(-1, pos_len)
            neg = np.hstack([g_neg for name_neg, g_neg in self.cls_group.items() if name_neg != name])
            group_neg = self.rs.choice(neg, (group.shape[0], neg_len), replace=False)
            group = np.hstack([group, group_neg])
            bucket.append(group)

        bucket = np.vstack(bucket)  # [N,bs]
        rest = len(bucket) % self.world_size
        if rest > 0:
            idx = self.rs.random_integers(0, len(bucket) - 1, self.world_size - rest)
            bucket = np.vstack([bucket, bucket[idx]])

        # shuffle triplets
        bucket = bucket.reshape(-1, 3, self.bs // 3).transpose(0, 2, 1)
        self.rs.shuffle(bucket)
        bucket = bucket.transpose(0, 2, 1).reshape(-1, self.bs)

        bucket = bucket.reshape(-1, self.world_size, self.bs).transpose(0, 2, 1)

        self.img_idxs = bucket.flatten()
        self.data_len = len(self.img_idxs)

    def __getitem__(self, idx) -> Tuple[Tuple[str, 'DataSource'], Tuple[int, int]]:
        file_idx = self.img_idxs[idx]
        datas = self.source[file_idx]
        datas['image_size'] = self.target_size
        return datas
