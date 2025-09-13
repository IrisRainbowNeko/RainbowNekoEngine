from typing import Tuple, Union, Callable

import numpy as np

from .base import BaseBucket
from ..handler import AutoSizeHandler

from multiprocessing import shared_memory
from rainbowneko import _share
from tqdm import tqdm


class PosNegBucket(BaseBucket):
    can_shuffle = False
    handler = AutoSizeHandler(mode='full')

    def __init__(self, target_size: Union[Tuple[int, int], int] = 512, pos_rate=0.5, num_bucket=None, **kwargs):
        self.target_size = (target_size, target_size) if isinstance(target_size, int) else target_size
        self.pos_rate = pos_rate
        self.num_bucket = num_bucket  # for non-indexable dataset

    def build(self, bs: int, world_size: int, source: 'DataSource'):
        self.bs = bs  # bs per device
        self.world_size = world_size
        self.source = source

        try:
            _ = self.source[0]
            self.source_indexable = True
        except NotImplementedError:
            self.source_indexable = False

        if self.source_indexable:
            self.cls_group = {}  # {cls: idx}
            with self.source.return_source():
                for i, (data, source) in enumerate(tqdm(self.source)):
                    cls_name = data['label']
                    if cls_name not in self.cls_group:
                        self.cls_group[cls_name] = []
                    self.cls_group[cls_name].append(i)
            self.cls_group = {k: np.array(v) for k, v in self.cls_group.items()}
            self.cls_group_idxs = np.hstack(list(self.cls_group.values()))
            self.cls_group_start = np.concatenate([[0], np.cumsum(list(map(len, self.cls_group.values())))])
            self.cls_names = list(self.cls_group.keys())
        else:
            self.cls_id_map = {}

        self.rest(0)

    def rest(self, epoch):
        self.rs = np.random.RandomState(42 + epoch)

        if self.source_indexable:
            # build bucket
            bucket = []
            pos_len = int(self.bs * self.pos_rate)
            neg_len = self.bs - pos_len
            for i, (name, group) in enumerate(self.cls_group.items()):
                rest = len(group) % pos_len
                if rest > 0:
                    group = np.hstack([group, self.rs.choice(group, pos_len - rest)])
                pos_num = len(group)

                group = group.reshape(-1, pos_len)
                neg_idx = self.rs.randint(0, len(self.cls_group_idxs) - pos_num, neg_len * group.shape[0])
                neg_idx[neg_idx >= self.cls_group_start[i]] += pos_num
                group_neg = self.cls_group_idxs[neg_idx].reshape(-1, neg_len)

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
        else:
            self.data_len = len(self.source)
            self.buffer_iter = None

    def __getitem__(self, idx) -> Tuple[Tuple[str, 'DataSource'], Tuple[int, int]]:
        file_idx = self.img_idxs[idx]
        datas = self.source[file_idx]
        datas['image_size'] = self.target_size
        return datas

    def __len__(self):
        return self.data_len

    def _buffer(self, bs, assign_bucket: Callable, bufsize=1000, initial=100, rs: np.random.RandomState = None):
        """Bucket and shuffle the data in the stream.

        This uses a buffer of size `bufsize`. Bucketing and shuffling datas of non-indexable source.

        bs: batch size
        assign_bucket: function for assign data to bucket
        bufsize: buffer size
        returns: iterator
        """
        from rainbowneko.data.utils import _neko_worker_info

        initial = min(initial, bufsize)
        buckets = [[] for _ in range(self.num_bucket)]
        count = 0

        source_iter = iter(self.source)

        select_bucket = None
        batch_idx = 0
        pos_len = int(bs * self.pos_rate)

        def pick():
            nonlocal count
            nonlocal select_bucket
            bidx = _neko_worker_info.s_idx % bs
            if rs is None:
                k = 0
                if bidx >= pos_len:  # select bucket for negative samples
                    for bucket in buckets:
                        if len(bucket) > 0:
                            select_bucket = bucket
                            break
            else:
                if bidx >= pos_len:  # select bucket for negative samples
                    candidate_bucket_idxs = np.where(np.array([len(bucket) > 0 for bucket in buckets], dtype=bool))[0]
                    idx = rs.choice(candidate_bucket_idxs)
                    select_bucket = buckets[idx]
                k = rs.randint(0, len(select_bucket))

            sample = select_bucket[k]
            select_bucket[k] = select_bucket[-1]
            select_bucket.pop()
            count -= 1
            return sample

        with self.source.return_source():
            for datas, source in source_iter:
                assign_bucket(datas, source, buckets)
                count += 1

                if count < bufsize:
                    try:
                        assign_bucket(*next(source_iter), buckets)
                        count += 1
                    except StopIteration:
                        pass

                if count >= initial:
                    if _neko_worker_info.s_idx // bs >= batch_idx:  # new batch, select new bucket
                        bucket_ready = np.array([len(b) for b in buckets], dtype=np.int32) >= pos_len
                        candidate_bucket_idxs = np.where(bucket_ready)[0]
                        if len(candidate_bucket_idxs) == 0:
                            continue
                        if rs is None:
                            select_bucket = candidate_bucket_idxs[0]
                        else:
                            idx = rs.choice(candidate_bucket_idxs)
                            select_bucket = buckets[idx]
                        batch_idx += 1

                    yield pick()

            while count > 0:
                if _neko_worker_info.s_idx // bs >= batch_idx:  # new batch, select new bucket
                    bucket_ready = np.array([len(b) for b in buckets], dtype=np.int32) >= pos_len
                    candidate_bucket_idxs = np.where(bucket_ready)[0]
                    if len(candidate_bucket_idxs) == 0:
                        break
                    if rs is None:
                        select_bucket = candidate_bucket_idxs[0]
                    else:
                        idx = rs.choice(candidate_bucket_idxs)
                        select_bucket = buckets[idx]
                    batch_idx += 1

                yield pick()

            # # repeat the rest data to fill a batch
            # bucket_no_empty = np.array([len(b) for b in buckets], dtype=np.int32) > 0
            # bsize_array[worker_id] = bucket_no_empty
            # _neko_worker_info.barrier.wait()
            # candidate_bucket_idxs = np.where(np.all(bsize_array, axis=0))[0]
            # rs.shuffle(candidate_bucket_idxs)
            # for idx in candidate_bucket_idxs:

    def next_data(self, shuffle=True):
        def assign_bucket(datas, source, buckets):
            cls_name = datas['label']
            if cls_name not in self.cls_id_map:
                self.cls_id_map[cls_name] = len(self.cls_id_map)
            bucket_idx = self.cls_id_map[cls_name]
            datas['label'] = bucket_idx
            buckets[bucket_idx].append(datas)

        if getattr(self, 'buffer_iter', None) is None:
            self.buffer_iter = self._buffer(self.bs, assign_bucket, rs=self.rs if shuffle else None)
        return next(self.buffer_iter)


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


class CategoryBucket(BaseBucket):
    can_shuffle = False
    handler = AutoSizeHandler(mode='full')

    def __init__(self, target_size: Union[Tuple[int, int], int] = 512, min_cat_size=2, num_bucket=None, **kwargs):
        self.target_size = (target_size, target_size) if isinstance(target_size, int) else target_size
        self.min_cat_size = min_cat_size
        self.num_bucket = num_bucket  # for non-indexable dataset

    def build(self, bs: int, world_size: int, source: 'DataSource'):
        self.bs = bs  # bs per device
        self.world_size = world_size
        self.source = source

        try:
            _ = self.source[0]
            self.source_indexable = True
        except NotImplementedError:
            self.source_indexable = False

        if self.source_indexable:
            rs = np.random.RandomState(42)

            self.cls_group = {}  # {cls: idx}
            with self.source.return_source():
                for i, (data, source) in enumerate(tqdm(self.source)):
                    cls_name = data['label']
                    if cls_name not in self.cls_group:
                        self.cls_group[cls_name] = []
                    self.cls_group[cls_name].append(i)
            self.cls_names = list(self.cls_group.keys())

            for name, group in self.cls_group.items():
                rest = len(group) % self.min_cat_size
                if rest > 0:
                    group.extend(rs.choice(group, self.min_cat_size - rest))
                self.cls_group[name] = np.array(group)
        else:
            self.cls_id_map = {}

        self.rest(0)

    def rest(self, epoch):
        self.rs = np.random.RandomState(42 + epoch)

        if self.source_indexable:
            # build bucket
            bucket_list = [x.copy() for x in self.cls_group.values()]
            # shuffle inter bucket
            for x in bucket_list:
                self.rs.shuffle(x)

            # shuffle of batches
            bucket_list = np.hstack(bucket_list).reshape(-1, self.min_cat_size).astype(int)
            self.rs.shuffle(bucket_list)

            self.img_idxs = bucket_list.reshape(-1)
            self.data_len = len(self.img_idxs)
        else:
            self.data_len = len(self.source)
            self.buffer_iter = None

    def __getitem__(self, idx) -> Tuple[Tuple[str, 'DataSource'], Tuple[int, int]]:
        file_idx = self.img_idxs[idx]
        datas = self.source[file_idx]
        datas['image_size'] = self.target_size
        return datas

    def __len__(self):
        return self.data_len

    def _buffer(self, bs, assign_bucket: Callable, bufsize=1000, initial=100, rs: np.random.RandomState = None):
        """Bucket and shuffle the data in the stream.

        This uses a buffer of size `bufsize`. Bucketing and shuffling datas of non-indexable source.

        bs: batch size
        assign_bucket: function for assign data to bucket
        bufsize: buffer size
        returns: iterator
        """
        from rainbowneko.data.utils import _neko_worker_info

        initial = min(initial, bufsize)
        buckets = [[] for _ in range(self.num_bucket)]
        count = 0

        source_iter = iter(self.source)

        batch_idx = 0

        def pick():
            nonlocal count
            nonlocal select_buckets
            bidx = _neko_worker_info.s_idx % bs

            select_bucket = buckets[select_buckets[bidx]]

            if rs is None:
                k = 0
            else:
                k = rs.randint(0, len(select_bucket))

            sample = select_bucket[k]
            select_bucket[k] = select_bucket[-1]
            select_bucket.pop()
            count -= 1
            return sample

        with self.source.return_source():
            for datas, source in source_iter:
                assign_bucket(datas, source, buckets)
                count += 1

                if count < bufsize:
                    try:
                        assign_bucket(*next(source_iter), buckets)
                        count += 1
                    except StopIteration:
                        pass

                if count >= initial:
                    if _neko_worker_info.s_idx // bs >= batch_idx:  # new batch, select new bucket
                        bucket_sizes = np.array([len(b) for b in buckets], dtype=np.int32)
                        ready_buckets = (bucket_sizes >= self.min_cat_size)
                        ready = bucket_sizes[ready_buckets].sum() >= bs

                        if not ready:
                            continue

                        bucket_idxs_up = np.argsort(bucket_sizes[ready_buckets])
                        bucket_idxs_up = np.where(ready_buckets)[0][bucket_idxs_up]
                        select_buckets = np.repeat(bucket_idxs_up, bucket_sizes[bucket_idxs_up])

                        if rs is None:
                            select_buckets = select_buckets[:bs]
                        else:
                            rs.shuffle(select_buckets)
                            select_buckets = select_buckets[:bs]

                        batch_idx += 1

                    yield pick()

            while count > 0:
                if _neko_worker_info.s_idx // bs >= batch_idx:  # new batch, select new bucket
                    bucket_sizes = np.array([len(b) for b in buckets], dtype=np.int32)
                    ready_buckets = (bucket_sizes >= self.min_cat_size)
                    ready = bucket_sizes[ready_buckets].sum() >= bs

                    if not ready:
                        continue

                    bucket_idxs_up = np.argsort(bucket_sizes[ready_buckets])
                    bucket_idxs_up = np.where(ready_buckets)[0][bucket_idxs_up]
                    select_buckets = np.repeat(bucket_idxs_up, bucket_sizes[bucket_idxs_up])

                    if rs is None:
                        select_buckets = select_buckets[:bs]
                    else:
                        rs.shuffle(select_buckets)
                        select_buckets = select_buckets[:bs]

                    batch_idx += 1

                yield pick()

            # # repeat the rest data to fill a batch
            # bucket_no_empty = np.array([len(b) for b in buckets], dtype=np.int32) > 0
            # bsize_array[worker_id] = bucket_no_empty
            # _neko_worker_info.barrier.wait()
            # candidate_bucket_idxs = np.where(np.all(bsize_array, axis=0))[0]
            # rs.shuffle(candidate_bucket_idxs)
            # for idx in candidate_bucket_idxs:

    def next_data(self, shuffle=True):
        def assign_bucket(datas, source, buckets):
            cls_name = datas['label']
            if cls_name not in self.cls_id_map:
                self.cls_id_map[cls_name] = len(self.cls_id_map)
            bucket_idx = self.cls_id_map[cls_name]
            datas['label'] = bucket_idx
            buckets[bucket_idx].append(datas)

        if getattr(self, 'buffer_iter', None) is None:
            self.buffer_iter = self._buffer(self.bs, assign_bucket, rs=self.rs if shuffle else None)
        return next(self.buffer_iter)