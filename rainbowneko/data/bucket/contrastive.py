from typing import Tuple, Union, Callable

import numpy as np

from .base import BaseBucket
from ..handler import AutoSizeHandler

from multiprocessing import shared_memory
from rainbowneko import _share

class PosNegBucket(BaseBucket):
    can_shuffle = False
    handler = AutoSizeHandler(mode='full')

    def __init__(self, target_size: Union[Tuple[int, int], int] = 512, pos_rate=0.5, num_bucket=None, **kwargs):
        self.target_size = (target_size, target_size) if isinstance(target_size, int) else target_size
        self.pos_rate = pos_rate
        self.num_bucket = num_bucket # for non-indexable dataset

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
                for i, (data, source) in enumerate(self.source):
                    cls_name = data['label']
                    if cls_name not in self.cls_group:
                        self.cls_group[cls_name] = []
                    self.cls_group[cls_name].append(i)
            self.cls_group = {k: np.array(v) for k, v in self.cls_group.items()}
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
        else:
            self.data_len = len(self.source)

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
        from torch.utils.data._utils.worker import _worker_info

        initial = min(initial, bufsize)
        buckets = [[] for _ in self.num_bucket]
        count = 0

        source_iter = iter(self.source)

        select_bucket = None
        bs_count = bs
        num_workers = _worker_info.num_workers
        worker_id = _worker_info.id
        batch_idx = 0
        pos_len = int(bs * self.pos_rate)

        if worker_id == 0:
            bsize_shm = shared_memory.SharedMemory(create=True, name=f'bucket_size_{_share.local_rank}', size=num_workers * len(buckets) * 1)
            bid_shm = shared_memory.SharedMemory(create=True, name=f'bucket_id_{_share.local_rank}', size=4)
            _neko_worker_info.barrier.wait()
        else:
            _neko_worker_info.barrier.wait()
            bsize_shm = shared_memory.SharedMemory(name=f'bucket_size_{_share.local_rank}')
            bid_shm = shared_memory.SharedMemory(name=f'bucket_id_{_share.local_rank}')

        bsize_array = np.ndarray((num_workers, len(buckets)), dtype=bool, buffer=bsize_shm.buf)
        bid_array = np.ndarray(1, dtype=np.int32, buffer=bid_shm.buf)

        def pick():
            nonlocal count
            nonlocal bs_count
            nonlocal select_bucket
            bidx = _neko_worker_info.s_idx % bs
            if rs is None:
                k = 0
                if bidx>=pos_len: # select bucket for negative samples
                    for bucket in buckets:
                        if len(bucket)>0:
                            select_bucket = bucket
                            break
            else:
                if bidx >= pos_len: # select bucket for negative samples
                    candidate_bucket_idxs = np.where(np.array([len(bucket)>0 for bucket in buckets], dtype=bool))[0]
                    idx = rs.choice(candidate_bucket_idxs)
                    select_bucket = buckets[idx]
                k = rs.randint(0, len(select_bucket))

            sample = select_bucket[k]
            select_bucket[k] = select_bucket[-1]
            select_bucket.pop()
            count -= 1
            bs_count += 1
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
                        next_batch_pos_shard_nums = (pos_len - _neko_worker_info.s_idx % bs) // num_workers + 1
                        _neko_worker_info.barrier.wait()  # wait for all workers to finish last batch
                        bucket_ready = np.array([len(b) for b in buckets], dtype=np.int32) >= next_batch_pos_shard_nums
                        bsize_array[worker_id] = bucket_ready
                        _neko_worker_info.barrier.wait()  # wait for all workers to update bucket size
                        candidate_bucket_idxs = np.where(np.all(bsize_array, axis=0))[0]
                        if len(candidate_bucket_idxs) == 0:
                            continue
                        if rs is None:
                            select_bucket = candidate_bucket_idxs[0]
                        else:
                            if worker_id == 0:
                                idx = rs.choice(candidate_bucket_idxs)
                                bid_array[0] = idx
                                _neko_worker_info.barrier.wait()
                            else:
                                _neko_worker_info.barrier.wait()
                                idx = bid_array[0]
                            select_bucket = buckets[idx]
                        batch_idx += 1

                    yield pick()

            while count > 0:
                if _neko_worker_info.s_idx // bs >= batch_idx:  # new batch, select new bucket
                    next_batch_shard_nums = (pos_len - _neko_worker_info.s_idx % bs) // num_workers + 1
                    _neko_worker_info.barrier.wait()  # wait for all workers to finish last batch
                    bucket_ready = np.array([len(b) for b in buckets], dtype=np.int32) >= next_batch_shard_nums
                    bsize_array[worker_id] = bucket_ready
                    _neko_worker_info.barrier.wait()  # wait for all workers to update bucket size
                    candidate_bucket_idxs = np.where(np.all(bsize_array, axis=0))[0]
                    if len(candidate_bucket_idxs) == 0:
                        break
                    if rs is None:
                        select_bucket = candidate_bucket_idxs[0]
                    else:
                        if worker_id == 0:
                            idx = rs.choice(candidate_bucket_idxs)
                            bid_array[0] = idx
                            _neko_worker_info.barrier.wait()
                        else:
                            _neko_worker_info.barrier.wait()
                            idx = bid_array[0]
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

            if worker_id == 0:
                bsize_shm.close()
                bsize_shm.unlink()
                bid_shm.close()
                bid_shm.unlink()

    def next_data(self, shuffle=True):
        def assign_bucket(datas, source, buckets):
            cls_name = datas['label']
            if cls_name not in self.cls_id_map:
                self.cls_id_map[cls_name] = len(self.cls_id_map)
            bucket_idx = self.cls_id_map[cls_name]
            buckets[bucket_idx].append(datas)

        if not hasattr(self, 'buffer_iter'):
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
