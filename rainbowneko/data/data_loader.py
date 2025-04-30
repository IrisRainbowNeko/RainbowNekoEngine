import itertools
import multiprocessing as mp
from queue import Empty
from typing import Union, Iterable, Optional, Callable, List, Any, TypeVar

from torch.utils.data import Sampler, IterableDataset, RandomSampler, SequentialSampler
from torch.utils.data._utils import worker as torch_worker
from torch.utils.data._utils.worker import WorkerInfo

from .dataset import BaseDataset

T = TypeVar('T')
_collate_fn_t = Callable[[List[T]], Any]


class NekoDataLoader:
    def __init__(self, dataset: BaseDataset, batch_size=1, shuffle=False, sampler: Union[Sampler, Iterable, None] = None,
                 num_workers: int = 0, collate_fn: Optional[_collate_fn_t] = None, generator=None, drop_last: bool = False,
                 prefetch_factor: Optional[int] = 2, ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.generator = generator
        self.drop_last = drop_last
        self.prefetch_factor = prefetch_factor

        if isinstance(dataset, IterableDataset):
            # See NOTE [ Custom Samplers and IterableDataset ]
            sampler = -1
        elif sampler is None:  # give default samplers
            if shuffle:
                sampler = RandomSampler(dataset, generator=generator)  # type: ignore[arg-type]
            else:
                sampler = SequentialSampler(dataset)  # type: ignore[arg-type]
        self.sampler = sampler

    @staticmethod
    def worker(worker_id, num_workers, dataset, sampler, queue, queue_next, bs, prefetch_factor):
        """
        每个子进程执行的函数
        """
        torch_worker._worker_info = WorkerInfo(id=worker_id, num_workers=num_workers, seed=42, dataset=dataset)

        batch = []
        batch_list = []
        for i, idx in enumerate(sampler):
            sample = dataset[idx]
            batch.append(sample)

            if len(batch_list) > 0: # Try to put prefetched data
                try:
                    _ = queue.get_nowait()
                    queue.put((worker_id, batch_list.pop(0)))
                except Empty:
                    pass

            if (i + 1) % bs == 0:
                batch_list.append(batch.copy())
                batch.clear()
                if len(batch_list) > prefetch_factor: # prefetch finish, wait to put data
                    queue_next.get()
                    queue.put((worker_id, batch_list.pop(0)))
        queue.put((worker_id, None))

    @staticmethod
    def worker_iter(worker_id, num_workers, dataset, queue, queue_next, bs, prefetch_factor):
        """
        每个子进程执行的函数
        """
        torch_worker._worker_info = WorkerInfo(id=worker_id, num_workers=num_workers, seed=42, dataset=dataset)

        batch = []
        batch_list = []
        count = 0
        for i, sample in enumerate(dataset):
            batch.append(sample)
            if (i + 1) % bs == 0:
                batch_list.append(batch.copy())
                batch.clear()
                count += 1
                if count > prefetch_factor:
                    queue_next.get()
                    queue.put((worker_id, batch_list.pop(0)))
                    count -= 1

        queue.put((worker_id, None))

    def multi_process_iterate(self, dataset, num_workers=4, bs=64):
        """
        多进程迭代 dataset
        """
        ctx = mp.get_context('spawn')  # 更安全的启动方式
        queue = ctx.Queue()
        queue_next_list = [ctx.Queue() for _ in range(num_workers)]
        processes = []

        # 将 dataset 拆分成多个子迭代器
        datasets = [dataset for _ in range(num_workers)]

        for worker_id in range(num_workers):
            if self.sampler == -1:
                p = ctx.Process(target=self.worker_iter, args=(worker_id, num_workers, datasets[worker_id], queue,
                                                               queue_next_list[worker_id], bs // num_workers,
                                                               self.prefetch_factor))
            else:
                p = ctx.Process(target=self.worker, args=(worker_id, num_workers, datasets[worker_id], self.sampler, queue,
                                                          queue_next_list[worker_id], bs // num_workers, self.prefetch_factor))
            p.start()
            processes.append(p)

        finished_workers = 0
        batch = [None for _ in range(num_workers)]
        data_count = 0
        for q in queue_next_list:
            q.put(1)
        while finished_workers < num_workers:
            worker_id, sample = queue.get()
            if sample is None:
                finished_workers += 1
            else:
                batch[worker_id] = sample
                data_count += 1
            if data_count == num_workers:
                data_count = 0
                for q in queue_next_list:
                    q.put(1)
                batch_flatten = list(itertools.chain.from_iterable(batch))
                batch_flatten = self.collate_fn(batch_flatten)
                yield batch_flatten

        for p in processes:
            p.join()

    def __iter__(self):
        return self.multi_process_iterate(self.dataset, num_workers=self.num_workers, bs=self.batch_size)

    def __len__(self):
        return len(self.dataset)//self.batch_size