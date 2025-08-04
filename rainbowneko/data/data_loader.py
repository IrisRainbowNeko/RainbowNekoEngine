import gc
import multiprocessing as mp
import platform
import warnings
from queue import Empty
from typing import Union, Iterable, Optional, Callable, List, Any, TypeVar

from torch.utils.data import Sampler, IterableDataset, RandomSampler, SequentialSampler
from torch.utils.data._utils import worker as torch_worker
from torch.utils.data._utils.worker import WorkerInfo
from rainbowneko.tools.show_info import show_note_info
from . import utils as data_utils
from .utils import DynamicBarrier

from .dataset import BaseDataset
from rainbowneko import _share

T = TypeVar('T')
_collate_fn_t = Callable[[List[T]], Any]


class NekoDataLoader:
    """
    A data loader that uses multiprocessing to efficiently load data from a dataset.
    Similar to PyTorch's DataLoader but with enhanced prefetching capabilities.
    """

    def __init__(self, dataset: Union[BaseDataset, Iterable], batch_size=1, shuffle=False,
                 sampler: Union[Sampler, Iterable, None] = None,
                 num_workers: int = 0, collate_fn: Optional[_collate_fn_t] = None,
                 generator=None, drop_last: bool = False, split_iter_worker=False,
                 prefetch_factor: Optional[int] = 2, timeout: int = 300):
        """
        Initialize the NekoDataLoader.

        Args:
            dataset: Dataset to load data from
            batch_size: Number of samples in each batch
            shuffle: Whether to shuffle the data
            sampler: Optional sampler to use for data sampling
            num_workers: Number of worker processes
            collate_fn: Function to collate samples into a batch
            generator: Random number generator for sampling
            drop_last: Whether to drop the last incomplete batch
            prefetch_factor: Number of batches to prefetch per worker
            timeout: Timeout for fetching data from workers
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.collate_fn = collate_fn if collate_fn is not None else lambda x: x
        self.generator = generator
        self.drop_last = drop_last
        self.split_iter_worker = split_iter_worker
        self.prefetch_factor = max(1, prefetch_factor) if prefetch_factor is not None else 2
        self.timeout = timeout
        self._processes = []
        self._queues = []

        # Set up the sampler
        if isinstance(dataset, IterableDataset):
            # See PyTorch NOTE [ Custom Samplers and IterableDataset ]
            sampler = -1
        elif sampler is None:
            if shuffle:
                sampler = RandomSampler(dataset, generator=generator)
            else:
                sampler = SequentialSampler(dataset)
        self.sampler = sampler

    def __del__(self):
        """Ensure proper cleanup when the dataloader is deleted."""
        self._cleanup_workers()

    def _cleanup_workers(self):
        """Clean up worker processes and release resources."""
        for p in self._processes:
            if p and p.is_alive():
                try:
                    p.terminate()
                except Exception:
                    pass

        # Clear all queues
        for q in self._queues:
            if q and Empty is not None:
                try:
                    while True:
                        q.get_nowait()
                except (Empty, Exception):
                    pass

        self._processes = []
        self._queues = []

        if gc is None: # gc maybe deleted
            return

        # Force garbage collection to clean up resources
        gc.collect()

    @staticmethod
    def _worker(worker_id, num_workers, dataset, sample_iter, queue: mp.Queue, queue_next: mp.Queue, event: mp.Event, barrier: DynamicBarrier, bs, prefetch_factor, drop_last, timeout):
        """
        Worker process function for data loading.

        Args:
            worker_id: ID of the worker
            num_workers: Total number of workers
            dataset: Dataset to load from
            sample_iter: Function to iterate over the dataset
            queue: Queue to put processed batches
            queue_next: Queue for flow control
            bs: Batch size for this worker
            prefetch_factor: Number of batches to prefetch
            drop_last: Whether to drop the last incomplete batch
            timeout: Timeout for fetching data from workers
        """
        torch_worker._worker_info = WorkerInfo(
            id=worker_id,
            num_workers=num_workers,
            seed=42,
            dataset=dataset
        )
        data_utils._neko_worker_info = data_utils.NekoWorkerInfo(
            s_idx=worker_id,
            barrier=barrier,
            event=event,
        )


        batch = []
        batch_list = []
        batch_count = 0

        def put_one():
            """Helper function to put a single batch in the queue."""
            b_i = batch_list.pop(0)
            queue.put((worker_id, b_i))
            del b_i
            gc.collect()

        try:
            for i, sample in enumerate(sample_iter(dataset)):
                try:
                    batch.append(sample)

                    # Try to put prefetched data
                    if batch_list:
                        try:
                            _ = queue_next.get_nowait()
                            put_one()
                        except Empty:
                            pass

                    # Create a new batch when current batch is filled
                    s_idx = worker_id + i * num_workers  # For load balancing
                    data_utils._neko_worker_info.s_idx = s_idx + num_workers
                    if (s_idx + num_workers) // bs > batch_count:
                        batch_list.append(batch)
                        batch = []
                        batch_count += 1
                        # Wait to put data if we've reached the prefetch limit
                        if len(batch_list) > prefetch_factor:
                            queue_next.get(timeout=timeout)
                            put_one()
                except Exception as e:
                    print(f"Worker {worker_id} encountered error processing sample {i}: {e}")
                    continue  # Skip this sample and continue
            
            barrier.deregister()

            # Handle remaining items if not dropping last batch
            if not drop_last and len(batch) > 0:
                batch_list.append(batch)

            # Send all remaining batches
            while batch_list:
                queue_next.get(timeout=timeout)
                put_one()

        except Exception as e:
            print(f"Worker {worker_id} failed with error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Signal completion and clean up resources
            queue.put((worker_id, None))
            del batch
            del batch_list
            gc.collect()

        event.wait()  # https://github.com/pytorch/pytorch/issues/60654

    @staticmethod
    def worker(worker_id, num_workers, dataset, sampler, queue, queue_next, event, barrier, bs, prefetch_factor, drop_last, timeout):
        """Worker for datasets that use a sampler."""

        def sample_iter(dataset):
            for i, idx in enumerate(sampler):
                if i % num_workers == worker_id:
                    yield dataset[idx]

        return NekoDataLoader._worker(
            worker_id, num_workers, dataset, sample_iter, queue, queue_next, event, barrier, bs, prefetch_factor, drop_last, timeout
        )

    @staticmethod
    def worker_iter(worker_id, num_workers, dataset, queue, queue_next, event, barrier, bs, prefetch_factor, drop_last, timeout, split=False):
        """Worker for iterable datasets."""

        def sample_iter(dataset):
            if split:
                for i, sample in enumerate(dataset):
                    if i % num_workers == worker_id:
                        yield sample
            else:
                for sample in dataset:
                    yield sample

        return NekoDataLoader._worker(
            worker_id, num_workers, dataset, sample_iter, queue, queue_next, event, barrier, bs, prefetch_factor, drop_last, timeout
        )

    @staticmethod
    def _worker_single(dataset, bs, drop_last, sampler=-1):
        if sampler == -1:  # Iterable dataset
            batch = []
            for i, sample in enumerate(dataset):
                batch.append(sample)
                if len(batch) == bs:
                    yield batch
                    batch = []

            if not drop_last and len(batch) > 0:
                yield batch
        else:
            # Use the sampler to iterate over the dataset
            batch = []
            for i, idx in enumerate(sampler):
                batch.append(dataset[idx])
                if len(batch) == bs:
                    yield batch
                    batch = []

            if not drop_last and len(batch) > 0:
                yield batch

    def get_context(self):
        if platform.system() == "Linux":
            available = mp.get_all_start_methods()
            
            # Try to use spawn for better compatibility
            if 'spawn' in available:
                try:
                    ctx = mp.get_context('spawn')

                    def simple_test():
                        return "test"
                    
                    p = ctx.Process(target=simple_test)
                    p.start()
                    p.join()

                    q = ctx.Queue()
                    q.put(1)
                    q.get()
                    return ctx
                except:
                    pass
            
            # Go back to fork
            show_note_info("NekoDataLoader", "'spawn' context is not available, using 'fork' context instead. Please add environment variable 'OMP_NUM_THREADS=1' for better compatibility.", once=True)
            return mp.get_context('fork')
        else:
            return mp.get_context('spawn')

    def multi_process_iterate(self, dataset, num_workers=4, bs=64):
        """
        Multiprocess dataset iteration.

        Args:
            dataset: Dataset to iterate over
            num_workers: Number of worker processes
            bs: Batch size

        Yields:
            Batches of data
        """
        if num_workers <= 0:
            # Use synchronous iteration if no workers requested
            for batch in self._worker_single(dataset, bs, self.drop_last, self.sampler):
                yield self.collate_fn(batch)
            return

        if num_workers > bs:
            warnings.warn('"num_workers > batch_size" is not support, setting num_workers to batch_size')
            num_workers = bs

        # Set up multiprocessing
        ctx = self.get_context()
        queue = ctx.Queue(maxsize=num_workers * 2)  # Double buffer for better throughput
        queue_next_list = [ctx.Queue(maxsize=self.prefetch_factor) for _ in range(num_workers)]
        event = mp.Event()  # https://github.com/pytorch/pytorch/issues/60654
        barrier = DynamicBarrier(num_workers)

        # Track all queues for cleanup
        self._queues = [queue] + queue_next_list
        self._processes = []

        # Prepare worker processes
        for worker_id in range(num_workers):
            # Distribute batch size among workers
            if self.sampler == -1:  # Iterable dataset
                p = ctx.Process(
                    target=self.worker_iter,
                    args=(worker_id, num_workers, dataset, queue, queue_next_list[worker_id], event, barrier, bs,
                          self.prefetch_factor, self.drop_last, self.timeout, self.split_iter_worker)
                )
            else:  # Regular dataset with sampler
                p = ctx.Process(
                    target=self.worker,
                    args=(worker_id, num_workers, dataset, self.sampler, queue, queue_next_list[worker_id],
                          event, barrier, bs, self.prefetch_factor, self.drop_last, self.timeout)
                )
            p.daemon = True
            p.start()
            self._processes.append(p)

        try:
            finished_workers = 0
            batch = [[] for _ in range(num_workers)]
            data_count = 0
            batch_count = 0

            # Initialize worker queue signals
            for q in queue_next_list:
                q.put(1)

            while finished_workers < num_workers:
                try:
                    # Fetch data with timeout
                    try:
                        worker_id, sample = queue.get(timeout=self.timeout)
                    except Empty:
                        # Check if workers are still alive
                        if all(not p.is_alive() for p in self._processes):
                            raise RuntimeError("All workers have died")
                        raise TimeoutError(f"Data worker timeout: {self.timeout}s")

                    # Process the data
                    if sample is None:
                        finished_workers += 1
                    else:
                        batch[worker_id] = sample
                        data_count += 1

                    if finished_workers==num_workers:
                        break

                    # Yield a batch when all workers have contributed
                    if data_count == num_workers-finished_workers:
                        data_count = 0
                        # Signal workers to continue
                        for q in queue_next_list:
                            q.put(1)

                        # Flatten and collate the batch
                        head_worker = (batch_count * bs) % num_workers
                        batch = batch[head_worker:] + batch[:head_worker]
                        batch_flatten = [lst[i] for i in range(max(map(len, batch))) for lst in batch if i < len(lst)]
                        batch_flatten = self.collate_fn(batch_flatten)
                        yield batch_flatten
                        # Reset for next batch
                        del batch_flatten
                        batch = [[] for _ in range(num_workers)]
                        batch_count += 1

                except (TimeoutError, RuntimeError) as e:
                    print(f"Error in data loading: {e}")
                    import traceback
                    traceback.print_exc()
                    break

                except Exception as e:
                    print(f"Unexpected error in main process: {e}")
                    import traceback
                    traceback.print_exc()
                    break

            event.set()

            if data_count > 0:
                head_worker = (batch_count * bs) % num_workers
                batch = batch[head_worker:] + batch[:head_worker]
                batch_flatten = [lst[i] for i in range(max(map(len, batch))) for lst in batch if lst is not None and i < len(lst)]
                batch_flatten = self.collate_fn(batch_flatten)
                yield batch_flatten
                del batch_flatten
                batch = [None for _ in range(num_workers)]

        finally:
            # Always clean up resources
            self._cleanup_workers()

    def __iter__(self):
        """Iterate over the dataset."""
        return self.multi_process_iterate(
            self.dataset,
            num_workers=self.num_workers,
            bs=self.batch_size
        )

    def __len__(self):
        """Return the number of batches."""
        try:
            if self.drop_last:
                return len(self.dataset) // self.batch_size
            else:
                return (len(self.dataset) + self.batch_size - 1) // self.batch_size
        except (TypeError, AttributeError):
            # For datasets without defined length
            return 0
