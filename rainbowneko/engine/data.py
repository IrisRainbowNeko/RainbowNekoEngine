from functools import partial
from typing import Dict
from typing import TYPE_CHECKING, Callable, List, Any

import torch
from omegaconf import DictConfig
from rainbowneko.data import CacheableDataset, NekoDataLoader, get_sampler
from torch.utils.data import IterableDataset


class NekoDataMixin:
    if TYPE_CHECKING:
        from rainbowneko.models.wrapper import BaseWrapper
        from rainbowneko.loggers import BaseLogger

        cfgs: Dict[str, Any] | DictConfig
        world_size: int  # Number of GPUs
        local_rank: int  # Process id (GPU id)
        model_wrapper: BaseWrapper # Model instance
        loggers: BaseLogger
        all_gather: Callable[[Any], List[Any]] # Gather objects from all process (GPUs)

    def build_dataset(self, data_builder: partial):
        batch_size = data_builder.keywords.pop("batch_size")

        dataset = data_builder()
        dataset.build_bucket(bs=batch_size, world_size=self.world_size)
        if isinstance(dataset, CacheableDataset):
            dataset.build_cache(self.model_wrapper, self.all_gather)
        self.loggers.info(f"len(dataset): {len(dataset)}")

        return dataset, batch_size

    def build_data(self, data_builder: partial, workers, train=True) -> torch.utils.data.DataLoader | NekoDataLoader:
        drop_last = data_builder.keywords.pop("drop_last", True)
        dataset, batch_size = self.build_dataset(data_builder)

        # Pytorch Data loader
        if isinstance(dataset, IterableDataset):
            sampler = None  # IterableDataset cannot be read randomly
        else:
            sampler = get_sampler(train)(
                dataset,
                num_replicas=self.world_size,
                rank=self.local_rank,
                shuffle=train and dataset.bucket.can_shuffle,
            )
        loader = NekoDataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers,
            sampler=sampler,
            collate_fn=dataset.collate_fn,
            drop_last=drop_last,
        )
        return loader
