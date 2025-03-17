from functools import partial
from typing import Union

import torch
from tqdm import tqdm

from rainbowneko.data import CacheableDataset, BaseDataset
from rainbowneko.data import get_sampler
from rainbowneko.data.handler import DataHandler
from .base import BasicAction


class HandlerAction(BasicAction):
    def __init__(self, handler: DataHandler, key_map_in=None, key_map_out=None):
        super().__init__(key_map_in=key_map_in, key_map_out=key_map_out)
        self.handler = handler

    def forward(self, **states):
        return self.handler(states)


class DataLoaderAction(BasicAction):
    def __init__(self, dataset: Union[partial, BaseDataset], actions: BasicAction, workers=0, key_map_in=None, key_map_out=None):
        super().__init__(key_map_in=key_map_in, key_map_out=key_map_out)
        self.data_builder = dataset
        self.actions = actions
        self.workers = workers

    def build_dataset(self, data_builder: partial):
        batch_size = data_builder.keywords.pop("batch_size")

        dataset = data_builder()
        dataset.build_bucket(bs=batch_size, world_size=self.world_size)
        if isinstance(dataset, CacheableDataset):
            raise ValueError('Do not use CacheableDataset in workflow! Can not cache datas in workflow!')
        print(f"len(dataset): {len(dataset)}")

        return dataset, batch_size

    def build_data(self, data_builder: partial, train=False) -> torch.utils.data.DataLoader:
        drop_last = False
        dataset, batch_size = self.build_dataset(data_builder)

        # Pytorch Data loader
        sampler = get_sampler(train)(
            dataset,
            num_replicas=self.world_size,
            rank=self.local_rank,
            shuffle=train and dataset.bucket.can_shuffle,
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.workers,
            sampler=sampler,
            collate_fn=dataset.collate_fn,
            drop_last=drop_last,
        )
        return loader

    def to_dev(self, x, device, dtype):
        if isinstance(x, torch.Tensor):
            if torch.is_floating_point(x):
                return x.to(device, dtype=dtype)
            else:
                return x.to(device)
        else:
            return x

    def forward(self, world_size, local_rank, device, dtype, **states):
        if not hasattr(self, 'loader'):
            self.world_size = world_size
            self.local_rank = local_rank
            self.loader = self.build_data(self.data_builder)

        states.update({"world_size": world_size, "local_rank": local_rank, "device": device, "dtype": dtype})
        for data in tqdm(self.loader):
            input_datas = {k: self.to_dev(v, device, dtype) for k, v in data.items() if k != "plugin_input"}
            if "plugin_input" in data:
                input_datas["plugin_input"] = {k: self.to_dev(v, device, dtype) for k, v in data["plugin_input"].items()}

            states_in = {**states, **input_datas}
            states = self.actions(**states_in)

        return states
