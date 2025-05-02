from unittest import TestCase

import torch
from rainbowneko.data import NekoDataLoader
from itertools import zip_longest


class DataLoaderTester(TestCase):
    def run_loader(self, dataset, batch_size, drop_last, num_workers):
        loader = NekoDataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=None,
            drop_last=drop_last,
        )

        loader_torch = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=None,
            drop_last=drop_last,
        )

        assert len(loader) == len(loader_torch)
        for i, (data, data_torch) in enumerate(zip_longest(loader, loader_torch)):
            print(i, data, data_torch)
            assert (torch.tensor(data) == data_torch).all()

    def test_nekoloader(self):
        dataset = list(range(100))
        self.run_loader(dataset, 8, False, 0)
        self.run_loader(dataset, 8, True, 0)
        self.run_loader(dataset, 8, False, 4)
        self.run_loader(dataset, 8, True, 4)
        self.run_loader(dataset, 10, False, 4)
        self.run_loader(dataset, 10, True, 4)
        self.run_loader(dataset, 11, False, 4)
        self.run_loader(dataset, 11, True, 4)
        # self.run_loader(dataset, 4, False, 6)