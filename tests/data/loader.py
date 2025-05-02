from unittest import TestCase

import torch
from torch.utils.data import IterableDataset
from rainbowneko.data import NekoDataLoader
from itertools import zip_longest

class TestDataset(IterableDataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __iter__(self):
        for item in self.data:
            yield item

    def __len__(self):
        return len(self.data)

class DataLoaderTester(TestCase):
    def run_loader(self, dataset, batch_size, drop_last, num_workers, split_iter_worker=False):
        loader = NekoDataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=None,
            drop_last=drop_last,
            split_iter_worker=split_iter_worker,
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

    def run_loader_iter(self, dataset, batch_size, drop_last, num_workers, split_iter_worker=False):
        loader = NekoDataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=None,
            drop_last=drop_last,
            split_iter_worker=split_iter_worker,
        )

        data_iter = iter(dataset)
        for i, data in enumerate(loader):
            gt = [next(data_iter) for _ in range(min(batch_size, len(dataset)-i*batch_size))]
            print(i, data, gt)
            assert (data == gt)

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

    def test_nekoloader_iter(self):
        dataset = list(range(100))
        dataset_iter = TestDataset(dataset)
        self.run_loader_iter(dataset_iter, 8, False, 0, split_iter_worker=True)
        self.run_loader_iter(dataset_iter, 8, True, 0, split_iter_worker=True)
        self.run_loader_iter(dataset_iter, 8, False, 4, split_iter_worker=True)
        self.run_loader_iter(dataset_iter, 8, True, 4, split_iter_worker=True)
        self.run_loader_iter(dataset_iter, 10, False, 4, split_iter_worker=True)
        self.run_loader_iter(dataset_iter, 10, True, 4, split_iter_worker=True)
        self.run_loader_iter(dataset_iter, 11, False, 4, split_iter_worker=True)
        self.run_loader_iter(dataset_iter, 11, True, 4, split_iter_worker=True)