from unittest import TestCase

import torch

from rainbowneko.data import NekoDataLoader
from rainbowneko.data import (WebDataset, WebDatasetSource, FixedBucket, LoadImageHandler, HandlerChain, WebDSImageLabelSource,
                              get_sampler, BaseDataset, UnLabelSource)
from rainbowneko.data.source.webds import image_pipeline


class ClassifyWebdsTester(TestCase):
    def test_webdataset(self):
        dataset = WebDataset(
            shuffle=True,
            bucket=FixedBucket(),
            source=dict(
                source1=WebDatasetSource(image_pipeline('./webds_test.tar')),
            ),
            handler=HandlerChain(
                load=LoadImageHandler(),
                bucket=FixedBucket.handler,
                key_map_in=('jpg -> image', 'image_size -> image_size', 'id -> id')
            )
        )

        batch_size = 4
        dataset.build_bucket(bs=batch_size, world_size=1)

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=4,
            sampler=None,
            collate_fn=dataset.collate_fn,
            drop_last=False,
        )

        dataset.bucket.rest(0)

        for i, data in enumerate(loader):
            print(i, data)

    def test_webdataset_label(self):
        dataset = WebDataset(
            shuffle=True,
            bucket=FixedBucket(),
            source=dict(
                source1=WebDSImageLabelSource(
                    pipeline=image_pipeline('./webds_test.tar'),
                    label_file='./webds_test.json',
                ),
            ),
            handler=HandlerChain(
                load=LoadImageHandler(),
                bucket=FixedBucket.handler,
            )
        )

        batch_size = 4
        dataset.build_bucket(bs=batch_size, world_size=1)

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=4,
            sampler=None,
            collate_fn=dataset.collate_fn,
            drop_last=False,
        )

        dataset.bucket.rest(0)

        for i, data in enumerate(loader):
            print(i, data)

    def test_webdataset_server(self):
        dataset = WebDataset(
            shuffle=True,
            data_server_mod=True,
            bucket=FixedBucket(),
            source=dict(
                source1=WebDatasetSource(image_pipeline('./webds_test.tar')),
            ),
            handler=HandlerChain(
                load=LoadImageHandler(),
                bucket=FixedBucket.handler,
                key_map_in=('jpg -> image', 'image_size -> image_size', 'id -> id')
            )
        )

        batch_size = 4
        dataset.build_bucket(bs=batch_size, world_size=1)

        loader = NekoDataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=4,
            sampler=None,
            collate_fn=dataset.collate_fn,
            drop_last=False,
        )

        dataset.bucket.rest(0)

        for i, data in enumerate(loader):
            print(i, data)

    def test_dataset(self):
        dataset = BaseDataset(
            shuffle=True,
            bucket=FixedBucket(),
            source=dict(
                source1=UnLabelSource(img_root=r'E:\dataset\test_ccip\22'),
            ),
            handler=HandlerChain(
                load=LoadImageHandler(),
                bucket=FixedBucket.handler,
            )
        )

        batch_size = 4
        dataset.build_bucket(bs=batch_size, world_size=1)

        sampler = get_sampler(False)(
            dataset,
            num_replicas=1,
            rank=0,
            shuffle=False,
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=4,
            sampler=sampler,
            collate_fn=dataset.collate_fn,
            drop_last=False,
        )

        dataset.bucket.rest(0)

        for i, data in enumerate(loader):
            print(i, data)
