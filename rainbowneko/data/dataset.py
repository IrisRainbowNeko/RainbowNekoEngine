"""
pair_dataset.py
====================
    :Name:        text-image pair dataset
    :Author:      Dong Ziyi
    :Affiliation: HCP Lab, SYSU
    :Created:     10/03/2023
    :Licence:     Apache-2.0
"""

from typing import Dict, Tuple, Union, List, Callable

import torch
import numpy as np
from torch.utils.data import Dataset

from .bucket import BaseBucket
from .source import DataSource, ComposeDataSource
from rainbowneko.utils import RandomContext
from .handler import DataHandler

class BaseDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(self, bucket: BaseBucket = None, source: Dict[str, DataSource] = None, handler: DataHandler = None,
                 batch_handler: DataHandler=None, **kwargs):
        self.bucket: BaseBucket = bucket
        self.source = ComposeDataSource(list(source.values()))
        self.handler = handler
        self.batch_handler = batch_handler

    def build_bucket(self, bs, world_size):
        self.bucket.build(bs=bs, world_size=world_size, source=self.source)

    def batch_process(self, batch: Dict[str, Union[List, torch.Tensor]]):
        if self.batch_handler is not None:
            batch = self.batch_handler(batch)
        return batch

    def __len__(self):
        return len(self.bucket)

    def __getitem__(self, index):
        datas = self.bucket[index]
        datas = self.handler(datas)
        return datas

    def collate_fn(self, batch):
        '''
        batch: [{
            img:tensor,
            label:tensor,
            ...,
            plugin_input:{...}
        }]
        '''
        datas = {k: [] for k in batch[0].keys()}

        has_plugin_input = 'plugin_input' in batch[0]
        if has_plugin_input:
            plugin_input = {k: [] for k in batch[0]['plugin_input'].keys()}
            del datas['plugin_input']

        for data in batch:
            if has_plugin_input:
                for k, v in data.pop('plugin_input').items():
                    plugin_input[k].append(v)

            for k, v in data.items():
                datas[k].append(v)

        def batch_merge(data):
            for k, v in data.items():
                if isinstance(v[0], torch.Tensor):
                    data[k] = torch.stack(v)
                elif isinstance(v[0], dict):
                    pass
                else:
                    try:
                        data[k] = self.create_tensor(v)
                    except:
                        pass
            return data

        datas = batch_merge(datas)

        if has_plugin_input:
            datas['plugin_input'] = batch_merge(plugin_input)

        datas = self.batch_process(datas)

        return datas

    @staticmethod
    def create_tensor(data):
        if isinstance(data, list) and isinstance(data[0], np.ndarray):
            return torch.tensor(np.array(data))
        else:
            return torch.tensor(data)
