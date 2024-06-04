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

class ImageLabelDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(self, bucket: BaseBucket = None, source: Dict[str, DataSource] = None, batch_transform: Callable=None, **kwargs):
        self.bucket: BaseBucket = bucket
        self.source = ComposeDataSource(list(source.values()))
        self.batch_transform = batch_transform

        self.random_context = None

    def load_image(self, data_id: str, data_source: DataSource, size: Tuple[int]):
        image = data_source.load_image(data_id)

        with RandomContext() as self.random_context:
            data, crop_coord = self.bucket.crop_resize(image, size)
            image = data_source.procees_image(data)  # resize to bucket size
        return image

    def load_label(self, data_id: str, data_source: DataSource):
        label = data_source.load_label(data_id)
        label = data_source.process_label(label)
        return {'label': label}

    def batch_process(self, batch: Dict[str, Union[List, torch.Tensor]]):
        if self.batch_transform is not None:
            batch = self.batch_transform(batch)
        return batch

    def __len__(self):
        return len(self.bucket)

    def __getitem__(self, index):
        (data_id, data_source), size = self.bucket[index]

        data = self.load_image(data_id, data_source, size)
        label = self.load_label(data_id, data_source)
        label = self.bucket.process_label(index, label)
        data.update(label)

        return data

    def collate_fn(self, batch):
        '''
        batch: [{
            img:tensor,
            label:{
                label:?,
                bbox:?,
                mask:?,
                ...
            },
            ...,
            plugin_input:{...}
        }]
        '''
        datas = {k: [] for k in batch[0].keys()}

        has_plugin_input = 'plugin_input' in batch[0]
        if has_plugin_input:
            plugin_input = {k: [] for k in batch[0]['plugin_input'].keys()}
            del datas['plugin_input']
        label = {k: [] for k in batch[0]['label'].keys()}
        del datas['label']

        for data in batch:
            if has_plugin_input:
                for k, v in data.pop('plugin_input').items():
                    plugin_input[k].append(v)

            # collate label
            for k, v in data.pop('label').items():
                label[k].append(v)

            for k, v in data.items():
                datas[k].append(v)

        for k, v in datas.items():
            if isinstance(v[0], torch.Tensor):
                datas[k] = torch.stack(v)
            else:
                datas[k] = self.create_tensor(v)

        if has_plugin_input:
            datas['plugin_input'] = {k: torch.stack(v) for k, v in plugin_input.items()}
        datas['label'] = {k: (torch.stack(v) if isinstance(v[0], torch.Tensor) else ImageLabelDataset.create_tensor(v))
                          for k, v in label.items()}

        datas = self.batch_process(datas)

        return datas

    @staticmethod
    def create_tensor(data):
        if isinstance(data, list) and isinstance(data[0], np.ndarray):
            return torch.tensor(np.array(data))
        else:
            return torch.tensor(data)
