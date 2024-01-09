"""
pair_dataset.py
====================
    :Name:        text-image pair dataset
    :Author:      Dong Ziyi
    :Affiliation: HCP Lab, SYSU
    :Created:     10/03/2023
    :Licence:     Apache-2.0
"""

from typing import Dict, Tuple

import torch
from torch.utils.data import Dataset

from .bucket import BaseBucket
from .source import DataSource, ComposeDataSource

class ImageLabelDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(self, bucket: BaseBucket = None, source: Dict[str, DataSource] = None, **kwargs):
        self.bucket: BaseBucket = bucket
        self.source = ComposeDataSource(source)

    def load_image(self, path: str, data_source: DataSource, size: Tuple[int]):
        image_dict = data_source.load_image(path)
        image = image_dict['image']

        data, crop_coord = self.bucket.crop_resize({"img":image}, size)
        image = data_source.procees_image(data['img'])  # resize to bucket size
        return {'img':image}

    def load_label(self, img_name: str, data_source: DataSource):
        label = data_source.load_label(img_name)
        return {'label':label}

    def __len__(self):
        return len(self.bucket)

    def __getitem__(self, index):
        (path, data_source), size = self.bucket[index]
        img_name = data_source.get_image_name(path)

        data = self.load_image(path, data_source, size)
        label = self.load_label(img_name, data_source)
        data.update(label)

        return data

    @staticmethod
    def collate_fn(batch):
        '''
        batch: [{img:tensor, ..., plugin_input:{...}}]
        '''
        has_plugin_input = 'plugin_input' in batch[0]
        if has_plugin_input:
            plugin_input = {k:[] for k in batch[0]['plugin_input'].keys()}

        datas = {k:[] for k in batch[0].keys() if k != 'plugin_input'}

        for data in batch:
            if has_plugin_input:
                for k, v in data.pop('plugin_input').items():
                    plugin_input[k].append(v)

            for k, v in data.items():
                datas[k].append(v)

        for k, v in datas.items():
            datas[k] = torch.stack(v)

        if has_plugin_input:
            datas['plugin_input'] = {k:torch.stack(v) for k, v in plugin_input.items()}

        return datas
