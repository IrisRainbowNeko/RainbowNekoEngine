from typing import Dict, Any, Union
from rainbowneko.data.label_loader import BaseLabelLoader, auto_label_loader
from rainbowneko.utils import types_support
from webdataset import DataPipeline
import webdataset as wds

from .base import DataSource


class WebDatasetSource(DataSource):
    def __init__(self, pipeline: DataPipeline, repeat=1, size=1<<15, **kwargs):
        super().__init__(repeat, **kwargs)
        self.pipeline = pipeline
        self.size = size

    def __getitem__(self, index) -> Dict[str, Any]:
        raise NotImplementedError(f'{self.__class__.__name__} is not indexable')

    def __iter__(self):
        self.pipeline_iter = iter(self.pipeline)
        return self

    def __next__(self):
        data = next(self.pipeline_iter)
        return {
            'id': data['__key__'],
            **{k: v for k, v in data.items() if not k.startswith('__')}
        }

    def __len__(self):
        return self.size

class WebDatasetImageSource(WebDatasetSource):
    def __next__(self):
        data = next(self.pipeline_iter)
        img_id = data['__key__']
        image = [v for k, v in data.items() if k.lower() in types_support][0]

        return {
            'id': img_id,
            'image': image,
        }

class WebDSImageLabelSource(WebDatasetSource):
    '''
    data.tar:
        - {id1}.{jpg|png|webp|...}
        - {id2}.{jpg|png|webp|...}

    label:
        id1: label1
        id2: label2
    '''

    def __init__(self, pipeline: DataPipeline, label_file, repeat=1, **kwargs):
        super().__init__(pipeline, repeat, **kwargs)
        self.label_dict = self._load_label_data(label_file)

    def _load_label_data(self, label_file: str | BaseLabelLoader):
        if label_file is None:
            return {}
        elif isinstance(label_file, str):
            return auto_label_loader(label_file).load()
        else:
            return label_file.load()

    def __next__(self):
        data = next(self.pipeline_iter)
        img_id = data['__key__']
        image = [v for k, v in data.items() if k.lower() in types_support][0]

        return {
            'id': img_id,
            'image': image,
            'label': self.label_dict.get(img_id, None),
        }

    def __len__(self):
        return len(self.label_dict)


def image_pipeline(url, buffer_size=300):
    return wds.DataPipeline(
        wds.SimpleShardList(url),
        wds.split_by_node,
        wds.split_by_worker,
        wds.tarfile_to_samples(),
        wds.shuffle(buffer_size),
        wds.decode("pil"),
    )
