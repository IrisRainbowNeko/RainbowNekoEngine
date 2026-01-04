import json
from io import BytesIO
from typing import Dict, Any

import webdataset as wds
import numpy as np
from PIL import Image
from rainbowneko.data.label_loader import BaseLabelLoader, auto_label_loader
from rainbowneko.utils import types_support
from webdataset import DataPipeline

from .base import DataSource


class WebDatasetSource(DataSource):
    def __init__(self, pipeline: DataPipeline, repeat=1, size=1 << 15, **kwargs):
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
        img_bytes = [v for k, v in data.items() if k.lower() in types_support][0]
        image = Image.open(BytesIO(img_bytes))

        return {
            'id': img_id,
            'image': image,
        }

    def get_image_size(self, data):
        return data['image'].size

def _decode_rgb_array(img_bytes: bytes, bg_color=(255, 255, 255)) -> np.ndarray:
    with Image.open(BytesIO(img_bytes)) as img:
        if img.mode == 'RGBA':
            x, y = img.size
            canvas = Image.new('RGBA', img.size, bg_color)
            canvas.paste(img, (0, 0, x, y), img)
            canvas = canvas.convert("RGB")     # force RGB and force decode
            arr = np.asarray(canvas, dtype=np.uint8)
            del canvas
        else:
            img = img.convert("RGB")     # force RGB and force decode
            arr = np.asarray(img, dtype=np.uint8)
    return arr  # no reference to img_bytes/img

class WebDSImageLabelSource(WebDatasetSource):
    '''
    data.tar:
        - {id1}.{jpg|png|webp|...}
        - {id2}.{jpg|png|webp|...}

    label:
        id1: label1
        id2: label2
    '''

    def __init__(self, pipeline: DataPipeline, label_file=None, repeat=1, size=1 << 15, **kwargs):
        super().__init__(pipeline, repeat, size=size, **kwargs)
        self.label_dict = self._load_label_data(label_file) if label_file else None

    def _load_label_data(self, label_file: str | BaseLabelLoader):
        if label_file is None:
            return {}
        elif isinstance(label_file, str):
            return auto_label_loader(label_file).load()
        else:
            return label_file.load()

    def parse_label(self, img_id, data):
        if self.label_dict is None:
            if 'txt' in data:
                return data['txt'].decode('utf8')
            elif 'json' in data:
                return json.loads(data['json'].decode('utf8'))
            else:
                return None
        else:
            return self.label_dict.get(img_id, None)

    def __next__(self):
        data = next(self.pipeline_iter)
        img_id = data['__key__']
        # img_bytes = [v for k, v in data.items() if k.lower() in types_support][0]
        # image = Image.open(BytesIO(img_bytes))
        img_bytes = next(v for k, v in data.items() if k.lower() in types_support)
        image = _decode_rgb_array(img_bytes)
        del img_bytes
        label = self.parse_label(img_id, data)

        return {
            'id': img_id,
            'image': image,
            'label': label,
        }

    def get_image_size(self, data):
        if isinstance(data['image'], Image.Image):
            return data['image'].size
        else:
            return data['image'].shape[1], data['image'].shape[0]

    def __len__(self):
        return self.size if self.label_dict is None else len(self.label_dict)


def image_pipeline(url, buffer_size=300):
    return wds.DataPipeline(
        wds.SimpleShardList(url),
        wds.split_by_node,
        wds.split_by_worker,
        wds.tarfile_to_samples(),
        wds.shuffle(buffer_size),
        # wds.decode("pil"),
    )
