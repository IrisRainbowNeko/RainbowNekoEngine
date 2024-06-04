from typing import Any, Dict, Tuple

import albumentations as A
import numpy as np
from PIL import Image

from .base import DataSource


class IndexSource(DataSource):
    def __init__(self, data, image_transforms=None, bg_color=(255, 255, 255), repeat=1, **kwargs):
        super(IndexSource, self).__init__(repeat=repeat)
        self.data = data

        self.label_dict = {}
        self.image_transforms = image_transforms
        self.bg_color = tuple(bg_color)

    def get_data_id(self, index: int) -> str:
        return index

    def __len__(self):
        return len(self.data)

    def check_image(self, image):
        image = dict(**image)
        image.pop('image')
        if len(image) > 0:
            keys = ', '.join(image.keys())
            print(f'images of {keys} image cannot proceesed by {self.__class__.__name__}')

    def procees_image(self, image):
        self.check_image(image)
        if isinstance(self.image_transforms, (A.BaseCompose, A.BasicTransform)):
            image_A = self.image_transforms(image=np.array(image['image']))
            image.update(image_A)
        else:
            image['image'] = self.image_transforms(image['image'])
        return image

    def process_label(self, label_dict):
        return label_dict

    def load_image(self, idx: int) -> Dict[str, Any]:
        image, label = self.data[idx]
        if image.mode == 'RGBA':
            x, y = image.size
            canvas = Image.new('RGBA', image.size, self.bg_color)
            canvas.paste(image, (0, 0, x, y), image)
            image = canvas.convert("RGB")

        # cache label
        if idx not in self.label_dict:
            self.label_dict[idx] = label
        return {'image': image}

    def load_label(self, idx: int) -> Dict[str, Any]:
        if idx in self.label_dict:
            label = self.label_dict[idx]
        else:
            label = self.data[idx][1]

        return {'label': label}

    def get_image_size(self, idx: int) -> Tuple[int, int]:
        image, label = self.data[idx]
        if isinstance(image, Image.Image):
            return image.size
        else:
            return image.shape[1], image.shape[0]
