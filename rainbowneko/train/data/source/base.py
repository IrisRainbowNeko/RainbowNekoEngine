import bisect
import os
from typing import Dict, List, Tuple, Any

import albumentations as A
import numpy as np
from PIL import Image
from rainbowneko.utils.img_size_tool import get_image_size


class DataSource:
    def __init__(self, repeat=1, **kwargs):
        self.repeat = repeat

    def get_data_id(self, index: int) -> str:
        raise NotImplementedError()

    def __getitem__(self, index) -> Tuple[str, "DataSource"]:
        return self.get_data_id(index), self

    def __len__(self):
        raise NotImplementedError()

    def procees_image(self, image):
        raise NotImplementedError()

    def process_label(self, label):
        raise NotImplementedError()

    def load_image(self, path) -> Dict[str, Any]:
        raise NotImplementedError()

    def load_label(self, img_name: str) -> Dict[str, Any]:
        raise NotImplementedError()



class ComposeDataSource(DataSource):
    def __init__(self, source_list: List[DataSource]):
        self.source_list = source_list

        offsets = [0]
        for source in self.source_list:
            offsets.append(offsets[-1] + len(source))
        self._offsets = offsets

    def __getitem__(self, index) -> Tuple[str, "DataSource"]:
        if index < 0 or index >= len(self):
            raise IndexError('Index out of range')

        # 使用二分查找来找到正确的序列
        seq_index = bisect.bisect_right(self._offsets, index) - 1
        index_within_seq = index - self._offsets[seq_index]
        return self.source_list[seq_index][index_within_seq]

    def __len__(self):
        return self._offsets[-1]


class VisionDataSource(DataSource):
    def __init__(self, img_root, image_transforms=None, bg_color=(255, 255, 255), repeat=1, **kwargs):
        super(VisionDataSource, self).__init__(repeat=repeat)

        self.img_root = img_root
        self.image_transforms = image_transforms
        self.bg_color = tuple(bg_color)

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

    def load_image(self, rel_path) -> Dict[str, Any]:
        path = os.path.join(self.img_root, rel_path)
        image = Image.open(path)
        if image.mode == 'RGBA':
            x, y = image.size
            canvas = Image.new('RGBA', image.size, self.bg_color)
            canvas.paste(image, (0, 0, x, y), image)
            image = canvas
        return {'image': image.convert("RGB")}

    def get_image_size(self, path: str) -> Tuple[int, int]:
        return get_image_size(path)