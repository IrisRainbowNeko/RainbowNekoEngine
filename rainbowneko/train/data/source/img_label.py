import os
from typing import Union, Dict, Any

from rainbowneko.train.data.label_loader import BaseLabelLoader, auto_label_loader
from rainbowneko.utils.img_size_tool import types_support
from rainbowneko.utils.utils import get_file_ext

from .base import VisionDataSource


class ImageLabelSource(VisionDataSource):
    def __init__(self, img_root, label_file, repeat=1, **kwargs):
        super(ImageLabelSource, self).__init__(img_root, repeat=repeat)

        self.label_dict = self._load_label_data(label_file)
        self.img_ids = self._load_img_ids(self.label_dict)

    def _load_img_ids(self, label_dict):
        return [x for x in label_dict.keys() if get_file_ext(x) in types_support] * self.repeat

    def _load_label_data(self, label_file: Union[str, BaseLabelLoader]):
        if label_file is None:
            return {}
        elif isinstance(label_file, str):
            return auto_label_loader(label_file).load()
        else:
            return label_file.load()

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index) -> Dict[str, Any]:
        path = os.path.join(self.img_root, self.img_ids[index])
        return {
            'image': path,
            'label': self.label_dict.get(self.img_ids[index], None)
        }
