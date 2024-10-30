import os
from typing import Union, Dict, Any

from rainbowneko.train.data.label_loader import BaseLabelLoader, auto_label_loader
from rainbowneko.utils.img_size_tool import types_support
from rainbowneko.utils.utils import get_file_ext

from .base import VisionDataSource


class UnLabelSource(VisionDataSource):
    def __init__(self, img_root, repeat=1, **kwargs):
        super(UnLabelSource, self).__init__(img_root, repeat=repeat)

        self.img_ids = self._load_img_ids(img_root)

    def _load_img_ids(self, img_root):
        return [x for x in os.listdir(img_root) if get_file_ext(x) in types_support] * self.repeat

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index) -> Dict[str, Any]:
        path = os.path.join(self.img_root, self.img_ids[index])
        return {
            'image': path,
        }
