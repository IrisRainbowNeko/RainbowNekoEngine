import os
from typing import Union

from rainbowneko.train.data.label_loader import BaseLabelLoader, auto_label_loader
from rainbowneko.utils.img_size_tool import types_support
from rainbowneko.utils.utils import get_file_ext
from .base import VisionDataSource


class ImageLabelSource(VisionDataSource):
    def __init__(self, img_root, label_file, image_transforms=None, bg_color=(255, 255, 255), repeat=1, **kwargs):
        super(ImageLabelSource, self).__init__(img_root, image_transforms=image_transforms, bg_color=bg_color,
                                               repeat=repeat)

        self.img_paths = self._load_img_paths(img_root)
        self.label_dict = self._load_label_data(label_file)

    def _load_img_paths(self, img_root):
        return [os.path.join(img_root, x) for x in os.listdir(img_root) if
                get_file_ext(x) in types_support] * self.repeat

    def _load_label_data(self, label_file: Union[str, BaseLabelLoader]):
        if label_file is None:
            return {}
        elif isinstance(label_file, str):
            return auto_label_loader(label_file).load()
        else:
            return label_file.load()

    def get_path(self, index: int) -> str:
        return self.img_paths[index]

    def __len__(self):
        return len(self.img_paths)

    def load_label(self, path: str) -> str:
        img_name = os.path.basename(path)
        label = self.label_dict.get(img_name, None)
        label = self.process_label({'label': label})
        return label
