import os
from typing import Dict, List, Tuple
from typing import Union, Any

from PIL import Image

from rainbowneko.train.data.label_loader import BaseLabelLoader, auto_label_loader
from rainbowneko.utils.img_size_tool import types_support
from rainbowneko.utils.utils import get_file_ext
from .base import DataSource


class ImageLabelSource(DataSource):
    def __init__(self, img_root, label_file, image_transforms=None, bg_color=(255, 255, 255), repeat=1, **kwargs):
        super(ImageLabelSource, self).__init__(img_root, repeat=repeat)

        self.label_dict = self.load_label_data(label_file)
        self.image_transforms = image_transforms
        self.bg_color = tuple(bg_color)

    def load_label_data(self, label_file: Union[str, BaseLabelLoader]):
        if label_file is None:
            return {}
        elif isinstance(label_file, str):
            return auto_label_loader(label_file).load()
        else:
            return label_file.load()

    def get_image_list(self) -> List[Tuple[str, DataSource]]:
        imgs = [(os.path.join(self.img_root, x), self) for x in os.listdir(self.img_root) if
                get_file_ext(x) in types_support]
        return imgs * self.repeat

    def procees_image(self, image):
        return self.image_transforms(image)

    def process_label(self, label_dict):
        return label_dict

    def load_image(self, path) -> Dict[str, Any]:
        image = Image.open(path)
        if image.mode == 'RGBA':
            x, y = image.size
            canvas = Image.new('RGBA', image.size, self.bg_color)
            canvas.paste(image, (0, 0, x, y), image)
            image = canvas
        return {'image': image.convert("RGB")}

    def load_label(self, img_name: str) -> str:
        label = self.label_dict.get(img_name, None)
        label = self.process_label({'label': label})
        return label
