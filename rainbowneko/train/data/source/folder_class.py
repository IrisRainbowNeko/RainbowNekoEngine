from .img_label import ImageLabelSource
from typing import List, Tuple, Union
import os
from rainbowneko.utils.utils import get_file_name, get_file_ext
from rainbowneko.utils.img_size_tool import types_support

class ImageFolderClassSource(ImageLabelSource):
    def __init__(self, img_root, image_transforms=None, bg_color=(255, 255, 255), repeat=1, **kwargs):
        super(ImageFolderClassSource, self).__init__(img_root, None, image_transforms, bg_color=bg_color, repeat=repeat, **kwargs)

    def load_label_data(self, *args):
        sub_folders = [os.path.join(self.img_root, x) for x in os.listdir(self.img_root)]
        label_dict = {}
        for class_folder in sub_folders:
            class_name = os.path.basename(class_folder)
            for x in os.listdir(class_folder):
                if get_file_ext(x) in types_support:
                    label_dict[f'{class_name}/{get_file_name(x)}'] = class_name
        return label_dict

    def get_image_list(self) -> List[Tuple[str, "ImageFolderClassSource"]]:
        sub_folders = [os.path.join(self.img_root, x) for x in os.listdir(self.img_root)]
        class_imgs = []
        for class_folder in sub_folders:
            class_name = os.path.basename(class_folder)
            imgs = [(os.path.join(class_folder, x), self) for x in os.listdir(class_folder) if get_file_ext(x) in types_support]
            class_imgs.extend(imgs*self.repeat[class_name])
        return class_imgs


    def get_image_name(self, path: str) -> str:
        img_root, img_name = os.path.split(path)
        img_name = img_name.rsplit('.')[0]
        img_root, class_name = os.path.split(img_root)
        return f'{class_name}/{img_name}'
