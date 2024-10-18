import os
from typing import Any, Dict

from addict import Dict as ADict
from rainbowneko.utils.img_size_tool import types_support
from rainbowneko.utils.utils import get_file_ext

from .base import VisionDataSource


class ImageFolderClassSource(VisionDataSource):
    def __init__(self, img_root, repeat=1, use_cls_index=True, **kwargs):
        super(ImageFolderClassSource, self).__init__(img_root, repeat=repeat, **kwargs)

        self.use_cls_index = use_cls_index
        self.img_paths, self.label_dict = self._load_img_label(img_root)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index) -> Dict[str, Any]:
        path = self.img_paths[index]
        return {
            'image': path,
            'label': self.load_label(path)
        }

    def _load_img_label(self, img_root):
        sub_folders = [os.path.join(img_root, x) for x in os.listdir(img_root)]
        class_imgs = []
        label_dict = ADict()
        cls_id_dict = {}
        for class_folder in sub_folders:
            class_name = os.path.basename(class_folder)
            imgs = []
            for x in os.listdir(class_folder):
                if get_file_ext(x) in types_support:
                    imgs.append(os.path.join(class_folder, x))
                    if self.use_cls_index:
                        if class_name not in cls_id_dict:
                            cls_id_dict[class_name] = len(cls_id_dict)
                        label_dict[class_name][x] = cls_id_dict[class_name]
                    else:
                        label_dict[class_name][x] = class_name
            if isinstance(self.repeat, int):
                class_imgs.extend(imgs * self.repeat)
            else:
                class_imgs.extend(imgs * self.repeat[class_name])
        if self.use_cls_index:  # record class name to index map
            self.cls_id_dict = cls_id_dict
        return class_imgs, label_dict

    def load_label(self, path: str) -> Dict[str, Any]:
        img_root, img_name = os.path.split(path)
        img_root, class_name = os.path.split(img_root)

        label = self.label_dict[class_name].get(img_name, None)
        return label
