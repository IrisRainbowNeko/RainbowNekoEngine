import os
from typing import List, Tuple

from rainbowneko.utils.img_size_tool import types_support
from rainbowneko.utils.utils import get_file_name, get_file_ext
from .base import VisionDataSource
from addict import Dict


class ImageFolderClassSource(VisionDataSource):
    def __init__(self, img_root, image_transforms=None, bg_color=(255, 255, 255), repeat=1, use_cls_index=True,
                 **kwargs):
        super(ImageFolderClassSource, self).__init__(img_root, image_transforms=image_transforms, bg_color=bg_color,
                                                     repeat=repeat, **kwargs)

        self.use_cls_index = use_cls_index
        self.img_paths, self.label_dict = self._load_img_label(img_root)

    def get_path(self, index: int) -> str:
        return self.img_paths[index]

    def __len__(self):
        return len(self.img_paths)

    def _load_img_label(self, img_root):
        sub_folders = [os.path.join(img_root, x) for x in os.listdir(img_root)]
        class_imgs = []
        label_dict = Dict()
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
                        label_dict[class_name][get_file_name(x)] = cls_id_dict[class_name]
                    else:
                        label_dict[class_name][get_file_name(x)] = class_name
            if isinstance(self.repeat, int):
                class_imgs.extend(imgs * self.repeat)
            else:
                class_imgs.extend(imgs * self.repeat[class_name])
        if self.use_cls_index:  # record class name to index map
            self.cls_id_dict = cls_id_dict
        return class_imgs, label_dict

    def get_image_list(self) -> List[Tuple[str, "ImageFolderClassSource"]]:
        sub_folders = [os.path.join(self.img_root, x) for x in os.listdir(self.img_root)]
        class_imgs = []
        for class_folder in sub_folders:
            class_name = os.path.basename(class_folder)
            imgs = [(os.path.join(class_folder, x), self) for x in os.listdir(class_folder) if
                    get_file_ext(x) in types_support]
            class_imgs.extend(imgs * self.repeat[class_name])
        return class_imgs

    def get_class_name(self, path: str) -> str:
        img_root, img_name = os.path.split(path)
        img_root, class_name = os.path.split(img_root)
        return class_name

    def load_label(self, path: str) -> str:
        img_root, img_name = os.path.split(path)
        img_root, class_name = os.path.split(img_root)

        label = self.label_dict[class_name].get(img_name, None)
        label = self.process_label({'label': label})
        return label
