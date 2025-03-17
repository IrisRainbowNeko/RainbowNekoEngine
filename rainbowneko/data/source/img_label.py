from typing import Union, Dict, Any

from rainbowneko.data.label_loader import BaseLabelLoader, auto_label_loader
from rainbowneko.utils import is_image_file, Path_Like

from .base import VisionDataSource


class ImageLabelSource(VisionDataSource):
    def __init__(self, img_root: Path_Like, label_file, repeat=1, **kwargs):
        super().__init__(img_root, repeat=repeat)

        self.label_dict = self._load_label_data(label_file)
        self.img_ids = self._load_img_ids(self.label_dict)

    def _load_img_ids(self, label_dict):
        return [x for x in label_dict.keys() if is_image_file(x)] * self.repeat

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
        img_id = self.img_ids[index]
        path = self.img_root / img_id
        return {
            'id': img_id,
            'image': path,
            'label': self.label_dict.get(self.img_ids[index], None)
        }
