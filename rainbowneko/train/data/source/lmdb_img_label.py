from io import BytesIO
from typing import List, Tuple, Dict, Any

import lmdb
from PIL import Image

from .base import DataSource
from .img_label import ImageLabelSource


class LMDBImageLabelSource(ImageLabelSource):
    def __init__(self, img_root, label_file, repeat=1, **kwargs):
        super().__init__(img_root, label_file, repeat, **kwargs)

        self.env = lmdb.open(img_root, readonly=True, lock=False)  # 打开 LMDB 数据库

    def __getitem__(self, index) -> Dict[str, Any]:
        with self.env.begin(write=False) as txn:
            img_data = txn.get(self.img_ids[index].encode())  # 从 LMDB 获取图像数据
        image = Image.open(BytesIO(img_data))
        return {
            'image': image,
            'label': self.label_dict.get(self.img_ids[index], None)
        }