import os
from typing import Dict, Any

from rainbowneko.utils.utils import is_image_file

from .base import VisionDataSource


class UnLabelSource(VisionDataSource):
    def __init__(self, img_root, repeat=1, **kwargs):
        super(UnLabelSource, self).__init__(img_root, repeat=repeat)

        self.img_ids = self._load_img_ids(img_root)

    def _load_img_ids(self, img_root):
        return [x for x in os.listdir(img_root) if is_image_file(x)] * self.repeat

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index) -> Dict[str, Any]:
        img_id = self.img_ids[index]
        path = os.path.join(self.img_root, img_id)
        return {
            'id': img_id,
            'image': path,
        }
