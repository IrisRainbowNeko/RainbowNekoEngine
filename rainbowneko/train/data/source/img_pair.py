import os
from typing import Dict, Any

from .img_label import ImageLabelSource


class ImagePairSource(ImageLabelSource):
    def __init__(self, img_root, label_file, repeat=1, **kwargs):
        super().__init__(img_root=img_root, label_file=label_file, repeat=repeat)

    def __getitem__(self, index) -> Dict[str, Any]:
        path = os.path.join(self.img_root, self.img_ids[index])
        label_img_name = self.label_dict.get(self.img_ids[index], None)
        return {
            'image': path,
            'label': os.path.join(self.img_root, label_img_name)
        }
