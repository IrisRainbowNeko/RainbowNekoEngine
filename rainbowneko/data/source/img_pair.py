from typing import Dict, Any

from .img_label import ImageLabelSource


class ImagePairSource(ImageLabelSource):
    def __init__(self, img_root, label_file, repeat=1, **kwargs):
        super().__init__(img_root=img_root, label_file=label_file, repeat=repeat)

    def __getitem__(self, index) -> Dict[str, Any]:
        img_id = self.img_ids[index]
        path = self.img_root / img_id
        label_img_name = self.label_dict.get(img_id, None)
        return {
            'id': img_id,
            'image': path,
            'label': self.img_root / label_img_name
        }
