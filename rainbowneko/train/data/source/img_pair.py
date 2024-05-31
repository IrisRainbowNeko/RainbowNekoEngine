from typing import Dict, Any

import albumentations as A
import numpy as np

from .img_label import ImageLabelSource


class ImagePairSource(ImageLabelSource):
    def __init__(self, img_root, label_file, image_transforms=None, target_transforms=None, bg_color=(255, 255, 255), repeat=1, **kwargs):
        super().__init__(img_root=img_root, label_file=label_file, image_transforms=image_transforms, bg_color=bg_color, repeat=repeat)
        self.target_transforms = target_transforms

    def load_label(self, img_id: str) -> Dict[str, Any]:
        target_img_path = self.label_dict.get(img_id, None)
        target_img = self.load_image(target_img_path)['image']
        return {'target_image': target_img}

    def process_label(self, image):
        if isinstance(self.target_transforms, (A.BaseCompose, A.BasicTransform)):
            image_A = self.target_transforms(image=np.array(image))
            image = image_A['image']
        else:
            image = self.target_transforms(image)
        return image
