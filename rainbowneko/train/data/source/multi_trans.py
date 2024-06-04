from .img_label import ImageLabelSource
from typing import List, Callable
import albumentations as A
import numpy as np

class MultiTransSource(ImageLabelSource):
    def __init__(self, img_root, label_file, image_transforms:List[Callable], bg_color=(255, 255, 255), repeat=1,
                 trans_keys=('weak', 'strong'), **kwargs):
        super().__init__(img_root, label_file, image_transforms=image_transforms, bg_color=bg_color,
                                               repeat=repeat, **kwargs)
        self.trans_keys = trans_keys


    def procees_image(self, image_dict):
        image = image_dict.pop('image')
        for i, transform in enumerate(self.image_transforms):
            if isinstance(transform, (A.BaseCompose, A.BasicTransform)):
                image_A = transform(image=np.array(image))
                image_dict[f'image_{self.trans_keys[i]}'] = image_A['image']
            else:
                image_dict[f'image_{self.trans_keys[i]}'] = transform(image)
        return image_dict