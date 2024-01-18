from typing import Tuple, Any

import cv2


class BaseBucket:
    can_shuffle = True

    def __getitem__(self, idx):
        '''
        :return: (file name of image), (target image size)
        '''
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def build(self, bs: int, world_size: int, source: 'DataSource'):
        raise NotImplementedError()

    def rest(self, epoch):
        pass

    def crop_resize(self, image, size, mask_interp=cv2.INTER_CUBIC) -> Tuple[Any, Tuple]:
        return image, (*size, 0, 0, *size)
