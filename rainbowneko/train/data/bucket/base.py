from typing import Tuple, Any

import cv2


class BaseBucket:
    can_shuffle = True

    def __getitem__(self, idx) -> Tuple[Tuple[str, 'DataSource'], Tuple[int, int]]:
        '''
        :return: (file name of image), (target image size)
        '''
        return self.source[idx], (0,0)

    def __len__(self):
        return len(self.source)

    def build(self, bs: int, world_size: int, source: 'DataSource'):
        self.source = source

    def rest(self, epoch):
        pass

    def crop_resize(self, image, size, mask_interp=cv2.INTER_CUBIC) -> Tuple[Any, Tuple]:
        return image, (*size, 0, 0, *size)

    def process_label(self, idx:int, label):
        return label