from typing import Tuple, Any, Dict

import cv2


class BaseBucket:
    can_shuffle = True

    def __getitem__(self, idx) -> Dict[str, Any]:
        '''
        :return: (file name of image), (target image size)
        '''
        return self.source[idx]

    def __len__(self):
        return len(self.source)

    def build(self, bs: int, world_size: int, source: 'DataSource'):
        self.source = source

    def rest(self, epoch):
        pass

    def process_label(self, idx:int, label):
        return label