import os
from typing import Dict, Any

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image

from .base import DataHandler
from ..utils import resize_crop_fix, pad_crop_fix
from rainbowneko.utils import Path_Like


class LoadImageHandler(DataHandler):
    def __init__(self, bg_color=(255, 255, 255), mode='RGB', key_map_in=('image -> image',), key_map_out=('image -> image',)):
        super().__init__(key_map_in, key_map_out)
        self.bg_color = bg_color
        self.mode = mode

    def load_image(self, path) -> Image.Image:
        image = Image.open(path)
        if image.mode == 'RGBA':
            x, y = image.size
            canvas = Image.new('RGBA', image.size, self.bg_color)
            canvas.paste(image, (0, 0, x, y), image)
            image = canvas
        return image.convert(self.mode)

    def handle(self, image):
        if isinstance(image, Path_Like):
            image = self.load_image(image)
        elif isinstance(image, Image.Image):
            image = image
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            raise NotImplementedError(f'image with type {type(image)} not supported')
        return {'image': image}

class ImageHandler(DataHandler):
    def __init__(self, transform, bg_color=(255, 255, 255), key_map_in=('image -> image',), key_map_out=('image -> image',)):
        super().__init__(key_map_in, key_map_out)
        self.transform = transform
        self.bg_color = bg_color


    def load_image(self, path) -> Dict[str, Any]:
        path = os.path.join(path)
        image = Image.open(path)
        if image.mode == 'RGBA':
            x, y = image.size
            canvas = Image.new('RGBA', image.size, self.bg_color)
            canvas.paste(image, (0, 0, x, y), image)
            image = canvas
        return image.convert("RGB")

    def procees_image(self, image):
        if isinstance(self.transform, (A.BaseCompose, A.BasicTransform)):
            image_A = self.transform(image=np.array(image))
            if isinstance(image_A['image'], np.ndarray):
                image = Image.fromarray(image_A['image'])
            else:
                image = image_A['image']
        else:
            image = self.transform(image)
        return image

    def handle(self, image):
        if isinstance(image, str):
            image = self.load_image(image)
        elif isinstance(image, (Image.Image, torch.Tensor)):
            image = image
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            raise NotImplementedError(f'image with type {type(image)} not supported')

        image = self.procees_image(image)
        return {'image': image}

class AutoSizeHandler(DataHandler):
    def __init__(self, mode='resize', key_map_in=('image -> image', 'image_size -> size'), key_map_out=('image -> image', 'coord -> coord')):
        '''
        
        :param mode: ['full', 'resize', 'pad']
        '''
        super().__init__(key_map_in, key_map_out)
        self.mode = mode

    def handle(self, image, size):
        if self.mode == 'full':
            w, h = image.size
            coord = [h, w, 0, 0, h, w]
        elif self.mode == 'resize':
            image, coord = resize_crop_fix({'image': image}, size)
            image = image['image']
        elif self.mode == 'pad':
            image, coord = pad_crop_fix({'image': image}, size)
            image = image['image']
        else:
            raise NotImplementedError(f'mode {self.mode} not supported')
        coord = torch.tensor(coord, dtype=torch.float)
        return {'image': image, 'coord': coord}


