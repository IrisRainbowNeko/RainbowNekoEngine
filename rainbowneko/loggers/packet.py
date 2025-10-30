from dataclasses import dataclass
from typing import Union, List

import numpy as np
import torch
from PIL import Image

Scalar = Union[int, float, np.number, torch.Tensor]


class LogPacket:
    pass


@dataclass
class ScalarLog(LogPacket):
    value: List[Scalar]
    format: str = '{}'

    def __post_init__(self):
        if not isinstance(self.value, (list, tuple)):
            self.value = [self.value]


@dataclass
class ImageLog(LogPacket):
    image: Image.Image
    caption: str
    format: str = 'png'


@dataclass
class TextFileLog(LogPacket):
    text: str
    file_name: str
    html_template: str = '{}'
