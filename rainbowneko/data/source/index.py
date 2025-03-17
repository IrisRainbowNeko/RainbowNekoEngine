from typing import Any, Dict, Tuple

from PIL import Image

from .base import VisionDataSource


class IndexSource(VisionDataSource):
    def __init__(self, data, repeat=1, **kwargs):
        super().__init__('', repeat=repeat)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Dict[str, Any]:
        image, label = self.data[index]
        return {
            'id': index,
            'image': image,
            'label': label
        }

    def get_image_size(self, data: Dict[str, Any]) -> Tuple[int, int]:
        image = data['image']
        if isinstance(image, Image.Image):
            return image.size
        else:
            return image.shape[1], image.shape[0]
