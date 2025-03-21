from pathlib import Path
from typing import Any, Dict, Union

from rainbowneko.data.label_loader import BaseLabelLoader
from rainbowneko.utils import Path_Like

from .img_label import ImageLabelSource


class ImageFolderClassSource(ImageLabelSource):
    def __init__(self, img_root, repeat=1, **kwargs):
        super().__init__(img_root, img_root, repeat=repeat, **kwargs)

    def __getitem__(self, index) -> Dict[str, Any]:
        img_id = self.img_ids[index]
        path = self.img_root / img_id
        return {
            'id': img_id,
            'image': path,
            'label': self.cls_id_dict[self.label_dict[self.img_ids[index]]]
        }

    def _load_label_data(self, img_root: Union[Path_Like, BaseLabelLoader]):
        ''' {class_name/image.ext: label} '''
        if img_root is None:
            label_dict = {}
        elif isinstance(img_root, Path_Like):
            label_dict = {}
            img_root = Path(img_root)
            for class_folder in img_root.iterdir():
                cls_name = class_folder.name
                for img_path in class_folder.iterdir():
                    label_dict[f'{cls_name}/{img_path.name}'] = cls_name
        else:
            label_dict = img_root.load()

        # label to cls_id
        self.cls_id_dict = {cls_name: cls_id for cls_id, cls_name in enumerate(sorted(set(label_dict.values())))}
        return label_dict
