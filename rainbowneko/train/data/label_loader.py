import glob
import json
import os
from typing import Dict

import yaml
from rainbowneko.utils.img_size_tool import types_support
from loguru import logger

class BaseLabelLoader:
    def __init__(self, path):
        self.path = path

    def _load(self):
        raise NotImplementedError

    def load(self):
        retval = self._load()
        logger.info(f'{len(retval)} record(s) loaded with {self.__class__.__name__}, from path {self.path!r}')
        return retval

    @staticmethod
    def clean_ext(captions: Dict[str, str]):
        ''' image.ext -> image '''

        def rm_ext(path):
            name, ext = os.path.splitext(path)
            if len(ext)>0 and ext[1:] in types_support:
                return name
            return path

        return {rm_ext(k):v for k, v in captions.items()}

class JsonLabelLoader(BaseLabelLoader):
    def _load(self):
        with open(self.path, 'r', encoding='utf-8') as f:
            return json.loads(f.read())

class YamlLabelLoader(BaseLabelLoader):
    def _load(self):
        with open(self.path, 'r', encoding='utf-8') as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)

class TXTLabelLoader(BaseLabelLoader):
    def _load(self):
        txt_files = glob.glob(os.path.join(self.path, '*.txt'))
        captions = {}
        for file in txt_files:
            with open(file, 'r', encoding='utf-8') as f:
                captions[os.path.basename(file)] = f.read().strip()
        return captions

def auto_label_loader(path):
    if os.path.isdir(path):
        json_files = glob.glob(os.path.join(path, '*.json'))
        if json_files:
            return JsonLabelLoader(json_files[0])

        yaml_files = [
            *glob.glob(os.path.join(path, '*.yaml')),
            *glob.glob(os.path.join(path, '*.yml')),
        ]
        if yaml_files:
            return YamlLabelLoader(yaml_files[0])

        txt_files = glob.glob(os.path.join(path, '*.txt'))
        if txt_files:
            return TXTLabelLoader(path)

        raise FileNotFoundError(f'Caption file not found in directory {path!r}.')

    elif os.path.isfile(path):
        _, ext = os.path.splitext(path)
        if ext == '.json':
            return JsonLabelLoader(path)
        elif ext in {'.yaml', '.yml'}:
            return YamlLabelLoader(path)
        else:
            raise FileNotFoundError(f'Unknown caption file {path!r}.')

    else:
        raise FileNotFoundError(f'Unknown caption file type {path!r}.')
