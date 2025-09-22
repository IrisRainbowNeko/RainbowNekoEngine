import glob
import json
import os
from pathlib import Path
from typing import Dict

import pandas as pd
import yaml
from loguru import logger
from rainbowneko.utils.img_size_tool import types_support


class BaseLabelLoader:
    def __init__(self, path: str, **kwargs):
        self.path = Path(path)

    def _load(self):
        raise NotImplementedError

    def load(self):
        ''' {image.ext: label} '''
        retval = self._load()
        logger.info(f'{len(retval)} record(s) loaded with {self.__class__.__name__}, from path {self.path!r}')
        return retval

    @staticmethod
    def clean_ext(captions: Dict[str, str]):
        ''' image.ext -> image '''

        def rm_ext(path):
            name, ext = os.path.splitext(path)
            if len(ext) > 0 and ext[1:] in types_support:
                return name
            return path

        return {rm_ext(k): v for k, v in captions.items()}


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
        # get all txt files and their related images
        img_map = {}
        txt_list = []
        for file in self.path.iterdir():
            file: Path
            if file.is_file():
                if file.suffix == '.txt':
                    txt_list.append(file)
                else:
                    img_map[file.stem] = file.name

        # load captions
        captions = {}
        for txt in txt_list:
            with open(txt, 'r', encoding='utf-8') as f:
                captions[img_map[txt.stem]] = f.read().strip()
        return captions


class ParquetLabelLoader(BaseLabelLoader):
    def __init__(self, path: str, index_column: str = 'id'):
        super().__init__(path)
        self.index_column = index_column

    def _load(self):
        df = pd.read_parquet(self.path)
        return df.set_index(self.index_column).to_dict(orient='index')


def auto_label_loader(path, **kwargs):
    if os.path.isdir(path):
        json_files = glob.glob(os.path.join(path, '*.json'))
        if json_files:
            return JsonLabelLoader(json_files[0], **kwargs)

        yaml_files = [
            *glob.glob(os.path.join(path, '*.yaml')),
            *glob.glob(os.path.join(path, '*.yml')),
        ]
        if yaml_files:
            return YamlLabelLoader(yaml_files[0], **kwargs)

        parquet_files = glob.glob(os.path.join(path, '*.parquet'))
        if parquet_files:
            return ParquetLabelLoader(parquet_files[0], **kwargs)

        txt_files = glob.glob(os.path.join(path, '*.txt'))
        if txt_files:
            return TXTLabelLoader(path, **kwargs)

        raise FileNotFoundError(f'Caption file not found in directory {path!r}.')

    elif os.path.isfile(path):
        _, ext = os.path.splitext(path)
        if ext == '.json':
            return JsonLabelLoader(path, **kwargs)
        elif ext in {'.yaml', '.yml'}:
            return YamlLabelLoader(path, **kwargs)
        elif ext in '.parquet':
            return ParquetLabelLoader(path, **kwargs)
        else:
            raise FileNotFoundError(f'Unknown caption file {path!r}.')

    else:
        raise FileNotFoundError(f'Unknown caption file type {path!r}.')
