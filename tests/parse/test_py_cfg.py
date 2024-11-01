from unittest import TestCase
from rainbowneko.parser import PythonCfgParser
import importlib
from omegaconf import OmegaConf

class PyCFGTester(TestCase):
    def test_file_parse(self):
        parser = PythonCfgParser()
        module = importlib.import_module('cfgs.py.train.classify.multi_class')

        code = parser.get_code(module.make_cfg)
        code_format = parser.transform_code(code)
        parser.print_code(code_format)

    def test_cfg_load(self):
        parser = PythonCfgParser()
        cfg = parser.load_config('cfgs/py/train/classify/multi_class_mixup.py')
        cfg = OmegaConf.to_yaml(cfg)
        print(cfg)