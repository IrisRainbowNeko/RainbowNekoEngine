from unittest import TestCase
from rainbowneko.parser import PythonCfgParser, ConfigCodeReconstructor
import inspect
from omegaconf import OmegaConf

class PyCFGTester(TestCase):
    def test_file_parse(self):
        parser = PythonCfgParser()
        from cfgs.py.train.classify import multi_class
        source = inspect.getsource(multi_class.make_cfg)
        parser.print_code(source)

    def test_cfg_load(self):
        parser = PythonCfgParser()
        cfg = parser.load_config('cfgs/py/train/classify/multi_class_mixup.py')
        cfg = OmegaConf.to_yaml(cfg)
        print(cfg)

    def test_cfg_unparse(self):
        parser = PythonCfgParser()
        cfg = parser.load_config('cfgs/py/train/classify/multi_class_mixup.py')
        coder = ConfigCodeReconstructor()
        cfg_code = coder.generate_code(cfg)
        print(cfg_code)