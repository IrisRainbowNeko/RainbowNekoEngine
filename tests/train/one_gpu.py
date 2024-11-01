from unittest import TestCase
from rainbowneko.train.trainer.trainer_ac_single import neko_train
import sys

class PyCFGTester(TestCase):
    def test_multi_class_py(self):
        argv = sys.argv
        sys.argv += ['--cfg', 'cfgs/py/train/classify/multi_class.py', 'train.train_steps=40', 'train.train_epochs=null']
        neko_train()
        sys.argv = argv

    def test_multi_class_yaml(self):
        argv = sys.argv
        sys.argv += ['--cfg', 'cfgs/yaml/train/classify/multi_class.yaml', 'train.train_steps=40', 'train.train_epochs=null']
        neko_train()
        sys.argv = argv