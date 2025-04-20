from unittest import TestCase
from rainbowneko.train.trainer.trainer_ac_single import neko_train
import sys

class PyTrainWebdsTester(TestCase):
    def test_webdataset_py(self):
        argv = sys.argv
        sys.argv = sys.argv+['--cfg', 'cfgs/py/train/classify/multi_class_webds.py', 'train.train_steps=100', 'train.train_epochs=null']
        neko_train()
        sys.argv = argv

