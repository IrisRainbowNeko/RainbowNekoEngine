from unittest import TestCase
from rainbowneko.train.trainer.trainer_ac_single import neko_train
import sys

class PyTrainTester(TestCase):
    def test_multi_class_py(self):
        argv = sys.argv
        sys.argv = sys.argv+['--cfg', 'cfgs/py/train/classify/multi_class.py', 'train.train_steps=100', 'train.train_epochs=null']
        neko_train()
        sys.argv = argv

    def test_multi_class_workflow_eval(self):
        argv = sys.argv
        sys.argv = sys.argv+['--cfg', 'cfgs/py/train/classify/multi_class_floweval.py', 'train.train_steps=100', 'train.train_epochs=null']
        neko_train()
        sys.argv = argv

    def test_multi_class_onnx(self):
        argv = sys.argv
        sys.argv = sys.argv+['--cfg', 'cfgs/py/train/classify/multi_class_onnx.py', 'train.train_steps=40', 'train.train_epochs=null']
        neko_train()
        sys.argv = argv

    def test_multi_class_py_acc(self):
        argv = sys.argv
        sys.argv = sys.argv+['--cfg', 'cfgs/py/train/classify/multi_class.py', 'train.train_steps=40', 'train.train_epochs=null',
                             'train.gradient_accumulation_steps=4']
        neko_train()
        sys.argv = argv

    def test_multi_class_yaml(self):
        argv = sys.argv
        sys.argv = sys.argv+['--cfg', 'cfgs/yaml/train/classify/multi_class.yaml', 'train.train_steps=40', 'train.train_epochs=null']
        neko_train()
        sys.argv = argv

    def test_semi_py(self):
        argv = sys.argv
        sys.argv = sys.argv+['--cfg', 'cfgs/py/train/classify/semi_supervise.py', 'train.train_steps=40', 'train.train_epochs=null']
        neko_train()
        sys.argv = argv

    def test_wd_scheduler_py(self):
        argv = sys.argv
        sys.argv = sys.argv+['--cfg', 'cfgs/py/train/classify/multi_class_dynamic_wd.py', 'train.train_steps=40', 'train.train_epochs=null']
        neko_train()
        sys.argv = argv