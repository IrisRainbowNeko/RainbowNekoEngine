from unittest import TestCase
from rainbowneko.infer.infer_workflow import run_workflow
import sys

class ClassifyInferTester(TestCase):
    def test_multi_class_py(self):
        argv = sys.argv
        sys.argv = sys.argv+['--cfg', 'cfgs/py/infer/multi_class.py']
        run_workflow()
        sys.argv = argv