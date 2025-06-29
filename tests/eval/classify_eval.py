from unittest import TestCase
from rainbowneko.evaluate.eval_ac_single import neko_eval
import sys

class ClassifyEvalTester(TestCase):
    def test_eval_multi_class_py(self):
        argv = sys.argv
        sys.argv = sys.argv[:1]+['--cfg', 'cfgs/py/eval/multi_class.py']
        neko_eval()
        sys.argv = argv

    def test_workflow_eval_multi_class_py(self):
        argv = sys.argv
        sys.argv = sys.argv[:1]+['--cfg', 'cfgs/py/eval/multi_class_flow.py']
        neko_eval()
        sys.argv = argv