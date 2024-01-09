from typing import Dict, Any, List

class BaseEvaluator:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def evaluate(self, pred, target):
        raise NotImplementedError

class EvaluatorGroup:
    def __init__(self, evaluator_list: List[BaseEvaluator]):
        self.evaluator_list = evaluator_list

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        for evaluator in self.evaluator_list:
            evaluator.evaluate(*args, **kwargs)