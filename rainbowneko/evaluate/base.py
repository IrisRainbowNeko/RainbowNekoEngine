from typing import Dict
from functools import partial

class BaseEvaluator:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def evaluate(self, pred, target):
        raise NotImplementedError

    def to(self, device):
        pass

class EvaluatorContainer(BaseEvaluator):
    def __init__(self, evaluator):
        super().__init__()
        self.evaluator = evaluator

    def evaluate(self, pred, target):
        raise NotImplementedError

    def to(self, device):
        self.evaluator.to(device)

class EvaluatorGroup:
    def __init__(self, evaluator_dict: Dict[str, BaseEvaluator]):
        self.evaluator_dict = {k:(v() if isinstance(v, partial) else v) for k,v in evaluator_dict.items()}

    def __call__(self, *args, **kwargs):
        metric_dict = {}
        for name, evaluator in self.evaluator_dict.items():
            metric_dict[name] = evaluator(*args, **kwargs)
        return metric_dict

    def to(self, device):
        for evaluator in self.evaluator_dict.values():
            evaluator.to(device)

    @staticmethod
    def format(metrics_dict, format="{:.2e}"):
        if not isinstance(metrics_dict, dict):
            metrics_dict = {"metrics": metrics_dict}
        metrics_dict = {k: {"format": format, "data": [v]} for k, v in metrics_dict.items()}
        return metrics_dict