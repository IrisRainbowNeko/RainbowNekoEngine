from functools import partial
from typing import Dict

import torch


class BaseEvaluator:
    def __init__(self):
        pass

    def reset(self):
        raise NotImplementedError

    def update(self, pred, target):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def to(self, device):
        pass


class EvaluatorContainer(BaseEvaluator):
    def __init__(self, evaluator):
        super().__init__()
        self.evaluator = evaluator

    def reset(self):
        self.metric_list = []

    def evaluate(self):
        try:
            return torch.cat(self.metric_list).mean()
        except:
            return torch.tensor(self.metric_list).mean()

    def to(self, device):
        self.evaluator.to(device)


class EvaluatorGroup:
    def __init__(self, evaluator_dict: Dict[str, BaseEvaluator]):
        self.evaluator_dict = {k: (v() if isinstance(v, partial) else v) for k, v in evaluator_dict.items()}

    def reset(self):
        for name, evaluator in self.evaluator_dict.items():
            evaluator.reset()

    def update(self, *args, **kwargs):
        for name, evaluator in self.evaluator_dict.items():
            evaluator.update(*args, **kwargs)

    def evaluate(self):
        metric_dict = {}
        for name, evaluator in self.evaluator_dict.items():
            metric_dict[name] = evaluator.evaluate()
        return metric_dict

    def to(self, device):
        for evaluator in self.evaluator_dict.values():
            evaluator.to(device)

    @staticmethod
    def format(metrics_dict, format="{:.2e}", prefix=''):
        if not isinstance(metrics_dict, dict):
            metrics_dict = {"metrics": metrics_dict}
        metrics_dict = {prefix+k: {"format": format, "data": [v]} for k, v in metrics_dict.items()}
        return metrics_dict
