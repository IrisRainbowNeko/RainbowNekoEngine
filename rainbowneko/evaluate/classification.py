from .base import BaseEvaluator


class AccEvaluator(BaseEvaluator):

    def evaluate(self, pred, target):
        raise NotImplementedError