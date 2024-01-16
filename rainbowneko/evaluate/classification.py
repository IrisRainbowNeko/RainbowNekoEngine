from typing import Dict, Any

from .base import EvaluatorContainer


class ClsEvaluatorContainer(EvaluatorContainer):

    def evaluate(self, pred: Dict[str, Any], target: Dict[str, Any]):
        return self.evaluator(pred['pred'], target['label'])
