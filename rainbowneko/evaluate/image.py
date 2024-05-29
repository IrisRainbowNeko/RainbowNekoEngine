from typing import Dict, Any

from .base import EvaluatorContainer


class ImageEvaluatorContainer(EvaluatorContainer):

    def update(self, pred: Dict[str, Any], target: Dict[str, Any]):
        if isinstance(pred['pred'], list):
            for pred_cudai, target_cudai in zip(pred['pred'], target['target_image']):
                self.metric_list.append(self.evaluator(pred_cudai, target_cudai))
        else:
            self.metric_list.append(self.evaluator(pred['pred'], target['target_image']))
