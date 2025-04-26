from .base import BasicAction
from typing import Union, Dict
# from rainbowneko.evaluate import BaseMetric

class MetricAction(BasicAction):
    def __init__(self, metric: Union["BaseMetric", Dict[str, "BaseMetric"]], key_map_in=None, key_map_out=None):
        super().__init__(key_map_in=key_map_in, key_map_out=key_map_out)
        self.metric = metric

    def forward(self, pred, inputs, device, _metric=None, **states):
        if _metric is None:
            self.metric.to(device)
            self.metric.reset()
        self.metric.update(pred=pred, inputs=inputs)
        return {'_metric':self.metric}

# class MetricFinishAction(BasicAction):
#     def __init__(self, key_map_in=None, key_map_out=None):
#         super().__init__(key_map_in=key_map_in, key_map_out=key_map_out)
#
#     def forward(self, _metric, **states):
#         v_metric = _metric.finish(self.trainer.accelerator.gather, self.trainer.is_local_main_process)
#         return {'v_metric':_metric.finish()}