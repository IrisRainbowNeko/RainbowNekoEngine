from functools import partial
from typing import Dict

import torch
from rainbowneko.utils import addto_dictlist, KeyMapper

class BaseMetric:
    def __init__(self):
        pass

    def reset(self):
        raise NotImplementedError

    def update(self, pred, target):
        raise NotImplementedError

    def finish(self, gather, is_local_main_process):
        raise NotImplementedError

    def to(self, device):
        pass


class MetricContainer(BaseMetric):
    def __init__(self, metric, device='cpu', key_map=None):
        super().__init__()
        self.key_mapper = KeyMapper(metric, key_map)
        self.metric = metric
        self.device = device

    def reset(self):
        self.metric_list = []

    def update(self, pred, inputs):
        args, kwargs = self.key_mapper(pred=pred, inputs=inputs)
        v_metric = self.metric(*args, **kwargs)
        self.metric_list.append(v_metric)

    def finish(self, gather, is_local_main_process):
        total = torch.tensor([len(self.metric_list)]).float()
        try:
            metric_all = torch.cat(self.metric_list).mean()
        except:
            metric_all = torch.tensor(self.metric_list).mean()

        metric_all = metric_all.to(self.device)
        total = total.to(self.device)
        if gather is not None:
            metric_all = gather(metric_all)
            total = gather(total)
        return (metric_all*total/total.sum()).sum().item()

    def to(self, device):
        self.device = device
        if hasattr(self.metric, 'to'):
            self.metric.to(device)

class FullMetricContainer(MetricContainer):

    def reset(self):
        self.args_list = []
        self.kwargs_list = {}

    def update(self, pred, inputs):
        args, kwargs = self.key_mapper(pred=pred, inputs=inputs)

        if len(self.args_list) == 0:
            for v in args:
                self.args_list.append([v])
        else:
            for i,v in enumerate(args):
                self.args_list[i].append(v)

        addto_dictlist(self.kwargs_list, kwargs)

    def finish(self, gather, is_local_main_process):
        for i, v in enumerate(self.args_list):
            self.args_list[i] = torch.cat(v)

        for k, v in self.kwargs_list.items():
            self.kwargs_list[k] = torch.cat(v)

        self.args_list = gather(self.args_list)
        self.kwargs_list = gather(self.kwargs_list)

        v_metric = self.metric(*self.args_list, **self.kwargs_list)

        return v_metric.item()


class MetricGroup(BaseMetric):
    def __init__(self, **metric_dict: BaseMetric):
        self.metric_dict = {k: (v() if isinstance(v, partial) else v) for k, v in metric_dict.items()}

    def reset(self):
        for name, metric in self.metric_dict.items():
            metric.reset()

    def update(self, *args, **kwargs):
        for name, metric in self.metric_dict.items():
            metric.update(*args, **kwargs)

    def finish(self, gather, is_local_main_process):
        metric_dict = {}
        for name, metric in self.metric_dict.items():
            metric_dict[name] = metric.finish(gather, is_local_main_process)
        return metric_dict

    def to(self, device):
        for metric in self.metric_dict.values():
            metric.to(device)

    @staticmethod
    def format(metrics_dict, format="{:.2e}", prefix=''):
        if not isinstance(metrics_dict, dict):
            metrics_dict = {"metrics": metrics_dict}
        metrics_dict = {prefix+k: {"format": format, "data": [v]} for k, v in metrics_dict.items()}
        return metrics_dict
