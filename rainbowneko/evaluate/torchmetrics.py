from .metrics import MetricContainer
import torch

class TorchMetricsContainer(MetricContainer):

    def update(self, pred, target):
        args, kwargs = self.key_mapper(pred=pred, target=target)
        self.metric.update(*args, **kwargs)

    def finish(self, gather, is_local_main_process):
        total = torch.tensor([self.metric.total]).float()
        metric_all = torch.tensor([self.metric.compute()]).float()

        metric_all = gather(metric_all)
        total = gather(total)
        return metric_all*total/total.sum()