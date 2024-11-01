from typing import Dict

import torch
from tqdm.auto import tqdm

from rainbowneko.models.wrapper import BaseWrapper
from rainbowneko.train.data import DataGroup
from .metrics import BaseMetric, MetricGroup


class Evaluator:
    def __init__(self, trainer: "Trainer", data_loader_group: DataGroup, metric: BaseMetric, interval=100):
        self.data_loader_group = data_loader_group
        self.trainer = trainer
        self.metric = metric
        self.interval = interval

    def to_dev(self, x):
        if isinstance(x, torch.Tensor):
            if torch.is_floating_point(x):
                return x.to(self.trainer.device, dtype=self.trainer.weight_dtype)
            else:
                return x.to(self.trainer.device)
        else:
            return x

    def forward_one_step(self, ds_name, model, data):

        input_datas = {k: self.to_dev(v) for k, v in data.items() if k != "plugin_input"}
        if "plugin_input" in data:
            input_datas["plugin_input"] = {k: self.to_dev(v) for k, v in data["plugin_input"].items()}

        model_pred = model(ds_name, **input_datas)

        return model_pred, input_datas

    @torch.inference_mode()
    def evaluate(self, step:int, model: BaseWrapper):
        if step % self.interval != 0:
            return

        model.eval()
        self.metric.reset()

        for data_dict in tqdm(self.data_loader_group, disable=not self.trainer.is_local_main_process):
            for ds_name, data in data_dict.items():
                pred, input_datas = self.forward_one_step(ds_name, model, data)
                self.metric.update(pred, input_datas)

        v_metric = self.metric.finish(self.trainer.accelerator.gather, self.trainer.is_local_main_process)

        if not isinstance(v_metric, dict):
            v_metric = {'metric': v_metric}

        data_size = {name:len(loader.dataset) for name, loader in self.data_loader_group.loader_dict.items()}
        self.trainer.loggers.info(f'Evaluate: data size {data_size}')
        log_data = {
            "eval/Step": {
                "format": "{}",
                "data": [self.trainer.global_step],
            }
        }
        log_data.update(MetricGroup.format(v_metric, prefix='eval/'))
        self.trainer.loggers.log(log_data, self.trainer.global_step)

    def to(self, device):
        self.metric.to(device)

class EvaluatorGroup:
    def __init__(self, loggers, evaluator_dict:Dict[str, Evaluator]):
        self.loggers = loggers
        self.evaluator_dict = evaluator_dict

    def evaluate(self, step:int, model: BaseWrapper):
        for name, evaluator in self.evaluator_dict.items():
            self.loggers.info(f'Evaluator {name}:')
            evaluator.evaluate(step, model)

    def to(self, device):
        for evaluator in self.evaluator_dict.values():
            evaluator.to(device)
