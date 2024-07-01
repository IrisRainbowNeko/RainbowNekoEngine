from rainbowneko.models.wrapper import BaseWrapper
from tqdm.auto import tqdm
from rainbowneko.utils import addto_dictlist
import torch
from .base import BaseMetric, MetricGroup
from rainbowneko.train.data import DataGroup
from typing import Dict

class Evaluator:
    def __init__(self, trainer: "Trainer", data_loader_group: DataGroup, metric: BaseMetric, interval=100):
        self.data_loader_group = data_loader_group
        self.trainer = trainer
        self.metric = metric
        self.interval = interval

    def forward_one_step(self, model, data):
        device = self.trainer.device
        weight_dtype = self.trainer.weight_dtype

        image = data.pop("image").to(device, dtype=weight_dtype)
        target = {k: v.to(device) for k, v in data.pop("label").items()}
        other_datas = {
            k: v.to(device, dtype=weight_dtype) for k, v in data.items() if k != "plugin_input"
        }
        if "plugin_input" in data:
            other_datas["plugin_input"] = {
                k: v.to(device, dtype=weight_dtype) for k, v in data["plugin_input"].items()
            }

        model_pred = model(image, **other_datas)

        return model_pred, target

    @torch.inference_mode()
    def evaluate(self, step:int, model: BaseWrapper):
        if step % self.interval != 0:
            return

        model.eval()
        self.metric.reset()

        for loader in self.data_loader_group.loader_dict.values():
            for data in tqdm(loader, disable=not self.trainer.is_local_main_process):
                pred, target = self.forward_one_step(model, data)
                self.metric.update(pred, target)

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
