from types import ModuleType
from typing import Dict, Union

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from rainbowneko.infer import WorkflowRunner
from rainbowneko.models.wrapper import BaseWrapper
from rainbowneko.parser import load_config
from .metrics import BaseMetric, MetricGroup


class Evaluator:
    def __init__(self, trainer: "Trainer", data_loader: DataLoader, metric: BaseMetric, ds_name=None, interval=100, dataset:None=None):
        assert dataset is None, '"dataset" is a placeholder for cfg, should always be None.'
        self.data_loader = data_loader
        self.trainer = trainer
        self.metric = metric
        self.interval = interval
        self.ds_name = ds_name

        self.data_loader.dataset.bucket.rest(0)

        self.metric.to(self.device)

    @property
    def device(self):
        return self.trainer.device

    @property
    def dtype(self):
        return self.trainer.weight_dtype

    def forward_one_step(self, model, data):
        input_datas = {k: self.trainer.to_dev(v) for k, v in data.items() if k != "plugin_input"}
        if "plugin_input" in data:
            input_datas["plugin_input"] = {k: self.trainer.to_dev(v) for k, v in data["plugin_input"].items()}

        model_pred = model(self.ds_name, **input_datas)

        return model_pred, input_datas
    
    def cpu_gather(self, tensor):
        if self.trainer.world_size>1:
            if not hasattr(self, 'gloo_group'): # Transfer data on cpu
                self.gloo_group = dist.new_group(backend='gloo')

            world_size = dist.get_world_size()
            gathered_tensors = [torch.empty_like(tensor) for _ in range(world_size)]
            dist.all_gather(gathered_tensors, tensor, group=self.gloo_group)
            return torch.cat(gathered_tensors, dim=0)
        else:
            return tensor

    @torch.no_grad()
    def evaluate(self, step: int, model: BaseWrapper, prefix='eval/'):
        if step % self.interval != 0:
            return

        # record training layers
        training_layers = [layer for layer in model.modules() if layer.training]

        # reset metric
        model.eval()
        self.metric.reset()

        for data in tqdm(self.data_loader, disable=not self.trainer.is_local_main_process):
            pred, input_datas = self.forward_one_step(model, data)
            # update data to metric
            self.metric.update(pred, input_datas)

        v_metric = self.metric.finish(self.cpu_gather, self.trainer.is_local_main_process)
        if not isinstance(v_metric, dict):
            v_metric = {'metric': v_metric}

        data_size = len(self.data_loader.dataset)
        self.trainer.loggers.info(f'Evaluate: data size {data_size}')
        log_data = {
            "eval/Step": {
                "format": "{}",
                "data": [step],
            }
        }
        log_data.update(MetricGroup.format(v_metric, prefix=prefix))
        self.trainer.loggers.log(log_data, step, force=True)

        for layer in training_layers:
            layer.train()

    def to(self, device):
        self.metric.to(device)


class EvaluatorGroup:
    def __init__(self, loggers, evaluator_dict: Dict[str, Evaluator]):
        self.loggers = loggers
        self.evaluator_dict = evaluator_dict

    def evaluate(self, step: int, model: BaseWrapper):
        for name, evaluator in self.evaluator_dict.items():
            self.loggers.info(f'Evaluator {name}:')
            evaluator.evaluate(step, model, prefix=f'eval/{name}/')

    def to(self, device):
        for evaluator in self.evaluator_dict.values():
            evaluator.to(device)


class WorkflowEvaluator(Evaluator):
    def __init__(self, trainer: "Trainer", workflow: Union[str, ModuleType, Dict], ds_name=None, interval=100):
        self.trainer = trainer
        self.interval = interval
        self.ds_name = ds_name

        if isinstance(workflow, (ModuleType,str)):
            parser, conf = load_config(workflow)
            self.workflow_runner = WorkflowRunner(parser, conf)
        else:
            self.workflow_runner = WorkflowRunner(trainer.parser, workflow)

    @torch.no_grad()
    def evaluate(self, step: int, model: BaseWrapper, prefix='eval/'):
        if step % self.interval != 0:
            return

        # record training layers
        training_layers = [layer for layer in model.modules() if layer.training]

        model.eval()

        states = self.workflow_runner.run(model=model, in_preview=True, device=self.device, dtype=self.dtype,
                                          world_size=self.trainer.world_size, local_rank=self.trainer.local_rank)
        metric = states['_metric']

        v_metric = metric.finish(self.trainer.accelerator.gather, self.trainer.is_local_main_process)
        if not isinstance(v_metric, dict):
            v_metric = {'metric': v_metric}

        log_data = {
            "eval/Step": {
                "format": "{}",
                "data": [step],
            }
        }
        log_data.update(MetricGroup.format(v_metric, prefix=prefix))
        self.trainer.loggers.log(log_data, step, force=True)

        for layer in training_layers:
            layer.train()

    def to(self, device):
        pass