import os
from functools import partial
from types import ModuleType
from typing import Dict

import torch
from tqdm.auto import tqdm

from rainbowneko.engine import NekoEngineMixin, NekoModelMixin, NekoDataMixin, NekoAccelerateMixin, NekoLoggerMixin, NekoCkptMixin, \
    NekoAccelerateSingleCardMixin
from rainbowneko.infer import WorkflowRunner
from rainbowneko.models.wrapper import BaseWrapper
from rainbowneko.parser import load_config
from rainbowneko.utils import weight_dtype_map
from .metrics import BaseMetric, MetricGroup


class Evaluator(NekoEngineMixin, NekoAccelerateMixin, NekoModelMixin, NekoDataMixin, NekoCkptMixin, NekoLoggerMixin):
    def __init__(self, parser, cfgs_raw, ds_name=None, interval=100, trainer=None, **cfgs):
        super().__init__(parser, cfgs_raw, **cfgs)
        if trainer is None:
            self.init_context(cfgs_raw)
            self.build_loggers(cfgs_raw)

            self.build_model()

            self.cfgs.dataset: partial
            workers = self.cfgs.dataset.keywords.pop("workers", 0)
            self.data_loader = self.build_data(self.cfgs.dataset, workers, train=False)
            self.data_loader.dataset.bucket.rest(0)

            self.model_wrapper.post_init()
            self.config_model()
            self.load_resume(self.cfgs.resume)
            self.prepare()
        else:
            self.accelerator = trainer.accelerator
            self.local_rank = trainer.local_rank
            self.world_size = trainer.world_size
            self.loggers = trainer.loggers

            self.model_wrapper = trainer.model_wrapper
            self.weight_dtype = trainer.weight_dtype
            print(self.cfgs.dataset)

            workers = self.cfgs.dataset.keywords.pop("workers", 0)
            self.data_loader = self.build_data(self.cfgs.dataset, workers, train=False)
            self.data_loader.dataset.bucket.rest(0)

        torch.backends.cuda.matmul.allow_tf32 = self.cfgs.get('allow_tf32', False)
        self.metric: BaseMetric = self.cfgs.metric
        self.interval = interval
        self.ds_name = ds_name
        self.metric.to(self.device)

    def prepare(self):
        # Prepare everything with accelerator.
        prepare_name_list, prepare_obj_list = [], []
        for k, v in self.model_wrapper.trainable_models.items():
            prepare_obj_list.append(v)
            prepare_name_list.append(k)
        N_trainable_models = len(prepare_obj_list)

        prepared_obj = self.accelerator.prepare(*prepare_obj_list)

        # prepared model
        if prepare_name_list[0] == "self":  # entire model is trainable
            if N_trainable_models == 1:
                self.model_wrapper = prepared_obj
            else:
                self.model_wrapper = prepared_obj[0]
        else:
            for name, obj in zip(prepare_name_list[:N_trainable_models], prepared_obj[:N_trainable_models]):
                setattr(self.model_wrapper, name, obj)

        self.model_wrapper: BaseWrapper = self.model_wrapper.to(self.device)
        if self.cfgs.model.force_cast_precision:
            self.model_wrapper.to(dtype=self.weight_dtype)

    def forward_one_step(self, model, data):
        input_datas = {k: self.to_dev(v) for k, v in data.items() if k != "plugin_input"}
        if "plugin_input" in data:
            input_datas["plugin_input"] = {k: self.to_dev(v) for k, v in data["plugin_input"].items()}

        with torch.autocast(self.device.type, dtype=self.weight_dtype, enabled=self.weight_dtype != torch.float32):
            model_pred = model(self.ds_name, **input_datas)

        return model_pred, input_datas

    @torch.no_grad()
    def evaluate(self, step: int, prefix='eval/'):
        if step % self.interval != 0:
            return

        # record training layers
        training_layers = [layer for layer in self.model_raw.modules() if layer.training]

        # reset metric
        self.model_wrapper.eval()
        self.metric.reset()

        for data in tqdm(self.data_loader, disable=not self.is_local_main_process):
            pred, input_datas = self.forward_one_step(self.model_wrapper, data)
            # update data to metric
            self.metric.update(pred, input_datas)

        v_metric = self.metric.finish(self.cpu_gather, self.is_local_main_process)
        if not isinstance(v_metric, dict):
            v_metric = {'metric': v_metric}

        data_size = len(self.data_loader.dataset)
        self.loggers.info(f'Evaluate: data size {data_size}')
        log_data = {
            "eval/Step": {
                "format": "{}",
                "data": [step],
            }
        }
        log_data.update(MetricGroup.format(v_metric, prefix=prefix))
        self.loggers.log(log_data, step, force=True)

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


class EvaluatorSingle(NekoAccelerateSingleCardMixin, Evaluator):
    pass


class WorkflowEvaluator(Evaluator):
    def __init__(self, parser, cfgs_raw, workflow: str | ModuleType | Dict, ds_name=None, interval=100, trainer=None,
                 mixed_precision=None, seed=42, **cfgs):
        cfgs['seed'] = seed
        cfgs['mixed_precision'] = mixed_precision
        super(Evaluator, self).__init__(parser, cfgs_raw, **cfgs)  # skip Evaluator init

        self.model_wrapper: BaseWrapper | None
        if trainer is None:
            self.init_context(cfgs_raw)
            if self.cfgs.get('logger', None) is not None:
                self.build_loggers(cfgs_raw)
            else:
                self.loggers = None
            self.weight_dtype = weight_dtype_map.get(self.cfgs.mixed_precision, torch.float32)

            if isinstance(workflow, (ModuleType, str)):
                parser, conf = load_config(workflow)
                self.workflow_runner = WorkflowRunner(parser, conf)
            else:
                self.workflow_runner = WorkflowRunner(parser, workflow, cfgs_raw=cfgs_raw)
            self.in_preview = False
        else:
            self.accelerator = trainer.accelerator
            self.local_rank = trainer.local_rank
            self.world_size = trainer.world_size
            self.loggers = trainer.loggers

            self.model_wrapper = trainer.model_wrapper
            self.weight_dtype = trainer.weight_dtype

            if isinstance(workflow, (ModuleType, str)):
                parser, conf = load_config(workflow)
                self.workflow_runner = WorkflowRunner(parser, conf)
            else:
                self.workflow_runner = WorkflowRunner(parser, workflow, cfgs_raw=cfgs_raw)
            self.in_preview = True

        self.interval = interval
        self.ds_name = ds_name

    @torch.no_grad()
    def evaluate(self, step: int, prefix='eval/'):
        if step % self.interval != 0:
            return

        # record training layers
        if getattr(self, 'model_wrapper', None) is not None:
            training_layers = [layer for layer in self.model_raw.modules() if layer.training]
            self.model_wrapper.eval()
            model = self.model_wrapper
        else:
            training_layers = []
            model = None

        states = self.workflow_runner.run(model=model, in_preview=self.in_preview, device=self.device, dtype=self.weight_dtype,
                                          world_size=self.world_size, local_rank=self.local_rank)
        metric = states['_metric']
        loggers = states.get('loggers', None)

        v_metric = metric.finish(self.accelerator.gather, self.is_local_main_process)
        if not isinstance(v_metric, dict):
            v_metric = {'metric': v_metric}

        log_data = {
            "eval/Step": {
                "format": "{}",
                "data": [step],
            }
        }
        log_data.update(MetricGroup.format(v_metric, prefix=prefix))
        if self.loggers is not None:
            self.loggers.log(log_data, step, force=True)
        elif loggers is not None:
            loggers.log(log_data, step, force=True)
        else:
            print(', '.join([f"{os.path.basename(k)} = {v['format'].format(*v['data'])}" for k, v in log_data.items()]))

        for layer in training_layers:
            layer.train()

    def to(self, device):
        pass

class WorkflowEvaluatorSingle(NekoAccelerateSingleCardMixin, WorkflowEvaluator):
    pass