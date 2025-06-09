from pathlib import Path
from types import ModuleType
from typing import Dict

import torch

from .evaluator import WorkflowEvaluator


class WorkflowPreviewer(WorkflowEvaluator):
    def __init__(self, parser, cfgs_raw, workflow: str | ModuleType | Dict, ds_name=None, interval=100, trainer=None,
                 mixed_precision=None, seed=42, **cfgs):
        super().__init__(parser, cfgs_raw, workflow, ds_name=ds_name, interval=interval, trainer=trainer,
                         mixed_precision=mixed_precision, seed=seed, **cfgs)
        if trainer is None:
            self.exp_dir = self.cfgs.exp_dir
        else:
            self.exp_dir = trainer.exp_dir

    @torch.no_grad()
    def evaluate(self, step: int, prefix='eval/'):
        if step % self.interval != 0:
            return

        # record training layers
        if self.model_wrapper is not None:
            training_layers = [layer for layer in self.model_raw.modules() if layer.training]
            self.model_wrapper.eval()
            model = self.model_wrapper
        else:
            training_layers = []
            model = None

        if self.loggers is not None:
            self.loggers.info(f'Preview')

        states = self.workflow_runner.run(model=model, in_preview=True, device=self.device, dtype=self.weight_dtype,
                                          preview_root=Path(self.exp_dir) / 'imgs',
                                          world_size=self.world_size, local_rank=self.local_rank)

        for layer in training_layers:
            layer.train()
