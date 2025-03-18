import os.path

import torch
from tqdm.auto import tqdm

from pathlib import Path
from rainbowneko.models.wrapper import BaseWrapper
from torch.utils.data import DataLoader
from .evaluator import Evaluator, WorkflowEvaluator
from .renderer import Renderer


class Previewer(Evaluator):
    def __init__(self, trainer: "Trainer", data_loader: DataLoader, img_dir: str, renderer: Renderer, ds_name=None, interval=100, dataset:None=None):
        assert dataset is None, '"dataset" is a placeholder for cfg, should always be None.'
        self.data_loader = data_loader
        self.trainer = trainer
        self.img_dir = img_dir
        self.renderer = renderer
        self.interval = interval
        self.ds_name = ds_name

    def save_images(self, pred, target, img_dir, step):
        img_list = self.renderer(pred, target)
        for img in img_list:
            img.save(os.path.join(img_dir, f"{step}-{self.img_count}.png"))
            self.img_count += 1

    @torch.no_grad()
    def evaluate(self, step: int, model: BaseWrapper):
        if step % self.interval != 0:
            return

        # record training layers
        training_layers = [layer for layer in model.modules() if layer.training]

        model.eval()

        img_dir = os.path.join(self.trainer.exp_dir, self.img_dir)
        self.trainer.loggers.info(f'Preview to {img_dir}')
        self.img_count=0
        for data in tqdm(self.data_loader, disable=not self.trainer.is_local_main_process):
            pred, input_datas = self.forward_one_step(model, data)
            self.save_images(pred, input_datas, img_dir=img_dir, step=step)

        for layer in training_layers:
            layer.train()

    def to(self, device):
        self.renderer.to(device)


class WorkflowPreviewer(WorkflowEvaluator):
    @torch.no_grad()
    def evaluate(self, step: int, model: BaseWrapper, prefix='eval/'):
        if step % self.interval != 0:
            return

        # record training layers
        training_layers = [layer for layer in model.modules() if layer.training]

        model.eval()
        self.trainer.loggers.info(f'Preview')

        states = self.workflow_runner.run(model=model, in_preview=True, device=self.device, dtype=self.dtype,
                                          preview_root=Path(self.trainer.exp_dir)/'imgs',
                                          world_size=self.trainer.world_size, local_rank=self.trainer.local_rank)

        for layer in training_layers:
            layer.train()