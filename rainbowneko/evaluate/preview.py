import os.path

import torch
from tqdm.auto import tqdm

from rainbowneko.models.wrapper import BaseWrapper
from rainbowneko.train.data import DataGroup
from .evaluator import Evaluator
from .renderer import Renderer


class Previewer(Evaluator):
    def __init__(self, trainer: "Trainer", data_loader_group: DataGroup, img_dir: str, renderer: Renderer, interval=100):
        self.data_loader_group = data_loader_group
        self.trainer = trainer
        self.img_dir = img_dir
        self.renderer = renderer
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

    def save_images(self, pred, target, img_dir, step):
        img_list = self.renderer(pred, target)
        for img in img_list:
            img.save(os.path.join(img_dir, f"{step}-{self.img_count}.png"))
            self.img_count += 1

    @torch.inference_mode()
    def evaluate(self, step: int, model: BaseWrapper):
        if step % self.interval != 0:
            return

        model.eval()

        img_dir = os.path.join(self.trainer.exp_dir, self.img_dir)
        self.trainer.loggers.info(f'Preview to {img_dir}')
        self.img_count=0
        for loader in self.data_loader_group.loader_dict.values():
            for data in tqdm(loader, disable=not self.trainer.is_local_main_process):
                pred, target = self.forward_one_step(model, data)
                self.save_images(pred, target, img_dir=img_dir, step=step)

    def to(self, device):
        self.renderer.to(device)
