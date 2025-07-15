from pathlib import Path
from typing import Dict, Any, TYPE_CHECKING, Callable

import torch
from omegaconf import DictConfig
from rainbowneko.ckpt_manager import NekoResumer, NekoSaver


class NekoCkptMixin:
    if TYPE_CHECKING:
        from rainbowneko.models.wrapper import BaseWrapper

        cfgs: Dict[str, Any] | DictConfig
        model_wrapper: BaseWrapper # Model instance
        optimizer: torch.optim.Optimizer
        is_local_main_process: Callable[[], bool]

    @torch.no_grad()
    def load_resume(self, resumer: NekoResumer):
        if resumer is not None:
            resumer.load_to(
                model=self.model_wrapper,
                optimizer=getattr(self, "optimizer", None),
                plugin_groups=getattr(self, "all_plugin", None),
                model_ema=getattr(self, "ema_model", None)
            )

    def build_ckpt_saver(self, exp_dir: str | Path):
        self.ckpt_saver: Dict[str, NekoSaver] = self.cfgs.ckpt_saver
        self.ckpt_dir = Path(exp_dir) / "ckpts"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        for ckpt_saver in self.ckpt_saver.values():
            ckpt_saver.prefix = self.ckpt_dir
