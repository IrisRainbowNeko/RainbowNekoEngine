import warnings
from typing import Dict, Any, TYPE_CHECKING

import torch
from omegaconf import DictConfig
from rainbowneko import _share
from rainbowneko.models.ema import ModelEMA
from rainbowneko.utils import weight_dtype_map, maybe_DDP, xformers_available


class NekoModelMixin:
    if TYPE_CHECKING:
        cfgs: Dict[str, Any] | DictConfig

    def build_model(self):
        self.model_wrapper = self.cfgs.model.wrapper()

        self.model_wrapper.requires_grad_(False)
        self.model_wrapper.eval()
        self.weight_dtype = weight_dtype_map.get(self.cfgs.mixed_precision, torch.float32)

        for callback in _share.model_callbacks:
            callback(self.model_wrapper)

    def build_ema(self):
        if self.cfgs.model.ema is not None:
            self.ema_model: ModelEMA = self.cfgs.model.ema(self.model_wrapper)

    def update_ema(self):
        if hasattr(self, "ema_model"):
            self.ema_model.step(self.model_raw.named_parameters())

    @property
    def model_raw(self):
        return maybe_DDP(self.model_wrapper)

    def config_model(self):
        if self.cfgs.model.enable_xformers:
            if xformers_available:
                self.model_wrapper.enable_xformers()
            else:
                warnings.warn("xformers is not available. Make sure it is installed correctly")

        if self.cfgs.model.gradient_checkpointing:
            self.model_wrapper.enable_gradient_checkpointing()
