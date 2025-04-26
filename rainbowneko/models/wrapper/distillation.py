from typing import Dict

import torch
from torch import nn

from .base import BaseWrapper
from rainbowneko.utils import maybe_DDP


class DistillationWrapper(BaseWrapper):
    def __init__(self, model_teacher, model_student, ema=None, key_map_in_student=None, key_map_in_teacher=None, key_map_out=None):
        super().__init__()
        self.model_teacher = model_teacher
        self.model_student = model_student

        self.key_map_in_student = self.build_mapper(key_map_in_student, default=('image -> 0',))
        self.key_map_in_teacher = self.build_mapper(key_map_in_teacher, default=('image -> 0',))
        self.key_mapper_out = self.build_mapper(key_map_out, default=('pred_student -> pred', 'pred_teacher -> pred_teacher'))

        if ema is not None:
            self.ema_teacher = ema(self.model_teacher)
            self.model_teacher = None

    def forward(self, ds_name, plugin_input={}, **kwargs):
        inputs_T_args, inputs_T_kwargs = self.get_inputs_feed(self.key_map_in_teacher, self.model_teacher or self.ema_teacher.model, kwargs, plugin_input,
                                                              ds_name=ds_name)
        inputs_S_args, inputs_S_kwargs = self.get_inputs_feed(self.key_map_in_student, maybe_DDP(self.model_student), kwargs, plugin_input,
                                                              ds_name=ds_name)

        res = {}
        if len(inputs_T_args)>0 or len(inputs_T_kwargs)>0:
            with torch.no_grad():
                if hasattr(self, 'ema_teacher'):
                    out_teacher = self.ema_teacher(*inputs_T_args, **inputs_T_kwargs)
                else:
                    out_teacher = self.model_teacher(*inputs_T_args, **inputs_T_kwargs)
            res['pred_teacher'] = out_teacher
        out_student = self.model_student(*inputs_S_args, **inputs_S_kwargs)
        res['pred_student'] = out_student

        return self.get_map_data(self.key_mapper_out, res, ds_name=ds_name)[1]

    def update_model(self, step:int):
        if hasattr(self, 'ema_teacher'):
            self.ema_teacher.step(maybe_DDP(self.model_student))

    @property
    def trainable_models(self) -> Dict[str, nn.Module]:
        return {'model_student': self.model_student}
