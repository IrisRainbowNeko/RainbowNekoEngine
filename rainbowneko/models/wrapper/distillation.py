from typing import Dict

import torch
from torch import nn

from .base import BaseWrapper


class DistillationWrapper(BaseWrapper):
    def __init__(self, model_teacher, model_student):
        super().__init__()
        self.model_teacher = model_teacher
        self.model_student = model_student

    def forward(self, img, img_teacher=None, plugin_input={}, **kwargs):
        input_all = dict(**plugin_input)
        if img_teacher is None:
            img_teacher = img

        if hasattr(self.model_teacher, 'input_feeder'):
            input_all['img'] = img_teacher
            for feeder in self.model_teacher.input_feeder:
                feeder(input_all)
        if hasattr(self.model_student, 'input_feeder'):
            input_all['img'] = img
            for feeder in self.model_student.input_feeder:
                feeder(input_all)

        with torch.inference_mode():
            out_teacher = self.model_teacher(img_teacher, **kwargs)
        out_student = self.model_student(img, **kwargs)
        return {'pred': out_student, 'pred_teacher': out_teacher}

    @property
    def trainable_models(self) -> Dict[str, nn.Module]:
        return {'model_student': self.model_student}
