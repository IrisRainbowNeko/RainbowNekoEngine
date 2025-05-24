from typing import Dict, List, Tuple, Union

import torch
from rainbowneko.utils import FILE_LIKE
from io import BytesIO

from rainbowneko.utils import is_list
from .base import CkptFormat

import onnx
import onnxruntime


class ONNXFormat(CkptFormat):
    EXT = 'onnx'

    def __init__(self, inputs: Dict[str, Tuple] = {'input': (1, 3, 224, 224)}, output_names: List[str] = ['output'],
                 dynamic_axes: Union[Dict[str, Dict], str] = 'batch'):
        self.inputs, self.input_names, parsed_dynamic_axes = self.parse_inputs(inputs)
        self.output_names = output_names

        if len(parsed_dynamic_axes) > 0:
            self.dynamic_axes = parsed_dynamic_axes
            if dynamic_axes == 'batch':
                self.dynamic_axes.update({key: {0: 'batch'} for key in output_names})
            else:
                self.dynamic_axes.update(dynamic_axes)
        elif dynamic_axes == 'batch':
            self.dynamic_axes = {key: {0: 'batch'} for key in inputs.keys()}
            self.dynamic_axes.update({key: {0: 'batch'} for key in output_names})
        else:
            self.dynamic_axes = dynamic_axes

    def parse_inputs(self, inputs: Dict[str, Tuple]):
        '''
        {'input': (('batch', 1),3,224,224)} -> inputs={'input':(1,3,224,224)}, dynamic_axes={'input':{0:'batch'}}
        :param inputs:
        :return:
        '''
        input_tensors = []
        input_names = []
        dynamic_axes = {}

        for key, shape in inputs.items():
            dynamic = {}
            shape_clean = []
            for i, x in enumerate(shape):
                if is_list(x):
                    dynamic[i] = x[0]
                    shape_clean.append(x[1])
                else:
                    shape_clean.append(x)
            input_names.append(key)
            input_tensors.append(torch.randn(shape_clean))
            if len(dynamic) > 0:
                dynamic_axes[key] = dynamic
        return tuple(input_tensors), input_names, dynamic_axes

    def save_ckpt(self, sd_model: Dict[str, torch.nn.Module], save_f: FILE_LIKE, **kwargs):
        torch.onnx.export(sd_model, self.inputs, save_f, input_names=self.input_names, output_names=self.output_names,
                          dynamic_axes=self.dynamic_axes, **kwargs)

    def load_ckpt(self, ckpt_f: FILE_LIKE, providers=['CPUExecutionProvider', 'CUDAExecutionProvider']):
        if isinstance(ckpt_f, BytesIO):
            ckpt_f = ckpt_f.getvalue()
        ort_session = onnxruntime.InferenceSession(ckpt_f, providers=providers)
        return ort_session
