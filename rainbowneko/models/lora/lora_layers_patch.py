"""
lora_layers.py
====================
    :Name:        lora layers
    :Author:      Dong Ziyi
    :Affiliation: HCP Lab, SYSU
    :Created:     09/04/2023
    :Licence:     Apache-2.0
"""

import math

import torch
from einops import einsum
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair

from .lora_base_patch import LoraBlock


class LoraLayer(LoraBlock):
    def __init__(self, lora_id: int, host, rank=1, dropout=0.1, alpha=1.0, bias=False, alpha_auto_scale=True, **kwargs):
        super().__init__(lora_id, host, rank, dropout, alpha=alpha, bias=bias, alpha_auto_scale=alpha_auto_scale, **kwargs)

    class LinearLayer(LoraBlock.LinearLayer):
        def __init__(self, host: nn.Linear, rank, bias, block):
            super().__init__(host, rank, bias, block)
            if isinstance(self.rank, float):
                self.rank = max(round(host.out_features * self.rank), 1)

            self.in_features = host.in_features
            self.out_features = host.out_features
            self.W_down = nn.Parameter(torch.empty(self.rank, host.in_features))
            self.W_up = nn.Parameter(torch.empty(host.out_features, self.rank))
            if bias:
                self.bias = nn.Parameter(torch.empty(host.out_features))
            else:
                self.register_parameter('bias', None)

        def extra_repr(self) -> str:
            return f'in_features={self.in_features}, rank={self.rank}, out_features={self.out_features}'

        def reset_parameters(self):
            nn.init.kaiming_uniform_(self.W_down, a=math.sqrt(5))
            nn.init.zeros_(self.W_up)
            if self.bias:
                nn.init.zeros_(self.bias)

        def get_weight(self):
            return torch.mm(self.W_up, self.W_down)

        def get_bias(self):
            return self.bias

        def forward(self, x, weight, bias=None):
            # make it faster
            x_shape = x.shape
            if bias is None:
                return torch.mm(x.view(-1, x_shape[-1]), weight.transpose(0, 1)).view(*x_shape[:-1], -1)
            else:
                return torch.mm(x.view(-1, x_shape[-1]), weight.transpose(0, 1)).view(*x_shape[:-1], -1) + bias
            # return F.linear(x, weight, bias) # linear is slow

        def get_collapsed_param(self):
            w = self.W_up.data @ self.W_down.data
            b = self.bias.data if self.bias else None
            return w, b

    class Conv2dLayer(LoraBlock.Conv2dLayer):
        def __init__(self, host: nn.Conv2d, rank, bias, block):
            super().__init__(host, rank, bias, block)
            if isinstance(self.rank, float):
                self.rank = max(round(host.out_channels * self.rank), 1)

            self.in_channels = host.in_channels
            self.out_channels = host.out_channels
            self.kernel_size = host.kernel_size

            self.W_down = nn.Parameter(torch.empty(self.rank, host.in_channels, *host.kernel_size))
            self.W_up = nn.Parameter(torch.empty(host.out_channels, self.rank, 1, 1))
            if bias:
                self.bias = nn.Parameter(torch.empty(host.out_channels))
            else:
                self.register_parameter('bias', None)

            self.stride = host.stride
            self.padding = host.padding
            self.dilation = host.dilation
            self.groups = host.groups
            self.padding_mode = host.padding_mode
            self._reversed_padding_repeated_twice = host._reversed_padding_repeated_twice

        def extra_repr(self):
            s = ('{in_channels}, {out_channels}, rank={rank}, kernel_size={kernel_size}'
                 ', stride={stride}')
            if self.padding != (0,) * len(self.padding):
                s += ', padding={padding}'
            if self.dilation != (1,) * len(self.dilation):
                s += ', dilation={dilation}'
            if self.groups != 1:
                s += ', groups={groups}'
            if self.bias is None:
                s += ', bias=False'
            if self.padding_mode != 'zeros':
                s += ', padding_mode={padding_mode}'
            return s.format(**self.__dict__)

        def reset_parameters(self):
            nn.init.kaiming_uniform_(self.W_down, a=math.sqrt(5))
            nn.init.zeros_(self.W_up)
            if self.bias:
                nn.init.zeros_(self.bias)

        def get_weight(self):
            return einsum(self.W_up, self.W_down, 'o r ..., r i ... -> o i ...')

        def get_bias(self):
            return self.bias if self.bias else None

        def forward(self, x, weight, bias=None):
            if self.padding_mode != 'zeros':
                return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                weight, bias, self.stride, _pair(0), self.dilation, self.groups)
            return F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)

        def get_collapsed_param(self):
            w = einsum(self.W_up.data, self.W_down.data, 'o r ..., r i ... -> o i ...')
            b = self.bias.data if self.bias else None
            return w, b

if __name__ == '__main__':
    from timm.models.swin_transformer_v2 import swinv2_base_window8_256

    model = swinv2_base_window8_256()
    lora_layers = LoraLayer.wrap_model(0, model)
    print(model)