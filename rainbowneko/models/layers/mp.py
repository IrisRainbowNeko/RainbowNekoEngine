import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t
import math
from typing import Union


def MP_add(a, b, alpha=1):
    # a+b*alpha
    return torch.add(a, b, alpha=alpha) / torch.sqrt(1 + alpha ** 2)


def MP_add_full(a, b, alpha=1, beta=1):
    return (a * alpha + b * beta) / torch.sqrt(alpha ** 2 + beta ** 2)


def MP_mul(a, b):
    norm = (a.norm(2, dim=-1).sqrt() * b.norm(2, dim=-1).sqrt()).unsqueeze(-1)
    return (a / norm) * b


def MP_cat(a, b, dim=-1, t=0.5):
    Na, Nb = a.shape[dim], b.shape[dim]
    C = math.sqrt((Na + Nb) / ((1. - t) ** 2 + t ** 2))

    a = a * (1. - t) / math.sqrt(Na)
    b = b * t / math.sqrt(Nb)
    return C * torch.cat((a, b), dim=dim)


class MPConv2d(nn.Conv2d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            stride: _size_2_t = 1,
            padding: Union[str, _size_2_t] = 0,
            dilation: _size_2_t = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
            device=None,
            dtype=None,
            eps=1e-4
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device,
                         dtype)

        self.eps = eps
        self.fan_in_sqrt = math.sqrt(in_channels // groups * math.prod(kernel_size))

    def normalize_weight(self, weight, eps=1e-4):
        weight_shape = weight.shape
        weight, ps = weight.flatten(1)
        normed_weight = F.normalize(weight, dim=-1, eps=eps) * self.fan_in_sqrt
        return normed_weight.view(weight_shape)

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                normed_weight = self.normalize_weight(self.weight, eps=self.eps)
                self.weight.copy_(normed_weight)
            weight = self.weight / self.fan_in_sqrt
        else:
            weight = self.normalize_weight(self.weight, eps=self.eps) / self.fan_in_sqrt

        output = self._conv_forward(x, weight, self.bias)
        if self.bias is None:
            return output
        else:
            return output / torch.sqrt(1 + self.bias.norm(2).square())


class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, eps=1e-4):
        super().__init__(in_features, out_features, bias, device, dtype)

        self.eps = eps
        self.fan_in_sqrt = math.sqrt(in_features)

    def normalize_weight(self, weight, eps=1e-4):
        weight_shape = weight.shape
        weight, ps = weight.flatten(1)
        normed_weight = F.normalize(weight, dim=-1, eps=eps) * self.fan_in_sqrt
        return normed_weight.view(weight_shape)

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                normed_weight = self.normalize_weight(self.weight, eps=self.eps)
                self.weight.copy_(normed_weight)
            weight = self.weight / self.fan_in_sqrt
        else:
            weight = self.normalize_weight(self.weight, eps=self.eps) / self.fan_in_sqrt

        output = F.linear(x, weight, self.bias)
        if self.bias is None:
            return output
        else:
            return output / torch.sqrt(1 + self.bias.norm(2).square())


class MPSiLU(nn.Module):
    def forward(self, x):
        return F.silu(x) / 0.596


class MPGELU(nn.Module):
    def forward(self, x):
        return F.gelu(x) / 0.64958


class MPGEGLU(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * F.gelu(x2)
        return F.gelu(x) / 0.64958
