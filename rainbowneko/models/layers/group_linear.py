import torch
from torch import nn
import math

class GroupLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, group: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group = group
        self.weight = nn.Parameter(torch.empty((group, out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(group, 1, out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    @staticmethod
    def _calculate_fan_in_and_fan_out(tensor):
        dimensions = tensor.dim()
        if dimensions < 2:
            raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

        num_input_fmaps = tensor.size(-2)
        num_output_fmaps = tensor.size(-1)
        fan_in = num_input_fmaps
        fan_out = num_output_fmaps

        return fan_in, fan_out

    @staticmethod
    def kaiming_uniform_(
            tensor: torch.Tensor,
            a: float = 0,
            nonlinearity: str = "leaky_relu",
            generator = None,
    ):
        fan_in, _ = GroupLinear._calculate_fan_in_and_fan_out(tensor)
        gain = nn.init.calculate_gain(nonlinearity, a)
        std = gain / math.sqrt(fan_in)
        bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        with torch.no_grad():
            return tensor.uniform_(-bound, bound, generator=generator)


    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        self.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = self._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        '''
        input(G,N,CI) x weight.T(G,CI,CO) [+ bias(G,1,CO)] -> out(G,N,CO)
        '''
        if self.bias is None:
            return torch.bmm(input, self.weight.transpose(1,2))
        else:
            return torch.baddbmm(self.bias, input, self.weight.transpose(1, 2))

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, group={self.group}, bias={self.bias is not None}'