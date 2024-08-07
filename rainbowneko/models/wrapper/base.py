from torch import nn
from typing import Dict, List

from rainbowneko.utils import KeyMapper


class BaseWrapper(nn.Module):

    @property
    def trainable_models(self) -> Dict[str, nn.Module]:
        raise NotImplementedError

    @property
    def trainable_parameters(self) -> List[nn.Parameter]:
        return [v for k, v in self.named_parameters() if v.requires_grad]

    @property
    def trainable_layers(self) -> List[nn.Module]:
        return self._trainable_layers

    @trainable_layers.setter
    def trainable_layers(self, layers: List[nn.Module]):
        self._trainable_layers = layers

    def update_model(self, step:int):
        pass

    def freeze_model(self):
        self.eval()

    def enable_gradient_checkpointing(self):
        pass

    def enable_xformers(self):
        pass

    def train(self, mode: bool = True):
        self.training = mode
        
        if hasattr(self, 'trainable_layers'):
            for layer in self.trainable_layers:
                if layer==self:
                    super().train(mode)
                else:
                    layer.train(mode)
        else:
            super().train(mode)
        return self

class SingleWrapper(BaseWrapper):
    def __init__(self, model, key_map=None):
        super().__init__()
        self.model = model
        self.key_mapper = KeyMapper(model, key_map or {'pred': '0'})

    def forward(self, input_data, plugin_input={}, **kwargs):
        input_all = dict(input_data=input_data, **plugin_input)

        if hasattr(self.model, 'input_feeder'):
            for feeder in self.model.input_feeder:
                feeder(input_all)

        out = self.model(input_data, **kwargs)
        if not isinstance(out, tuple):
            out = (out,)
        return self.key_mapper.map_data(out)

    @property
    def trainable_models(self) -> Dict[str, nn.Module]:
        return {'self':self}
