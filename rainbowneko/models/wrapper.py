from torch import nn
from typing import Dict

class BaseWrapper(nn.Module):

    @property
    def trainable_models(self) -> Dict[str, nn.Module]:
        raise NotImplementedError

    @property
    def trainable_named_parameters(self) -> Dict[str, nn.Parameter]:
        return {k:v for k, v in self.named_parameters() if v.requires_grad}

    def freeze_model(self):
        self.eval()

    def enable_gradient_checkpointing(self):
        pass

    def enable_xformers(self):
        pass

class SingleWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, plugin_input={}, **kwargs):
        input_all = dict(x=x, **plugin_input)

        if hasattr(self.model, 'input_feeder'):
            for feeder in self.model.input_feeder:
                feeder(input_all)

        out = self.model(x, **kwargs)
        return out

    @property
    def trainable_models(self) -> Dict[str, nn.Module]:
        return {'self':self}
