from torch import nn
from typing import Dict, List

class BaseWrapper(nn.Module):

    @property
    def trainable_models(self) -> Dict[str, nn.Module]:
        raise NotImplementedError

    @property
    def trainable_parameters(self) -> List[nn.Parameter]:
        return [v for k, v in self.named_parameters() if v.requires_grad]

    def freeze_model(self):
        self.eval()

    def enable_gradient_checkpointing(self):
        pass

    def enable_xformers(self):
        pass

class SingleWrapper(BaseWrapper):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, plugin_input={}, **kwargs):
        input_all = dict(x=x, **plugin_input)

        if hasattr(self.model, 'input_feeder'):
            for feeder in self.model.input_feeder:
                feeder(input_all)

        out = self.model(x, **kwargs)
        return {'pred': out}

    @property
    def trainable_models(self) -> Dict[str, nn.Module]:
        return {'self':self}
