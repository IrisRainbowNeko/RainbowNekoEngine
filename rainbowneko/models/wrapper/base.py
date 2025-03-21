from torch import nn
from typing import Dict, List, Tuple, Any

from rainbowneko.utils import KeyMapper, is_dict, is_list


class BaseWrapper(nn.Module):
    def post_init(self):
        pass

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

    def build_mapper(self, key_map, model=None, default=None):
        if is_dict(key_map) and len(key_map)>0:
            first_key = next(iter(key_map.values()))
            if is_dict(first_key) or isinstance(first_key, tuple) or is_list(first_key):
                return {k:KeyMapper(model, v or default) for k, v in key_map.items()}
        return KeyMapper(model, key_map or default)

    def get_map_data(self, key_mapper, data, ds_name:str=None) -> Tuple[Tuple, Dict[str, Any]]:
        if is_dict(key_mapper):
            model_args, model_kwargs = key_mapper[ds_name].map_data(data)
        else:
            model_args, model_kwargs = key_mapper.map_data(data)
        return model_args, model_kwargs

    def get_inputs_feed(self, key_mapper_in, model, kwargs, plugin_input={}, ds_name=None) -> Tuple[Tuple, Dict[str, Any]]:
        model_args, model_kwargs = self.get_map_data(key_mapper_in, kwargs, ds_name)

        input_all = dict(_args_=model_args, **model_kwargs, **plugin_input)

        if hasattr(model, 'input_feeder'):
            for feeder in model.input_feeder:
                feeder(input_all)
        return model_args, model_kwargs

class SingleWrapper(BaseWrapper):
    def __init__(self, model, key_map_in=None, key_map_out=None):
        super().__init__()
        self.model = model
        self.key_mapper_in = self.build_mapper(key_map_in, model, {0: 'image'})
        self.key_mapper_out = self.build_mapper(key_map_out, model, {'pred': 0})

    def forward(self, ds_name=None, plugin_input={}, **kwargs):
        model_args, model_kwargs = self.get_inputs_feed(self.key_mapper_in, self.model, kwargs, plugin_input, ds_name=ds_name)

        out = self.model(*model_args, **model_kwargs)
        if not isinstance(out, (tuple, dict)):
            out = (out,)
        return self.get_map_data(self.key_mapper_out, out, ds_name=ds_name)[1]

    @property
    def trainable_models(self) -> Dict[str, nn.Module]:
        return {'self':self}
