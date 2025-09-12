from .base import SingleWrapper


class FeatWrapper(SingleWrapper):
    def __init__(self, model, key_map_in=None, key_map_out=('pred -> pred', 'feat -> feat')):
        super().__init__(model, key_map_in, key_map_out)

    def forward(self, ds_name=None, plugin_input={}, **kwargs):
        model_args, model_kwargs = self.get_inputs_feed(self.key_mapper_in, self.model, kwargs, plugin_input, ds_name=ds_name)

        out, feat = self.model(*model_args, **model_kwargs)
        if not isinstance(feat, list):
            feat = [feat]
        return self.get_map_data(self.key_mapper_out, {'pred': out, 'feat': feat}, ds_name=ds_name)[1]
