from .base import SingleWrapper


class FeatWrapper(SingleWrapper):

    def forward(self, x, plugin_input={}, **kwargs):
        input_all = dict(x=x, **plugin_input)

        if hasattr(self.model, 'input_feeder'):
            for feeder in self.model.input_feeder:
                feeder(input_all)

        out, feat = self.model(x, **kwargs)
        if not isinstance(feat, list):
            feat = [feat]
        return {'pred': out, 'feat': feat}
