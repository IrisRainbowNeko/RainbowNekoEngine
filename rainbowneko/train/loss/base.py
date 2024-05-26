from torch import nn

class LossContainer(nn.Module):
    def __init__(self, loss, weight=1.0, key_map=None):
        super().__init__()
        self.loss = loss
        self.alpha = weight
        if key_map is None and loss is not None:
            if hasattr(loss, '_key_map'):
                self.key_map = loss._key_map
            else:
                self.key_map = {0: 'pred.pred', 1: 'target.label'}

    def forward(self, pred, target):
        args = []
        kwargs = {}
        for k_dst, k_src in self.key_map.items():
            keys = k_src.split('.')
            v = locals()[keys[0]]
            for k in keys[1:]:
                v = v[k]
            if isinstance(k_dst, int):
                args.append(v)
            else:
                kwargs[k_dst] = v

        return self.loss(*args, **kwargs) * self.alpha

class LossGroup(nn.Module):
    def __init__(self, loss_list):
        super().__init__()
        self.loss_list = loss_list

    def forward(self, pred, target):
        loss = 0
        for loss_item in self.loss_list:
            loss += loss_item(pred, target)
        return loss