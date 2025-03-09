from torch import nn
from rainbowneko.utils import KeyMapper

class LossContainer(nn.Module):
    def __init__(self, loss, weight=1.0, key_map=None):
        super().__init__()
        self.key_mapper = KeyMapper(loss, key_map)
        self.loss = loss
        self.weight = weight

    def forward(self, pred, inputs):
        args, kwargs = self.key_mapper(pred=pred, inputs=inputs)
        return self.loss(*args, **kwargs) * self.weight

class LossGroup(nn.Module):
    def __init__(self, loss_list):
        super().__init__()
        self.loss_list = loss_list

    def forward(self, pred, target):
        loss = 0
        for loss_item in self.loss_list:
            loss += loss_item(pred, target).squeeze()
        return loss