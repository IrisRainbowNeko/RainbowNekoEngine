from torch import nn

class LossContainer(nn.Module):
    def __init__(self, loss, weight=1.0):
        super().__init__()
        self.loss = loss
        self.alpha = weight

    def forward(self, pred, target):
        return self.loss(pred['pred'], target['label']) * self.alpha

class LossGroup(nn.Module):
    def __init__(self, loss_list):
        super().__init__()
        self.loss_list = loss_list

    def forward(self, pred, target):
        loss = 0
        for loss_item in self.loss_list:
            loss += loss_item(pred, target)
        return loss