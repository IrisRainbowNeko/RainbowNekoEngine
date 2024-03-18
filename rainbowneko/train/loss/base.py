from torch import nn

class LossContainer(nn.Module):
    def __init__(self, loss, weight=1.0, pred_key='pred', target_key='label'):
        super().__init__()
        self.loss = loss
        self.alpha = weight
        self.pred_key = pred_key
        self.target_key = target_key

    def forward(self, pred, target):
        return self.loss(pred[self.pred_key], target[self.target_key]) * self.alpha

class LossGroup(nn.Module):
    def __init__(self, loss_list):
        super().__init__()
        self.loss_list = loss_list

    def forward(self, pred, target):
        loss = 0
        for loss_item in self.loss_list:
            loss += loss_item(pred, target)
        return loss