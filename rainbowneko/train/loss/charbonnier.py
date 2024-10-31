import torch
from torch import nn

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, size_average=True, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.size_average = size_average
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sqrt((diff * diff) + (self.eps*self.eps))
        if self.size_average:
            loss = torch.mean(loss)
        return loss