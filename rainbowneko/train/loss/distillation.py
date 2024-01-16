from torch import nn
from .base import LossContainer
import torch.nn.functional as F

class DistillationLoss(LossContainer):
    def __init__(self, T, weight=0.95):
        super().__init__(None)
        self.T=T
        self.kl_div = nn.KLDivLoss()
        self.alpha = T*T * 2.0 * weight

    def forward(self, pred, target):
        return self.kl_div(F.log_softmax(pred['pred']/self.T), F.softmax(pred['pred_teacher']/self.T)) * self.alpha