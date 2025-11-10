from torch import nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T=T
        self.alpha = T*T

    _key_map = {'pred_student': 'pred.pred', 'pred_teacher': 'pred.pred_teacher'}
    def forward(self, pred_student, pred_teacher):
        return F.kl_div(F.log_softmax(pred_student/self.T), F.softmax(pred_teacher/self.T), reduction='batchmean') * self.alpha