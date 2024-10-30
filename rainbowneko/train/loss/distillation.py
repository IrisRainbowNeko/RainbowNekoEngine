from torch import nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, T, weight=0.95):
        super().__init__()
        self.T=T
        self.kl_div = nn.KLDivLoss()
        self.alpha = T*T * 2.0 * weight

    _key_map = {'pred_student': 'pred.pred', 'pred_teacher': 'pred.pred_teacher'}
    def forward(self, pred_student, pred_teacher):
        return self.kl_div(F.log_softmax(pred_student/self.T), F.softmax(pred_teacher/self.T)) * self.alpha