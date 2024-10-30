from torch import nn
import torch.nn.functional as F
import torch

class PseudoLabelLoss(nn.Module):
    def __init__(self, threshold=0.7):
        super().__init__()
        self.threshold=threshold

    _key_map = {'pred': 'pred.pred', 'pred_label': 'pred.pred_teacher'}
    def forward(self, pred, pred_label):
        pseudo_label = torch.softmax(pred_label.detach(), dim=-1)
        max_probs, pseudo_targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(self.threshold).float()

        loss_per_sample = F.cross_entropy(pred, pseudo_targets_u, reduction='none')
        return loss_per_sample * mask