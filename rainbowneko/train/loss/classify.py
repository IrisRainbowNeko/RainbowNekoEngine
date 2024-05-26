import torch.nn as nn
import torch.nn.functional as F

class SoftCELoss(nn.Module):
    def __init__(self, reduction='batchmean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, soft_targets):
        log_probabilities = F.log_softmax(logits, dim=-1)
        loss = F.kl_div(log_probabilities, soft_targets, reduction=self.reduction)
        return loss
