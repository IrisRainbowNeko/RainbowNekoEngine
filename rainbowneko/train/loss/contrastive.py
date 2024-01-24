import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from .base import LossContainer

class MLCEImageLoss(LossContainer):
    def __init__(self, weight=1.0, eps=1e-4, reduction='mean'):
        super().__init__(None, weight=weight)
        self.ce_loss = nn.CrossEntropyLoss()
        self.eps = eps
        self.reduction=reduction

    def forward(self, pred, target):
        input_tensor = pred['pred']  # [B,C]
        target_tensor = target['label']

        if 'label_weight' in target:
            input_tensor = input_tensor / target['label_weight']  # softmax(w*x)

        log_prob_raw = F.softmax(input_tensor, dim=1)

        same_mask = (target_tensor.unsqueeze(0) == target_tensor.unsqueeze(1)).long()  # [B,B]
        same_mask_diag0 = same_mask - torch.diag_embed(torch.diag(same_mask))  # diag=0

        log_prob_x = log_prob_raw.clone()
        log_prob_x[same_mask_diag0.bool()] = self.eps
        log_prob_x.diagonal().copy_((log_prob_raw * same_mask_diag0.detach()).sum(dim=1))
        log_prob_x = log_prob_x + torch.diag_embed(torch.ones(len(target_tensor)) * self.eps).to(input_tensor.device)
        y = torch.arange(0, len(target_tensor)).to(input_tensor.device)

        return F.nll_loss(log_prob_x.log(), y, reduction=self.reduction) * self.alpha
