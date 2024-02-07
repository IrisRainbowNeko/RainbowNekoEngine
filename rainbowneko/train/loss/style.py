import torch
from torch import nn
from torch.nn import functional as F

from .base import LossContainer
from rainbowneko.models.layers import StyleSimilarity
import math


class StyleLoss(LossContainer):
    def __init__(self, weight=1.0, eps=1e-4, reduction='mean'):
        super().__init__(None, weight=weight)
        self.ce_loss = nn.CrossEntropyLoss()
        self.eps = eps
        self.reduction = reduction
        self.style_sim = StyleSimilarity()

    def forward(self, pred, target):
        input_tensor = pred['feat']  # list([B,C,H,W])
        target_tensor = target['label']

        if 2 in target_tensor:  # triplet style loss
            query = [item[target_tensor == 0, ...] for item in input_tensor]
            positive_key = [item[target_tensor == 1, ...] for item in input_tensor]
            negative_keys = [item[target_tensor == 2, ...] for item in input_tensor]

            loss_pos = sum(self.style_sim(q, pos) for q, pos in zip(query, positive_key))/len(query)
            loss_neg = sum(self.style_sim(q, neg) for q, neg in zip(query, negative_keys))/len(query)
            return loss_pos - loss_neg*0.1
        else:
            query = [item[target_tensor == 0, ...] for item in input_tensor]
            positive_key = [item[target_tensor == 1, ...] for item in input_tensor]
            return sum(self.style_sim(q, pos) for q, pos in zip(query, positive_key))/len(query)
