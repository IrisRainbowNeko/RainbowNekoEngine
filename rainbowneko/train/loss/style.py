import torch
from torch import nn
from torch.nn import functional as F

from .base import LossContainer
from rainbowneko.models.layers import StyleSimilarity
import math


class StyleLoss(nn.Module):
    def __init__(self, eps=1e-4, reduction='mean'):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.eps = eps
        self.reduction = reduction
        self.style_sim = StyleSimilarity()

    _key_map = {'input_tensor': 'pred.feat', 'target_tensor': 'pred.label'}
    def forward(self, input_tensor, target_tensor):
        # input_tensor: list([B,C,H,W])

        if 2 in target_tensor:  # triplet style loss
            query = [item[target_tensor == 0, ...] for item in input_tensor]
            positive_key = [item[target_tensor == 1, ...] for item in input_tensor]
            negative_keys = [item[target_tensor == 2, ...] for item in input_tensor]

            return sum(self.style_sim(q, pos) + F.relu(-self.style_sim(q, neg)+5) for q, pos, neg in zip(query, positive_key, negative_keys))/len(query)
            
            #loss_pos = sum(self.style_sim(q, pos) for q, pos in zip(query, positive_key))/len(query)
            #loss_neg = sum(self.style_sim(q, neg) for q, neg in zip(query, negative_keys))/len(query)
            #return loss_pos + F.relu(-loss_neg+5)
        else:
            query = [item[target_tensor == 0, ...] for item in input_tensor]
            positive_key = [item[target_tensor == 1, ...] for item in input_tensor]
            return sum(self.style_sim(q, pos) for q, pos in zip(query, positive_key))/len(query)
