import torch
from torch import nn
from torch.nn import functional as F

from .base import LossContainer
import math


class StyleLoss(LossContainer):
    def __init__(self, weight=1.0, eps=1e-4, reduction='mean'):
        super().__init__(None, weight=weight)
        self.ce_loss = nn.CrossEntropyLoss()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred, target):
        input_tensor = pred['feat']  # list([B,C,H,W])
        target_tensor = target['label']

        if 2 in target_tensor:  # triplet style loss
            query = [item[target_tensor == 0, ...] for item in input_tensor]
            positive_key = [item[target_tensor == 1, ...] for item in input_tensor]
            negative_keys = [item[target_tensor == 2, ...] for item in input_tensor]

            loss_pos = sum(self.style_loss(q, pos) for q, pos in zip(query, positive_key))/len(query)
            loss_neg = sum(self.style_loss(q, neg) for q, neg in zip(query, negative_keys))/len(query)
            return loss_pos - loss_neg*0.0001
        else:
            query = [item[target_tensor == 0, ...] for item in input_tensor]
            positive_key = [item[target_tensor == 1, ...] for item in input_tensor]
            return sum(self.style_loss(q, pos) for q, pos in zip(query, positive_key))/len(query)

    def style_loss(self, x1, x2):
        return F.mse_loss(self.gram_matrix(x1), self.gram_matrix(x2))

    @staticmethod
    def gram_matrix(input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.reshape(a * b, c * d).div(math.sqrt(a * b * c * d))  # resize F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G
