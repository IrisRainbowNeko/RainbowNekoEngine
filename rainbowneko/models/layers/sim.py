import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class BatchCosineSimilarity(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image_features):  # x: BxN
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * torch.mm(image_features, image_features.transpose(0, 1))
        # if self.training:
        logits_per_image = logits_per_image - torch.diag_embed(torch.diag(logits_per_image))

        return logits_per_image


class StyleSimilarity(nn.Module):
    def __init__(self, batch_mean=False):
        super().__init__()
        self.batch_mean = batch_mean

    def forward(self, feat1, feat2):  # feat1: [B,C,H,W]
        sim = F.mse_loss(self.gram_matrix(feat1), self.gram_matrix(feat2), reduction='none')
        if self.batch_mean:
            return sim.mean(dim=(1, 2))*1000
        else:
            return sim.mean()*1000

    @staticmethod
    def gram_matrix(x, should_normalize=True):
        (b, c, h, w) = x.size()
        features = x.view(b, c, h * w)
        if should_normalize:
            features = features / math.sqrt(c * h * w)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t)  # [B,C,C]
        return gram
