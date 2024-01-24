import torch
from torch import nn
import numpy as np

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
        #if self.training:
        logits_per_image = logits_per_image - torch.diag_embed(torch.diag(logits_per_image))

        return logits_per_image