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

# Copy from https://github.com/RElbers/info-nce-pytorch/blob/main/info_nce/__init__.py
class InfoNCELoss(LossContainer):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113

    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.

    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.

    Returns:
         Value of the InfoNCE Loss.

     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, weight=1.0, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__(None, weight=weight)
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, pred, target):
        input_tensor = pred['feat'][0]  # [B,C]
        target_tensor = target['label']

        if 'label_neg' in target: # 显式给出pos和neg
            query = input_tensor[target_tensor==0, :]
            positive_key = input_tensor[target_tensor==1, :]
            negative_keys = input_tensor[target_tensor==2, :]
        else:
            pos = target_tensor[0]==target_tensor
            query, positive_key = input_tensor[pos, :].chunk(2)
            if query.shape[0] != positive_key.shape[0]:
                query = query[:-1]
            negative_keys = input_tensor[~pos, :]

        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode) * self.alpha


def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]