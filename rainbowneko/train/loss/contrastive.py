import torch
from torch import nn
from torch.nn import functional as F
from einops import repeat

from .base import LossContainer


class MLCEImageLoss(LossContainer):
    def __init__(self, weight=1.0, eps=1e-4, reduction='mean'):
        super().__init__(None, weight=weight)
        self.ce_loss = nn.CrossEntropyLoss()
        self.eps = eps
        self.reduction = reduction

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

        if 'label_neg' in target:  # 显式给出pos和neg
            query = input_tensor[target_tensor == 0, :]
            positive_key = input_tensor[target_tensor == 1, :]
            negative_keys = input_tensor[target_tensor == 2, :]
        else:
            pos = target_tensor[0] == target_tensor
            query, positive_key = input_tensor[pos, :].chunk(2)
            if query.shape[0] != positive_key.shape[0]:
                query = query[:-1]
            negative_keys = input_tensor[~pos, :]

        return self.info_nce(query, positive_key, negative_keys, T=self.temperature) * self.alpha

    def info_nce(self, query, pos, neg, T=0.1):
        '''
        :param query: [Nq, C]
        :param pos: [Np, C]
        :param neg: [Nn, C]
        '''
        query = F.normalize(query, dim=-1)
        pos = F.normalize(pos, dim=-1)
        neg = F.normalize(neg, dim=-1)

        sim_pos = torch.mm(query, pos.transpose(0, 1))  # [Nq, Np]
        sim_neg = torch.mm(query, neg.transpose(0, 1))  # [Nq, Nn]
        logits = torch.cat([sim_pos, sim_neg], dim=1)
        labels = torch.cat([torch.ones(sim_pos.shape[-1]), torch.zeros(sim_neg.shape[-1])]).to(query.device,
                                                                                               dtype=query.dtype)
        labels = repeat(labels, 'n -> Nq n', Nq=len(query))
        return F.cross_entropy(logits / T, labels, reduction=self.reduction)


class NoisyInfoNCELoss(InfoNCELoss):
    def info_nce(self, query, pos, neg, T=0.1):
        '''
        :param query: [Nq, C]
        :param pos: [Np, C]
        :param neg: [Nn, C]
        '''
        query = F.normalize(query, dim=-1)
        pos = F.normalize(pos, dim=-1)
        neg = F.normalize(neg, dim=-1)

        sim_pos = torch.mm(query, pos.transpose(0, 1))  # [Nq, Np]
        sim_neg = torch.mm(query, neg.transpose(0, 1))  # [Nq, Nn]
        logits = torch.cat([sim_pos, sim_neg], dim=1)
        labels = torch.cat([torch.ones(sim_pos.shape[-1]), torch.zeros(sim_neg.shape[-1])]).to(query.device,
                                                                                               dtype=query.dtype)
        labels = repeat(labels, 'n -> Nq n', Nq=len(query))

        # RCE+NCE https://arxiv.org/pdf/2006.13554.pdf
        rce = self.reverse_cross_entropy(logits / T, labels)
        nce = self.normalized_cross_entropy(logits / T, labels)
        return rce + nce

    def reverse_cross_entropy(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        labels = torch.clamp(labels, min=1e-4, max=1.0)
        rce = -torch.sum(pred * torch.log(labels), dim=1)
        return rce

    def normalized_cross_entropy(self, pred, labels):
        pred = F.log_softmax(pred, dim=1)
        nce = -torch.sum(labels * pred, dim=1) / (-pred.sum(dim=1))
        return nce
