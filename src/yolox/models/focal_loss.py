# losses/focal_loss.py
import torch
import torch.nn.functional as F
from torch import nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p = torch.sigmoid(inputs)
        p_t = p*targets + (1-p)*(1-targets)
        loss = bce * ((1 - p_t) ** self.gamma)
        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss = loss * alpha_factor
        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()
