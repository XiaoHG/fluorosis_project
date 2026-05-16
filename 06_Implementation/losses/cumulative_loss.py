"""
Cumulative (Coral/CORN) Loss: K-1 binary cross-entropy.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CumulativeLoss(nn.Module):
    """K-1 binary classification losses with monotonic probability constraint."""

    def __init__(self, num_classes=4):
        super().__init__()
        self.K_minus_1 = num_classes - 1

    def forward(self, ordinal_logits, targets):
        """ordinal_logits: (B, K-1), targets: (B,) integer labels in [0, K-1]."""
        K = self.K_minus_1
        binary_targets = (targets.unsqueeze(-1) > torch.arange(K, device=targets.device)).float()
        return F.binary_cross_entropy_with_logits(ordinal_logits, binary_targets)
