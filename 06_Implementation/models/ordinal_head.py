"""
Cumulative Ordinal Head: Coral-style K-1 binary head with monotonic bias.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CumulativeHead(nn.Module):
    """K-1 binary head with non-increasing bias (b_0 >= b_1 >= ... >= b_{K-2})."""

    def __init__(self, in_dim, num_classes=4, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.K_minus_1 = num_classes - 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_dim, self.K_minus_1)
        self.register_parameter('raw_bias', nn.Parameter(torch.zeros(self.K_minus_1)))

    def forward(self, x):
        x = self.dropout(x)
        logits = self.fc(x)                         # (B, K-1)
        b = -torch.cumsum(F.softplus(self.raw_bias), dim=0)  # b_0 >= b_1 >= ...
        return logits + b
