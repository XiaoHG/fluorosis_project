"""
ORCU Loss: Ordinal Regression for Calibration and Unimodality.

Based on Kim et al. (MedIA 2025).
L_ORCU = L_SCE (SORD soft-encoded CE) + L_REG (log-barrier ordinal regularization)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_sord_encoding(targets, num_classes):
    """Soft ORDinal encoding: distributes probability mass across adjacent classes
    based on ordinal distance using squared distance metric."""
    device = targets.device
    classes = torch.arange(num_classes, device=device, dtype=torch.float32)
    targets_f = targets.float()
    dist = (targets_f.unsqueeze(-1) - classes.unsqueeze(0)) ** 2
    return F.softmax(-dist, dim=-1)


class ORCULoss(nn.Module):
    """ORCU loss: L_SCE + L_REG."""

    def __init__(self, num_classes: int = 4, t: float = 3.0):
        super().__init__()
        self.num_classes = num_classes
        self.t = t

    def forward(self, z, targets):
        K = z.shape[-1]
        B = z.shape[0]

        y_sord = compute_sord_encoding(targets, K)
        log_probs = F.log_softmax(z, dim=-1)
        L_sce = -(y_sord * log_probs).sum(dim=-1).mean()

        reg = torch.tensor(0.0, device=z.device)
        threshold = -1.0 / (self.t ** 2)

        for k in range(K - 1):
            r_k = z[:, k] - z[:, k + 1]
            mask_below = (targets > k).float()
            mask_at_above = (targets <= k).float()
            linear_val = self.t * r_k - (1.0 / self.t) * torch.log(
                torch.tensor(1.0 / (self.t ** 2) + 1e-8, device=z.device)
            ) + self.t
            penalty = torch.where(
                r_k <= threshold,
                -(1.0 / self.t) * torch.log(-r_k + 1e-8),
                linear_val,
            )
            reg += (mask_below * penalty).sum()
            reg += (mask_at_above * penalty).sum()

        L_reg = reg / (B * (K - 1))
        return L_sce + L_reg


def check_unimodality(probs):
    """Check if predicted probability distribution is unimodal."""
    diff = torch.diff(probs, dim=-1)
    signs = torch.sign(diff)
    sign_changes = ((signs[:, 1:] != signs[:, :-1]) & (signs[:, 1:] != 0)).sum(dim=-1)
    return (sign_changes <= 1).float()
