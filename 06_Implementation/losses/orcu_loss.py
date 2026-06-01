"""
ORCU Loss: Ordinal Regression for Calibration and Unimodality.

Based on Kim et al. (MedIA 2025).
L_ORCU = L_SCE (SORD soft-encoded CE) + lambda_reg * L_REG (log-barrier ordinal regularization)

The log-barrier penalty enforces unimodal ordering of logits:
  - For adjacent classes (k, k+1) where k < target: enforce z_k < z_{k+1}
  - For adjacent classes where k >= target: enforce z_k > z_{k+1}

Uses a smooth log-barrier function with linear tail (t parameter controls barrier width),
providing a differentiable and well-behaved regularization surface compared to hinge loss.
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


def log_barrier_penalty(r_k, t):
    """Smooth log-barrier penalty for ordinal constraint enforcement.

    For enforcing r_k < 0 (z_k < z_{k+1}):
        - When r_k << 0 (safely negative): penalty ≈ 0
        - When r_k approaches 0 from below: log-barrier activates
        - When r_k >= 0 (violation): linear penalty tail

    Args:
        r_k: difference vector z_k - z_{k+1}, shape (B,)
        t: barrier width parameter (larger = tighter barrier)

    Returns:
        penalty: shape (B,)
    """
    threshold = -1.0 / (t ** 2)

    # Log-barrier zone: r_k < threshold (close to or crossing boundary)
    log_zone = -(1.0 / t) * torch.log(-r_k)

    # Linear tail zone: r_k >= threshold (safely satisfied)
    linear_zone = t * r_k - (1.0 / t) * torch.log(1.0 / (t ** 2)) + t

    return torch.where(r_k <= threshold, log_zone, linear_zone)


class ORCULoss(nn.Module):
    """ORCU loss: L_SCE + lambda_reg * L_REG.

    L_SCE: SORD soft-encoded cross-entropy
    L_REG: Log-barrier ordinal regularization (matches design doc, NOT hinge)
    """

    def __init__(self, num_classes: int = 4, t: float = 3.0, lambda_reg: float = 0.05, use_log_barrier: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.t = t
        self.lambda_reg = lambda_reg
        self.use_log_barrier = use_log_barrier  # False for A6 ablation (hinge mode)

    def forward(self, z, targets):
        K = z.shape[-1]
        B = z.shape[0]

        # SORD soft-encoded cross-entropy
        y_sord = compute_sord_encoding(targets, K)
        log_probs = F.log_softmax(z, dim=-1)
        L_sce = -(y_sord * log_probs).sum(dim=-1).mean()

        # Ordinal regularization (log-barrier or hinge for A6 ablation)
        reg = torch.tensor(0.0, device=z.device)
        for k in range(K - 1):
            r_k = z[:, k] - z[:, k + 1]  # (B,)

            mask_below = (targets > k).float()      # want r_k < 0
            mask_at_above = (targets <= k).float()  # want r_k > 0

            if self.use_log_barrier:
                # Log-barrier: smooth, differentiable penalty
                penalty_below = log_barrier_penalty(r_k, self.t)
                penalty_above = log_barrier_penalty(-r_k, self.t)
            else:
                # Hinge: simple ReLU penalty (A6 ablation baseline)
                penalty_below = F.relu(r_k)
                penalty_above = F.relu(-r_k)

            reg += (mask_below * penalty_below).sum()
            reg += (mask_at_above * penalty_above).sum()

        L_reg = reg / (B * (K - 1))
        return L_sce + self.lambda_reg * L_reg


def check_unimodality(probs):
    """Check if predicted probability distribution is unimodal."""
    diff = torch.diff(probs, dim=-1)
    signs = torch.sign(diff)
    sign_changes = ((signs[:, 1:] != signs[:, :-1]) & (signs[:, 1:] != 0)).sum(dim=-1)
    return (sign_changes <= 1).float()
