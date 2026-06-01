"""
Evidential Deep Learning (EDL) Loss.

Based on Sensoy et al. (NeurIPS 2018).
L_EDL = Type II MLE + KL annealing
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EDLLoss(nn.Module):
    """EDL loss with KL-divergence annealing."""

    def __init__(self, num_classes: int = 4, kl_lambda: float = 0.1, kl_anneal_cap: float = 0.5):
        super().__init__()
        self.num_classes = num_classes
        self.kl_lambda = kl_lambda
        self.kl_anneal_cap = kl_anneal_cap

    def forward(self, alpha, targets, epoch=None, total_epochs=None):
        K = alpha.shape[-1]
        # Support both hard labels (int tensor) and soft labels (float tensor)
        if targets.dim() == 1:
            y = F.one_hot(targets, num_classes=K).float()
        else:
            # Soft labels: already probability vectors, shape (B, K)
            y = targets.float()

        S = alpha.sum(dim=-1, keepdim=True)
        ll = (y * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=-1)

        alpha_tilde = y + (1 - y) * alpha
        S_tilde = alpha_tilde.sum(dim=-1, keepdim=True)
        kl = (
            torch.lgamma(S_tilde).squeeze()
            - torch.lgamma(S).squeeze()
            - torch.lgamma(alpha_tilde).sum(dim=-1)
            + torch.lgamma(alpha).sum(dim=-1)
            + ((alpha_tilde - alpha) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde))).sum(dim=-1)
        )

        if epoch is not None and total_epochs is not None:
            anneal = min(self.kl_anneal_cap, epoch / (total_epochs * 0.3))
        else:
            anneal = 1.0

        return ll.mean() + self.kl_lambda * anneal * kl.mean()


def edl_metrics(alpha, targets):
    """Compute EDL-specific evaluation metrics."""
    K = alpha.shape[-1]
    probs = alpha / alpha.sum(dim=-1, keepdim=True)
    preds = torch.argmax(probs, dim=-1)
    u = K / alpha.sum(dim=-1)
    acc = (preds == targets).float().mean()
    return {
        "acc": acc.item(),
        "mean_uncertainty": u.mean().item(),
        "mean_total_evidence": (alpha - 1.0).sum(dim=-1).mean().item(),
    }
