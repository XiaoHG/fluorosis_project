"""
Evidence Head: replaces standard classifier head for EDL-based diagnosis.

Outputs Dirichlet concentration params α = softplus(z) + 1,
ensuring α_k > 1 for unimodal Dirichlet distribution.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidenceHead(nn.Module):
    """Single FC layer → logits z → Dirichlet α.

    Shared logits serve both EDL (z → α) and ORCU (z → SORD regularization).
    """

    def __init__(self, in_dim: int, num_classes: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, features):
        """Returns alpha (Dirichlet params) and z (logits for ORCU)."""
        features = self.dropout(features)
        z = self.fc(features)
        alpha = F.softplus(z) + 1.0
        return alpha, z

    def predict(self, features):
        """Inference mode: return prediction + uncertainty + evidence + belief."""
        alpha, z = self.forward(features)
        probs = alpha / alpha.sum(dim=-1, keepdim=True)
        pred = torch.argmax(probs, dim=-1)
        u = self.num_classes / alpha.sum(dim=-1)
        evidence = alpha - 1.0
        belief = evidence / evidence.sum(dim=-1, keepdim=True)
        return {
            "pred": pred,
            "probs": probs,
            "alpha": alpha,
            "evidence": evidence,
            "uncertainty": u,
            "belief": belief,
            "total_evidence": alpha.sum(dim=-1),
        }
