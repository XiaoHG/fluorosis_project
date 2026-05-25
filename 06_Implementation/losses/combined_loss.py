"""
Three-stage training loss wrapper.

Stage 1 (CE warmup): standard cross-entropy on logits z
Stage 2 (EDL only):  evidential loss on Dirichlet alpha
Stage 3 (Joint):     L_EDL + lambda * L_ORCU with linear lambda annealing
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.edl_loss import EDLLoss
from losses.orcu_loss import ORCULoss


class CombinedLoss(nn.Module):
    """Three-stage loss: CE warmup -> EDL only -> EDL + ORCU."""

    def __init__(
        self,
        num_classes: int = 4,
        lambda_orcu: float = 0.05,
        lambda_kl: float = 0.02,
        kl_anneal_cap: float = 0.5,
        orcu_t: float = 3.0,
        orcu_lambda_reg: float = 0.005,
        stage_1_epochs: int = 3,
        stage_2_epochs: int = 10,
        total_epochs: int = 50,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_orcu = lambda_orcu
        self.stage_1_epochs = stage_1_epochs
        self.stage_2_epochs = stage_2_epochs
        self.total_epochs = total_epochs

        self.edl_loss = EDLLoss(num_classes=num_classes, kl_lambda=lambda_kl, kl_anneal_cap=kl_anneal_cap)
        self.orcu_loss = ORCULoss(num_classes=num_classes, t=orcu_t, lambda_reg=orcu_lambda_reg)

    def get_stage(self, epoch):
        if epoch < self.stage_1_epochs:
            return 1
        elif epoch < self.stage_1_epochs + self.stage_2_epochs:
            return 2
        else:
            return 3

    def forward(self, alpha, z, targets, epoch):
        stage = self.get_stage(epoch)

        if stage == 1:
            log_probs = F.log_softmax(z, dim=-1)
            loss = F.nll_loss(log_probs, targets)
            return loss, {"stage": 1, "L_ce": loss.item()}

        elif stage == 2:
            loss = self.edl_loss(alpha, targets, epoch, self.total_epochs)
            return loss, {"stage": 2, "L_edl": loss.item()}

        else:
            L_edl = self.edl_loss(alpha, targets, epoch, self.total_epochs)
            L_orcu = self.orcu_loss(z, targets)

            joint_start = self.stage_1_epochs + self.stage_2_epochs
            joint_total = self.total_epochs - joint_start
            progress = min(1.0, (epoch - joint_start) / max(1, joint_total * 0.5))
            lam = self.lambda_orcu * progress

            loss = L_edl + lam * L_orcu
            return loss, {
                "stage": 3,
                "L_edl": L_edl.item(),
                "L_orcu": L_orcu.item(),
                "lambda": lam,
                "L_total": loss.item(),
            }
