"""
Evaluation metrics for EDL+ORCU fluorosis diagnosis.

Implements all 7 metrics from experiment plan:
  Acc, macro-F1, QWK, ECE, SCE, %Unimodal, U-ECE
Plus AUROC(u vs error) for uncertainty discriminative power.
"""
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import cohen_kappa_score, roc_auc_score


def compute_metrics(alpha, z, targets, num_bins=15):
    """Compute all evaluation metrics from model outputs.

    Args:
        alpha: Dirichlet concentration params (N, K)
        z: logits (N, K)
        targets: integer labels (N,)
        num_bins: bins for ECE / U-ECE

    Returns:
        dict with all metric values (float scalars)
    """
    K = alpha.shape[-1]
    probs = F.softmax(z, dim=-1)
    preds = torch.argmax(probs, dim=-1)
    evidence = alpha - 1.0
    S = alpha.sum(dim=-1)
    u = K / S  # uncertainty

    y_true = targets.cpu().numpy()
    y_pred = preds.cpu().numpy()
    y_prob = probs.detach().cpu().numpy()
    u_np = u.detach().cpu().numpy()
    evidence_np = evidence.detach().cpu().numpy()

    # ---- Accuracy ----
    acc = (preds == targets).float().mean().item()

    # ---- Macro F1 ----
    from sklearn.metrics import f1_score
    f1 = f1_score(y_true, y_pred, average="macro")

    # ---- QWK (Quadratic Weighted Kappa) ----
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")

    # ---- ECE (Expected Calibration Error) ----
    ece = _compute_ece(y_prob, y_true, num_bins)

    # ---- SCE (Static Calibration Error) ----
    sce = _compute_sce(y_prob, y_true, num_bins)

    # ---- %Unimodal ----
    pct_unimodal = _compute_unimodal_ratio(probs)

    # ---- U-ECE (Uncertainty-ECE) ----
    u_ece = _compute_u_ece(u_np, y_prob, y_true, num_bins)

    # ---- AUROC: uncertainty vs error ----
    errors = (y_pred != y_true).astype(np.float32)
    try:
        auroc_u = roc_auc_score(errors, u_np)
    except ValueError:
        auroc_u = 0.5

    # ---- Per-class evidence (evidence for ground-truth class k) ----
    class_evidence = {}
    for k in range(K):
        mask = y_true == k
        if mask.sum() > 0:
            class_evidence[f"evidence_class_{k}"] = evidence_np[mask, k].mean().item()

    return {
        "acc": acc,
        "macro_f1": f1,
        "qwk": qwk,
        "ece": ece,
        "sce": sce,
        "pct_unimodal": pct_unimodal,
        "u_ece": u_ece,
        "auroc_u": auroc_u,
        "mean_uncertainty": u_np.mean().item(),
        "mean_evidence": evidence_np.sum(axis=1).mean().item(),
        **class_evidence,
    }


def _compute_ece(probs, labels, num_bins):
    """Expected Calibration Error."""
    n = len(labels)
    confs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    accs = (preds == labels).astype(np.float32)

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    for i in range(num_bins):
        in_bin = (confs > bin_boundaries[i]) & (confs <= bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            bin_acc = accs[in_bin].mean()
            bin_conf = confs[in_bin].mean()
            ece += (in_bin.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def _compute_sce(probs, labels, num_bins):
    """Static Calibration Error — per-class ECE averaged."""
    K = probs.shape[1]
    n = len(labels)
    sce = 0.0
    for k in range(K):
        bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
        for i in range(num_bins):
            in_bin = (probs[:, k] > bin_boundaries[i]) & (probs[:, k] <= bin_boundaries[i + 1])
            if in_bin.sum() > 0:
                bin_prob = probs[in_bin, k].mean()
                bin_freq = (labels[in_bin] == k).mean()
                sce += (in_bin.sum() / (n * K)) * abs(bin_prob - bin_freq)
    return float(sce)


def _compute_unimodal_ratio(probs):
    """Fraction of samples with unimodal predicted distribution."""
    K = probs.shape[-1]

    def is_unimodal(p):
        idx = p.argmax().item()
        for i in range(idx):
            if p[i] > p[i + 1]:
                return False
        for i in range(idx, K - 1):
            if p[i] < p[i + 1]:
                return False
        return True

    count = sum(1 for i in range(probs.shape[0]) if is_unimodal(probs[i]))
    return count / probs.shape[0]


def _compute_u_ece(u, probs, labels, num_bins):
    """Uncertainty-ECE: bins predictions by uncertainty, measures calibration."""
    n = len(labels)
    preds = probs.argmax(axis=1)
    correct = (preds == labels).astype(np.float32)

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    u_ece = 0.0
    for i in range(num_bins):
        in_bin = (u > bin_boundaries[i]) & (u <= bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            bin_acc = correct[in_bin].mean()
            bin_conf = 1.0 - u[in_bin].mean()  # confidence ≈ 1 - uncertainty
            u_ece += (in_bin.sum() / n) * abs(bin_acc - bin_conf)
    return float(u_ece)


@torch.no_grad()
def evaluate(model, dataloader, device="cuda"):
    """Run full evaluation over a dataloader."""
    model.eval()
    all_alpha = []
    all_z = []
    all_targets = []

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)
        alpha, z = model(images)
        all_alpha.append(alpha.cpu())
        all_z.append(z.cpu())
        all_targets.append(targets.cpu())

    alpha = torch.cat(all_alpha, dim=0)
    z = torch.cat(all_z, dim=0)
    targets = torch.cat(all_targets, dim=0)

    return compute_metrics(alpha, z, targets)
