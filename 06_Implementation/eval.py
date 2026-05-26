"""
Evaluation metrics for fluorosis diagnosis.

Metrics: Acc, macro-F1, QWK, ECE, SCE, %Unimodal, U-ECE, AUROC(u).
Handles both EDL (alpha+evidence) and non-EDL (cumulative/SORD/CE) modes.
"""
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import cohen_kappa_score, roc_auc_score


def compute_metrics(alpha, z, targets, num_bins=15, temperature=1.0):
    """Compute all evaluation metrics.

    Args:
        alpha: Dirichlet concentration params (N, K), or None for non-EDL modes
        z: logits (N, K) — always K-class logits for eval compatibility
        targets: integer labels (N,)
        num_bins: bins for ECE / U-ECE
        temperature: temperature scaling for confidence calibration (default 1.0 = no scaling)
    """
    K = z.shape[-1]
    z_scaled = z / temperature
    probs = F.softmax(z_scaled, dim=-1)
    preds = torch.argmax(probs, dim=-1)

    # EDL-specific: uncertainty from Dirichlet concentration
    if alpha is not None:
        S = alpha.sum(dim=-1)
        u = K / S
        evidence = alpha - 1.0
    else:
        # Non-EDL mode: use entropy as proxy uncertainty
        u = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1) / torch.log(torch.tensor(K, dtype=torch.float32))
        evidence = torch.zeros_like(z)

    y_true = targets.cpu().numpy()
    y_pred = preds.cpu().numpy()
    y_prob = probs.detach().cpu().numpy()
    u_np = u.detach().cpu().numpy()
    evidence_np = evidence.detach().cpu().numpy()

    acc = (preds == targets).float().mean().item()

    from sklearn.metrics import f1_score
    f1 = f1_score(y_true, y_pred, average="macro")
    w_f1 = f1_score(y_true, y_pred, average="weighted")
    per_class_f1 = f1_score(y_true, y_pred, average=None)

    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")

    ece = _compute_ece(y_prob, y_true, num_bins)
    sce = _compute_sce(y_prob, y_true, num_bins)

    pct_unimodal = _compute_unimodal_ratio(probs)

    u_ece = _compute_u_ece(u_np, y_prob, y_true, num_bins)

    errors = (y_pred != y_true).astype(np.float32)
    try:
        auroc_u = roc_auc_score(errors, u_np)
    except ValueError:
        auroc_u = 0.5

    class_evidence = {}
    if alpha is not None:
        for k in range(K):
            mask = y_true == k
            if mask.sum() > 0:
                class_evidence[f"evidence_class_{k}"] = evidence_np[mask, k].mean().item()

    # Confusion matrix
    cm = np.zeros((K, K), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    return {
        "acc": acc,
        "macro_f1": f1,
        "weighted_f1": w_f1,
        "per_class_f1": per_class_f1.tolist(),
        "qwk": qwk,
        "ece": ece,
        "sce": sce,
        "pct_unimodal": pct_unimodal,
        "u_ece": u_ece,
        "auroc_u": auroc_u,
        "mean_uncertainty": u_np.mean().item(),
        "mean_evidence": evidence_np.sum(axis=1).mean().item() if alpha is not None else 0.0,
        "confusion_matrix": cm.tolist(),
        **class_evidence,
    }


def _compute_ece(probs, labels, num_bins):
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
    n = len(labels)
    preds = probs.argmax(axis=1)
    correct = (preds == labels).astype(np.float32)
    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    u_ece = 0.0
    for i in range(num_bins):
        in_bin = (u > bin_boundaries[i]) & (u <= bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            bin_acc = correct[in_bin].mean()
            bin_conf = 1.0 - u[in_bin].mean()
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
        if alpha is not None:
            all_alpha.append(alpha.cpu())
        all_z.append(z.cpu())
        all_targets.append(targets.cpu())

    alpha = torch.cat(all_alpha, dim=0) if all_alpha else None
    z = torch.cat(all_z, dim=0)
    targets = torch.cat(all_targets, dim=0)

    return compute_metrics(alpha, z, targets)


@torch.no_grad()
def export_predictions(model, dataloader, device="cuda"):
    """Collect per-sample predictions for figure generation.

    Returns dict: y_true, y_pred, prob, alpha (EDL only), u, evidence (EDL only),
    entropy, logits. All values are numpy arrays.
    """
    model.eval()
    all_alpha, all_z, all_targets = [], [], []

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)
        alpha, z = model(images)
        if alpha is not None:
            all_alpha.append(alpha.cpu())
        all_z.append(z.cpu())
        all_targets.append(targets.cpu())

    alpha_t = torch.cat(all_alpha, dim=0) if all_alpha else None
    logits = torch.cat(all_z, dim=0)
    targets = torch.cat(all_targets, dim=0)
    K = logits.shape[-1]
    probs = F.softmax(logits, dim=-1)
    preds = torch.argmax(probs, dim=-1)

    if alpha_t is not None:
        S = alpha_t.sum(dim=-1)
        u = (K / S).cpu().numpy()
        evidence = (alpha_t - 1.0).cpu().numpy()
        alpha_np = alpha_t.cpu().numpy()
    else:
        u = (-torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
             / torch.log(torch.tensor(K, dtype=torch.float32))).cpu().numpy()
        evidence = None
        alpha_np = None

    entropy = (-torch.sum(probs * torch.log(probs + 1e-8), dim=-1)).cpu().numpy()

    return {
        "y_true": targets.cpu().numpy().astype(np.int64),
        "y_pred": preds.cpu().numpy().astype(np.int64),
        "prob": probs.cpu().numpy().astype(np.float32),
        "alpha": alpha_np.astype(np.float32) if alpha_np is not None else None,
        "u": u.astype(np.float32),
        "evidence": evidence.astype(np.float32) if evidence is not None else None,
        "entropy": entropy.astype(np.float32),
        "logits": logits.cpu().numpy().astype(np.float32),
    }


def save_predictions(data, filepath, mode="edl", task="df", seed=42, fold=-1):
    """Save per-sample predictions as .npz with metadata."""
    save_dict = {k: v for k, v in data.items() if v is not None}
    save_dict["_mode"] = mode
    save_dict["_task"] = task
    save_dict["_seed"] = seed
    save_dict["_fold"] = fold
    np.savez_compressed(filepath, **save_dict)
