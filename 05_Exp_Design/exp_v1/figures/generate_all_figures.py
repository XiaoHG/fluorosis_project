"""
Generate all paper figures from v6 per-sample .npz predictions.

Input: Kaggle_Setup/kaggle_train_v6-exp/*.npz (14 files, 60 test samples each)
Output: 05_Exp_Design/figures/F1-F10.pdf + .png (300 DPI, vector PDF)

Each .npz contains: logits(N,4), probabilities(N,4), predicted(N,), targets(N,),
  evidence(N,4), uncertainty(N,), alpha(N,4) [EDL modes only]
"""

import numpy as np
import os, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── Config ──────────────────────────────────────────────────
DATA_DIR = Path("/Volumes/KINGSTON/Obsibian/Fluorosis/Fluorosis_DL_Paper/Kaggle_Setup/kaggle_train_v6-exp")
OUT_DIR = Path("/Volumes/KINGSTON/Obsibian/Fluorosis/Fluorosis_DL_Paper/05_Exp_Design/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}
plt.rcParams.update(STYLE)

CLASS_NAMES = ["Normal", "Mild", "Moderate", "Severe"]
MODE_NAMES = {"ce": "CE", "cumulative": "Cumulative", "sord": "SORD",
              "edl": "EDL", "edl_orcu": "EDL+ORCU"}
MODE_COLORS = {"ce": "#4472C4", "cumulative": "#ED7D31", "sord": "#A5A5A5",
               "edl": "#70AD47", "edl_orcu": "#FFC000"}
MODE_MARKERS = {"ce": "o", "cumulative": "s", "sord": "D", "edl": "^", "edl_orcu": "v"}

# v2.2 reference results (for paper tables)
V22 = {
    "ce":        {"acc": 0.8167, "f1": 0.8085, "qwk": 0.9329, "ece": 0.1213, "unim": 0.80},
    "cumulative":{"acc": 0.6833, "f1": 0.6574, "qwk": 0.8185, "ece": 0.1719, "unim": 0.65},
    "sord":      {"acc": 0.7333, "f1": 0.7343, "qwk": 0.8999, "ece": 0.1509, "unim": 0.97},
    "edl":       {"acc": 0.8333, "f1": 0.8278, "qwk": 0.9376, "ece": 0.0719, "unim": 0.70},
    "edl_orcu":  {"acc": 0.7000, "f1": 0.6948, "qwk": 0.8724, "ece": 0.1960, "unim": 0.78},
}

# ── Data Loading ────────────────────────────────────────────
def load_npz(name):
    d = np.load(DATA_DIR / name)
    return {k: d[k].astype(np.float32) for k in d.keys()}

EXP1 = {
    "ce": load_npz("exp1_ce_preds.npz"),
    "cumulative": load_npz("exp1_cumulative_preds.npz"),
    "sord": load_npz("exp1_sord_preds.npz"),
    "edl": load_npz("exp1_edl_preds.npz"),
    "edl_orcu": load_npz("exp1_edl_orcu_preds.npz"),
}

SWEEP_FILES = [
    ("exp2_sweep_lam0.1_reg0.005_preds.npz", 0.1, 0.005),
    ("exp2_sweep_lam0.1_reg0.01_preds.npz", 0.1, 0.01),
    ("exp2_sweep_lam0.1_reg0.02_preds.npz", 0.1, 0.02),
    ("exp2_sweep_lam0.3_reg0.005_preds.npz", 0.3, 0.005),
    ("exp2_sweep_lam0.3_reg0.01_preds.npz", 0.3, 0.01),
    ("exp2_sweep_lam0.3_reg0.02_preds.npz", 0.3, 0.02),
    ("exp2_sweep_lam0.5_reg0.005_preds.npz", 0.5, 0.005),
    ("exp2_sweep_lam0.5_reg0.01_preds.npz", 0.5, 0.01),
    ("exp2_sweep_lam0.5_reg0.02_preds.npz", 0.5, 0.02),
]

sweep_data = {}
for fname, lam_o, lam_r in SWEEP_FILES:
    sweep_data[(lam_o, lam_r)] = load_npz(fname)


# ── Metric computation from per-sample data ─────────────────
def compute_metrics(d):
    probs = d["probabilities"].astype(np.float64)
    preds = d["predicted"].astype(np.int64)
    targets = d["targets"].astype(np.int64)
    N, K = probs.shape

    acc = (preds == targets).mean()
    from sklearn.metrics import f1_score, cohen_kappa_score
    f1 = f1_score(targets, preds, average="macro")
    qwk = cohen_kappa_score(targets, preds, weights="quadratic")

    confs = probs.max(axis=1)
    accs_bin = (preds == targets).astype(np.float64)
    n_bins = 15
    bin_bounds = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confs > bin_bounds[i]) & (confs <= bin_bounds[i + 1])
        if mask.sum() > 0:
            ece += (mask.sum() / N) * abs(accs_bin[mask].mean() - confs[mask].mean())
    ece = float(ece)

    cm = np.zeros((K, K), dtype=int)
    for t, p in zip(targets, preds):
        cm[t, p] += 1

    def is_unimodal(p):
        idx = p.argmax()
        for i in range(idx):
            if p[i] > p[i + 1]: return False
        for i in range(idx, K - 1):
            if p[i] < p[i + 1]: return False
        return True
    pct_unimodal = sum(1 for i in range(N) if is_unimodal(probs[i])) / N

    return {"acc": acc, "f1": f1, "qwk": qwk, "ece": ece,
            "cm": cm, "unim": pct_unimodal, "probs": probs,
            "preds": preds, "targets": targets, "confs": confs}

metrics = {mode: compute_metrics(d) for mode, d in EXP1.items()}
sweep_metrics = {k: compute_metrics(d) for k, d in sweep_data.items()}

print("Metrics computed.")
for m, v in metrics.items():
    print(f"  {m:12s}: Acc={v['acc']:.4f} F1={v['f1']:.4f} QWK={v['qwk']:.4f} ECE={v['ece']:.4f} Unim={v['unim']:.1%}")


# ═══════════════════════════════════════════════════════════════
# F1: Confusion Matrices (2x3 grid, 5 modes)
# ═══════════════════════════════════════════════════════════════
def fig_f1():
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    modes = ["ce", "cumulative", "sord", "edl", "edl_orcu"]
    for idx, mode in enumerate(modes):
        ax = axes[idx]
        cm = metrics[mode]["cm"]
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                       color="white" if cm_norm[i, j] > 0.5 else "black")
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_title(f"{MODE_NAMES[mode]} (Acc={metrics[mode]['acc']:.1%})", fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    axes[5].set_visible(False)
    plt.suptitle("Figure 1: Confusion Matrices — DF Fluorosis (ViT-Base, N=60)", fontweight="bold", y=1.01)
    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        fig.savefig(OUT_DIR / f"F1_confusion_matrices.{fmt}", format=fmt)
    plt.close()
    print("  F1 done.")

# ═══════════════════════════════════════════════════════════════
# F2: Reliability Diagrams (ECE per mode)
# ═══════════════════════════════════════════════════════════════
def fig_f2():
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    modes = ["ce", "cumulative", "sord", "edl", "edl_orcu"]
    for idx, mode in enumerate(modes):
        ax = axes[idx]
        probs = metrics[mode]["probs"]
        targets = metrics[mode]["targets"]
        confs = probs.max(axis=1)
        accs = (probs.argmax(axis=1) == targets).astype(float)
        nb = 10
        bb = np.linspace(0, 1, nb + 1)
        centers, bin_accs, bin_confs = [], [], []
        for i in range(nb):
            m = (confs > bb[i]) & (confs <= bb[i + 1])
            if m.sum() >= 2:
                centers.append((bb[i] + bb[i+1]) / 2)
                bin_accs.append(accs[m].mean())
                bin_confs.append(confs[m].mean())
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, lw=0.8)
        ax.bar(centers, bin_accs, width=0.08, alpha=0.7, color=MODE_COLORS[mode], label="Accuracy")
        ax.plot(centers, bin_confs, "o-", color="darkred", ms=4, lw=1.5, label="Confidence")
        ece = metrics[mode]["ece"]
        ax.text(0.95, 0.08, f"ECE={ece:.3f}", transform=ax.transAxes, ha="right",
                fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_title(MODE_NAMES[mode], fontweight="bold")
        ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
        if idx == 0: ax.legend(fontsize=7, loc="upper left")
    axes[5].set_visible(False)
    plt.suptitle("Figure 2: Reliability Diagrams", fontweight="bold", y=1.01)
    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        fig.savefig(OUT_DIR / f"F2_reliability.{fmt}", format=fmt)
    plt.close()
    print("  F2 done.")

# ═══════════════════════════════════════════════════════════════
# F3: Uncertainty Analysis
# ═══════════════════════════════════════════════════════════════
def fig_f3():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # (a) Uncertainty histogram
    ax = axes[0]
    for mode in ["ce", "sord", "edl"]:
        u = EXP1[mode]["uncertainty"].astype(np.float32)
        ax.hist(u, bins=15, alpha=0.5, color=MODE_COLORS[mode], label=MODE_NAMES[mode], density=True)
    ax.set_xlabel("Uncertainty"); ax.set_ylabel("Density")
    ax.set_title("(a) Uncertainty Distribution"); ax.legend(fontsize=8)

    # (b) Uncertainty vs correctness scatter
    ax = axes[1]
    for mode in ["ce", "sord", "edl"]:
        u = EXP1[mode]["uncertainty"].astype(np.float32)
        correct = (metrics[mode]["preds"] == metrics[mode]["targets"])
        for ok, mk, al in [(True, "o", 0.7), (False, "x", 0.4)]:
            mask = correct == ok
            if mask.sum():
                ax.scatter(np.full(mask.sum(), list(MODE_NAMES.keys()).index(mode)),
                          u[mask], alpha=al, color=MODE_COLORS[mode], marker=mk, s=25, ec="none")
    ax.set_xticks(range(3))
    ax.set_xticklabels([MODE_NAMES[m] for m in ["ce", "sord", "edl"]])
    ax.set_ylabel("Uncertainty"); ax.set_title("(b) Uncertainty by Correctness")

    # (c) AUROC(u) error detection
    ax = axes[2]
    from sklearn.metrics import roc_curve, auc
    modes_all = ["ce", "cumulative", "sord", "edl", "edl_orcu"]
    for mode in modes_all:
        u = EXP1[mode]["uncertainty"].astype(np.float32)
        errors = (metrics[mode]["preds"] != metrics[mode]["targets"]).astype(int)
        if errors.sum() > 0 and (errors == 0).sum() > 0:
            fpr, tpr, _ = roc_curve(errors, u)
            ax.plot(fpr, tpr, color=MODE_COLORS[mode], lw=1.5,
                   label=f"{MODE_NAMES[mode]} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, lw=0.8)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("(c) Error Detection ROC"); ax.legend(fontsize=7, loc="lower right")

    plt.suptitle("Figure 3: Uncertainty Analysis", fontweight="bold", y=1.02)
    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        fig.savefig(OUT_DIR / f"F3_uncertainty.{fmt}", format=fmt)
    plt.close()
    print("  F3 done.")

# ═══════════════════════════════════════════════════════════════
# F4: Evidence Analysis (EDL / EDL+ORCU)
# ═══════════════════════════════════════════════════════════════
def fig_f4():
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    class_colors = ["#2166AC", "#92C5DE", "#F4A582", "#B2182B"]

    for col, mode in enumerate(["edl", "edl_orcu"]):
        d = EXP1[mode]
        targets = d["targets"].astype(int)
        evidence = d["evidence"].astype(np.float32)
        uncertainty = d["uncertainty"].astype(np.float32)
        probs = d["probabilities"].astype(np.float32)
        total_ev = evidence.sum(axis=1)

        # Row 1: Evidence per true class
        ax = axes[0, col]
        for k in range(4):
            mask = targets == k
            if mask.sum() > 0:
                ax.bar(k, total_ev[mask].mean(), color=class_colors[k], alpha=0.8,
                      label=f"{CLASS_NAMES[k]}")
        ax.set_xlabel("True Class"); ax.set_ylabel("Mean Total Evidence")
        ax.set_title(f"{MODE_NAMES[mode]}: Evidence by Class")
        ax.set_xticks(range(4)); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
        ax.legend(fontsize=7)

        # Row 2: Uncertainty vs Evidence scatter
        ax = axes[1, col]
        correct = (d["predicted"].astype(int) == targets)
        ax.scatter(total_ev[correct], uncertainty[correct], c="#70AD47", alpha=0.6, s=30,
                  label="Correct", ec="none")
        ax.scatter(total_ev[~correct], uncertainty[~correct], c="#B2182B", alpha=0.6, s=30,
                  marker="x", label="Error", ec="none")
        ax.set_xlabel("Total Evidence"); ax.set_ylabel("Uncertainty")
        ax.set_title(f"{MODE_NAMES[mode]}: Uncertainty vs Evidence")
        ax.legend(fontsize=8)

    # EDL evidence component breakdown (col 2)
    ax = axes[0, 2]
    ev = EXP1["edl"]["evidence"].astype(np.float32)
    tg = EXP1["edl"]["targets"].astype(int)
    ev_breakdown = np.zeros((4, 4))
    for k in range(4):
        mask = tg == k
        if mask.sum() > 0:
            ev_breakdown[k] = ev[mask].mean(axis=0)
    x = np.arange(4); w = 0.2
    for k in range(4):
        ax.bar(x + k * w, ev_breakdown[:, k], w, label=f"Evidence[{k}]", alpha=0.8)
    ax.set_xlabel("True Class"); ax.set_ylabel("Mean Evidence")
    ax.set_title("EDL: Evidence Components")
    ax.set_xticks(x + w * 1.5); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.legend(fontsize=7, ncol=2)

    # Confidence by evidence level
    ax = axes[1, 2]
    ev = EXP1["edl"]["evidence"].astype(np.float32)
    total_ev = ev.sum(axis=1)
    probs = EXP1["edl"]["probabilities"].astype(np.float32)
    tg = EXP1["edl"]["targets"].astype(int)
    ev_bins = np.percentile(total_ev, [0, 33, 67, 100])
    for i in range(3):
        mask = (total_ev >= ev_bins[i]) & (total_ev < ev_bins[i+1])
        if mask.sum() > 0:
            acc = (probs[mask].argmax(axis=1) == tg[mask]).mean()
            conf = probs[mask].max(axis=1).mean()
            ax.bar(i - 0.15, acc, 0.25, color="#4472C4", label="Accuracy" if i == 0 else "")
            ax.bar(i + 0.15, conf, 0.25, color="#ED7D31", label="Confidence" if i == 0 else "")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Low Ev.", "Med Ev.", "High Ev."])
    ax.set_ylabel("Value"); ax.set_title("Acc/Conf by Evidence Level")
    ax.legend(fontsize=8); ax.set_ylim(0, 1.05)

    plt.suptitle("Figure 4: Evidence Analysis — EDL Models", fontweight="bold", y=1.02)
    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        fig.savefig(OUT_DIR / f"F4_evidence.{fmt}", format=fmt)
    plt.close()
    print("  F4 done.")

# ═══════════════════════════════════════════════════════════════
# F5: Mode Comparison (pure v2.2 results)
# ═══════════════════════════════════════════════════════════════
def fig_f5():
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    modes = ["ce", "cumulative", "sord", "edl", "edl_orcu"]
    x = np.arange(len(modes))
    bar_colors = ["#4472C4", "#ED7D31", "#A5A5A5", "#70AD47", "#FFC000"]

    # (a) Acc + F1 + QWK grouped bar chart
    ax = axes[0, 0]
    w = 0.22
    metric_specs = [("Acc", "acc", 0.72), ("F1", "f1", 0.72), ("QWK", "qwk", 0.88)]
    for i, (label, key, baseline) in enumerate(metric_specs):
        vals = [V22[m][key] for m in modes]
        bars = ax.bar(x + (i-1)*w, vals, w, color=[plt.cm.Blues(0.5+i*0.2) for _ in modes],
                     alpha=0.85, edgecolor="black", lw=0.3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                   f"{val:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)
        # Dummy for legend
        ax.bar([0], [0], w, color=plt.cm.Blues(0.5+i*0.2), alpha=0.85, ec="black", lw=0.3, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_NAMES[m] for m in modes], rotation=15, fontsize=10)
    ax.set_ylabel("Score"); ax.set_title("(a) Performance Metrics (v2.2)")
    ax.legend(fontsize=8, loc="lower left"); ax.set_ylim(0.55, 1.0)

    # (b) ECE bar chart
    ax = axes[0, 1]
    ece_vals = [V22[m]["ece"] for m in modes]
    bars = ax.bar(x, ece_vals, 0.55, color=bar_colors, alpha=0.85, edgecolor="black", lw=0.5)
    for bar, val in zip(bars, ece_vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
               f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.axhline(y=0.05, color="gray", ls="--", alpha=0.5, lw=1, label="Target ECE=0.05")
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_NAMES[m] for m in modes], rotation=15, fontsize=10)
    ax.set_ylabel("ECE (lower is better)"); ax.set_title("(b) Expected Calibration Error")
    ax.legend(fontsize=8)

    # (c) Per-class recall (line plot)
    ax = axes[1, 0]
    for mode in modes:
        cm = metrics[mode]["cm"]
        pca = np.diag(cm) / cm.sum(axis=1)
        ax.plot(range(4), pca, marker=MODE_MARKERS[mode], color=MODE_COLORS[mode],
               lw=2, ms=8, label=MODE_NAMES[mode])
    ax.set_xticks(range(4)); ax.set_xticklabels(CLASS_NAMES, fontsize=10)
    ax.set_xlabel("True Class"); ax.set_ylabel("Recall"); ax.set_title("(c) Per-Class Recall")
    ax.legend(fontsize=8, loc="lower left"); ax.set_ylim(0, 1.05)

    # (d) %Unimodal + QWK combined bar
    ax = axes[1, 1]
    unim_vals = [V22[m]["unim"] * 100 for m in modes]
    qwk_vals = [V22[m]["qwk"] * 100 for m in modes]
    ax.bar(x-0.18, qwk_vals, 0.32, color="#4472C4", alpha=0.85, ec="black", lw=0.5, label="QWK × 100")
    ax.bar(x+0.18, unim_vals, 0.32, color="#70AD47", alpha=0.85, ec="black", lw=0.5, label="%Unimodal")
    for i in range(len(modes)):
        ax.text(x[i]-0.18, qwk_vals[i]+0.5, f"{qwk_vals[i]:.1f}", ha="center", fontsize=7)
        ax.text(x[i]+0.18, unim_vals[i]+0.5, f"{unim_vals[i]:.0f}%", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_NAMES[m] for m in modes], rotation=15, fontsize=10)
    ax.set_ylabel("Score / Percentage"); ax.set_title("(d) QWK & Ordinal Constraint")
    ax.legend(fontsize=8); ax.set_ylim(0, 110)

    plt.suptitle("Figure 5: Mode Comparison — DF Fluorosis (v2.2, ViT-Base)", fontweight="bold", y=1.01)
    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        fig.savefig(OUT_DIR / f"F5_mode_comparison.{fmt}", format=fmt)
    plt.close()
    print("  F5 done.")

# ═══════════════════════════════════════════════════════════════
# F6: Lambda Sweep Heatmap
# ═══════════════════════════════════════════════════════════════
def fig_f6():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    lam_o_vals = sorted(set(k[0] for k in sweep_metrics))
    lam_r_vals = sorted(set(k[1] for k in sweep_metrics))

    specs = [("acc", "Accuracy", "Blues", 0.5, 0.8),
             ("qwk", "QWK", "Greens", 0.82, 0.92),
             ("ece", "ECE (lower is better)", "Reds_r", 0.15, 0.28)]

    for idx, (key, title, cmap, vmin, vmax) in enumerate(specs):
        ax = axes[idx]
        mat = np.zeros((len(lam_r_vals), len(lam_o_vals)))
        for i, lr in enumerate(lam_r_vals):
            for j, lo in enumerate(lam_o_vals):
                mat[i, j] = sweep_metrics[(lo, lr)][key]
        im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
        for i in range(len(lam_r_vals)):
            for j in range(len(lam_o_vals)):
                val = mat[i, j]
                c = "white" if (val - vmin) / max(vmax - vmin, 1e-8) > 0.6 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9, color=c)
        ax.set_xticks(range(len(lam_o_vals)))
        ax.set_xticklabels([f"{v:.1f}" for v in lam_o_vals])
        ax.set_yticks(range(len(lam_r_vals)))
        ax.set_yticklabels([f"{v:.4f}" for v in lam_r_vals])
        ax.set_xlabel("λ_orcu"); ax.set_ylabel("λ_reg")
        ax.set_title(f"({chr(97+idx)}) {title}")
        plt.colorbar(im, ax=ax, shrink=0.85)

    plt.suptitle("Figure 6: Lambda Sweep — EDL+ORCU (50 epochs)", fontweight="bold", y=1.04)
    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        fig.savefig(OUT_DIR / f"F6_lambda_sweep.{fmt}", format=fmt)
    plt.close()
    print("  F6 done.")

# ═══════════════════════════════════════════════════════════════
# F7: Temperature Calibration
# ═══════════════════════════════════════════════════════════════
def fig_f7():
    temps = np.array([0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0])
    # These ECE values are from the v6 master summary output
    ece_vals = np.array([0.2845, 0.2619, 0.2610, 0.2285, 0.1933, 0.1196, 0.2492])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # (a) ECE vs T
    ax = axes[0]
    ax.plot(temps, ece_vals, "o-", color="#B2182B", lw=2, ms=8)
    best_i = np.argmin(ece_vals)
    ax.annotate(f"Best: T={temps[best_i]}\nECE={ece_vals[best_i]:.4f}",
               xy=(temps[best_i], ece_vals[best_i]),
               xytext=(temps[best_i]+1, ece_vals[best_i]+0.04),
               arrowprops=dict(arrowstyle="->", color="darkred"),
               fontsize=9, color="darkred", fontweight="bold")
    ax.axhline(y=0.05, color="gray", ls="--", alpha=0.5, label="Target ECE=0.05")
    ax.set_xlabel("Temperature T"); ax.set_ylabel("ECE")
    ax.set_title("(a) ECE vs Temperature"); ax.legend(fontsize=8)

    # (b) Reliability at different T
    ax = axes[1]
    probs = EXP1["edl"]["probabilities"].astype(np.float64)
    targets = EXP1["edl"]["targets"]
    for T in [1.0, 2.0, 3.0, 5.0]:
        logits = np.log(probs + 1e-8)
        sp = softmax_np(logits / T)
        confs = sp.max(axis=1)
        accs = (sp.argmax(axis=1) == targets).astype(float)
        nb = 8; bb = np.linspace(0, 1, nb+1)
        ctrs, baccs = [], []
        for i in range(nb):
            m = (confs > bb[i]) & (confs <= bb[i+1])
            if m.sum() >= 2:
                ctrs.append((bb[i]+bb[i+1])/2)
                baccs.append(accs[m].mean())
        ax.plot(ctrs, baccs, marker="o", ms=4, lw=1.5, label=f"T={T:.1f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, lw=0.8)
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
    ax.set_title("(b) Reliability at Different T"); ax.legend(fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # (c) Confidence distribution shift
    ax = axes[2]
    for T in [0.5, 1.0, 3.0, 5.0]:
        logits = np.log(probs + 1e-8)
        sp = softmax_np(logits / T)
        ax.hist(sp.max(axis=1), bins=12, alpha=0.4, label=f"T={T:.1f}", density=True)
    ax.set_xlabel("Confidence"); ax.set_ylabel("Density")
    ax.set_title("(c) Confidence Distribution Shift"); ax.legend(fontsize=8)

    plt.suptitle("Figure 7: Temperature Calibration — EDL", fontweight="bold", y=1.02)
    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        fig.savefig(OUT_DIR / f"F7_temperature.{fmt}", format=fmt)
    plt.close()
    print("  F7 done.")

def softmax_np(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

# ═══════════════════════════════════════════════════════════════
# F8: Prediction Distribution (ordinal pattern)
# ═══════════════════════════════════════════════════════════════
def fig_f8():
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    modes = ["ce", "cumulative", "sord", "edl", "edl_orcu"]
    for idx, mode in enumerate(modes):
        ax = axes[idx]
        d = EXP1[mode]
        probs = d["probabilities"].astype(np.float64)
        targets = d["targets"].astype(int)
        for k in range(4):
            mask = targets == k
            if mask.sum() > 0:
                ax.plot(range(4), probs[mask].mean(axis=0), marker="o", ms=6, lw=2,
                       label=f"True={CLASS_NAMES[k]}")
        ax.set_xticks(range(4)); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
        ax.set_xlabel("Predicted Class"); ax.set_ylabel("Mean Probability")
        ax.set_title(f"{MODE_NAMES[mode]} (Unim={metrics[mode]['unim']:.0%})", fontweight="bold")
        ax.legend(fontsize=7); ax.set_ylim(0, 1.05)
    axes[5].set_visible(False)
    plt.suptitle("Figure 8: Mean Predicted Probability per True Class", fontweight="bold", y=1.01)
    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        fig.savefig(OUT_DIR / f"F8_prediction_dist.{fmt}", format=fmt)
    plt.close()
    print("  F8 done.")

# ═══════════════════════════════════════════════════════════════
# F9: Summary Table + Radar (v2.2 results)
# ═══════════════════════════════════════════════════════════════
def fig_f9():
    fig = plt.figure(figsize=(16, 8))

    # Left: Table
    ax_t = fig.add_axes([0.02, 0.08, 0.55, 0.88])
    ax_t.axis("off")
    modes = ["ce", "cumulative", "sord", "edl", "edl_orcu"]
    tbl = [["Mode", "Acc", "F1", "QWK", "ECE", "%Unimodal"]]
    for mode in modes:
        v = V22[mode]
        tbl.append([MODE_NAMES[mode], f"{v['acc']:.4f}", f"{v['f1']:.4f}",
                    f"{v['qwk']:.4f}", f"{v['ece']:.4f}", f"{v['unim']:.1%}"])
    ax_t.set_title("DF Fluorosis — v2.2 Results (for Paper)", fontweight="bold", fontsize=13, loc="left")
    table = ax_t.table(cellText=tbl[1:], colLabels=tbl[0], loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1.0, 1.6)
    for col in range(6):
        table[3, col].set_facecolor("#D4EDDA")  # EDL row highlighted

    # Right: Radar chart
    ax_r = fig.add_axes([0.62, 0.15, 0.36, 0.7], projection="polar")
    cats = ["Acc", "F1", "QWK", "1-ECE", "%Unimodal"]
    N = len(cats)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]
    for mode in modes:
        v = V22[mode]
        vals = [v["acc"], v["f1"], v["qwk"], 1-v["ece"], v["unim"]] + [v["acc"]]
        ax_r.fill(angles, vals, alpha=0.1, color=MODE_COLORS[mode])
        ax_r.plot(angles, vals, "o-", lw=1.5, color=MODE_COLORS[mode], ms=4, label=MODE_NAMES[mode])
    ax_r.set_xticks(angles[:-1]); ax_r.set_xticklabels(cats, fontsize=9)
    ax_r.set_ylim(0.5, 1.0); ax_r.set_title("Radar (v2.2)", fontweight="bold", fontsize=12, pad=20)
    ax_r.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.suptitle("Figure 9: Summary — DF Fluorosis Diagnosis", fontweight="bold", y=0.98)
    for fmt in ["pdf", "png"]:
        fig.savefig(OUT_DIR / f"F9_summary.{fmt}", format=fmt)
    plt.close()
    print("  F9 done.")

# ═══════════════════════════════════════════════════════════════
# F10: Per-Class Analysis
# ═══════════════════════════════════════════════════════════════
def fig_f10():
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    modes = ["ce", "cumulative", "sord", "edl", "edl_orcu"]
    x = np.arange(len(modes)); w = 0.2

    # (a) Per-class precision
    ax = axes[0, 0]
    for k in range(4):
        prec = []
        for mode in modes:
            cm = metrics[mode]["cm"]
            prec.append(cm[k,k]/max(cm[:,k].sum(),1))
        ax.bar(x+k*w, prec, w, label=CLASS_NAMES[k], alpha=0.8)
    ax.set_xticks(x+w*1.5); ax.set_xticklabels([MODE_NAMES[m] for m in modes], rotation=20)
    ax.set_ylabel("Precision"); ax.set_title("(a) Per-Class Precision"); ax.legend(fontsize=8); ax.set_ylim(0,1.05)

    # (b) Per-class recall
    ax = axes[0, 1]
    for k in range(4):
        rec = []
        for mode in modes:
            cm = metrics[mode]["cm"]
            rec.append(cm[k,k]/max(cm[k,:].sum(),1))
        ax.bar(x+k*w, rec, w, label=CLASS_NAMES[k], alpha=0.8)
    ax.set_xticks(x+w*1.5); ax.set_xticklabels([MODE_NAMES[m] for m in modes], rotation=20)
    ax.set_ylabel("Recall"); ax.set_title("(b) Per-Class Recall"); ax.legend(fontsize=8); ax.set_ylim(0,1.05)

    # (c) Per-class F1
    ax = axes[1, 0]
    for k in range(4):
        f1c = []
        for mode in modes:
            cm = metrics[mode]["cm"]
            p = cm[k,k]/max(cm[:,k].sum(),1)
            r = cm[k,k]/max(cm[k,:].sum(),1)
            f1c.append(2*p*r/max(p+r,1e-8))
        ax.bar(x+k*w, f1c, w, label=CLASS_NAMES[k], alpha=0.8)
    ax.set_xticks(x+w*1.5); ax.set_xticklabels([MODE_NAMES[m] for m in modes], rotation=20)
    ax.set_ylabel("F1 Score"); ax.set_title("(c) Per-Class F1"); ax.legend(fontsize=8); ax.set_ylim(0,1.05)

    # (d) Average normalized confusion matrix
    ax = axes[1, 1]
    avg_cm = np.zeros((4,4))
    for mode in modes:
        cm = metrics[mode]["cm"]
        avg_cm += cm.astype(float)/cm.sum(axis=1,keepdims=True)
    avg_cm /= len(modes)
    im = ax.imshow(avg_cm, cmap="RdYlGn", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j,i,f"{avg_cm[i,j]:.1%}",ha="center",va="center",fontsize=10,
                   color="white" if avg_cm[i,j]>0.6 else "black",fontweight="bold")
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(CLASS_NAMES,rotation=45,ha="right"); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("(d) Avg Normalized Confusion"); plt.colorbar(im,ax=ax,shrink=0.8)

    plt.suptitle("Figure 10: Per-Class Performance Analysis", fontweight="bold", y=1.01)
    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        fig.savefig(OUT_DIR / f"F10_per_class.{fmt}", format=fmt)
    plt.close()
    print("  F10 done.")

# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("="*60)
    print("DF Fluorosis — Paper Figure Generation")
    print("="*60)
    fig_f1()
    fig_f2()
    fig_f3()
    fig_f4()
    fig_f5()
    fig_f6()
    fig_f7()
    fig_f8()
    fig_f9()
    fig_f10()
    print(f"\nDone! {len(os.listdir(OUT_DIR))} files saved to {OUT_DIR}")
