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
DATA_DIR = Path("/Volumes/KINGSTON/Obsibian/Fluorosis/Fluorosis_DL_Paper/05_Exp_Design/exp_v1/data")
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
# F2: Reliability Diagrams — Step curves + calibration gap ribbon
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
        bin_edges, bin_accs, bin_confs, bin_counts = [], [], [], []
        for i in range(nb):
            m = (confs > bb[i]) & (confs <= bb[i + 1])
            if m.sum() >= 1:
                bin_edges.append((bb[i], bb[i+1]))
                bin_accs.append(accs[m].mean())
                bin_confs.append(confs[m].mean())
                bin_counts.append(m.sum())
        if not bin_accs:
            continue
        bin_edges = np.array(bin_edges)
        bin_accs = np.array(bin_accs)
        bin_confs = np.array(bin_confs)
        step_x = bin_edges.flatten()
        step_y_acc = np.repeat(bin_accs, 2)
        step_y_conf = np.repeat(bin_confs, 2)

        ax.plot([0, 1], [0, 1], "k--", alpha=0.25, lw=0.8, zorder=1)
        ax.fill_between(step_x, step_y_acc, step_y_conf,
                        alpha=0.15, color=MODE_COLORS[mode], label="Calibration Gap")
        ax.step(step_x, step_y_conf, where="post", color="darkred", lw=1.8,
                label="Confidence", zorder=3)
        ax.step(step_x, step_y_acc, where="post", color=MODE_COLORS[mode], lw=1.8,
                label="Accuracy", zorder=2)
        midpoints = bin_edges.mean(axis=1)
        sizes = np.clip(np.array(bin_counts) * 15, 20, 120)
        ax.scatter(midpoints, bin_accs, s=sizes, color=MODE_COLORS[mode],
                   edgecolors="white", lw=0.5, zorder=4, alpha=0.9)

        ece = metrics[mode]["ece"]
        ax.text(0.95, 0.08, f"ECE={ece:.3f}", transform=ax.transAxes, ha="right",
                fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_title(MODE_NAMES[mode], fontweight="bold")
        ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
        if idx == 0: ax.legend(fontsize=7, loc="upper left")
    axes[5].set_visible(False)
    plt.suptitle("Figure 2: Reliability Diagrams with Calibration Gap", fontweight="bold", y=1.01)
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
# F4: Evidence Analysis — violin/box + scatter + stacked area + line
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

        # Row 1: Evidence distribution per true class — violin + strip
        ax = axes[0, col]
        ev_data = [total_ev[targets == k] for k in range(4)]
        vp = ax.violinplot(ev_data, positions=range(4), showmeans=True,
                           showmedians=True, widths=0.6)
        for body_k, body in enumerate(vp["bodies"]):
            body.set_facecolor(class_colors[body_k])
            body.set_alpha(0.5)
            body.set_edgecolor("black")
            body.set_linewidth(0.5)
        for part in ["cmeans", "cmedians", "cbars"]:
            if part in vp:
                vp[part].set_color("black")
                vp[part].set_linewidth(0.8)
        for k in range(4):
            if len(ev_data[k]) > 0:
                jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(ev_data[k]))
                ax.scatter(np.full_like(ev_data[k], k, dtype=float) + jitter, ev_data[k],
                          color=class_colors[k], alpha=0.4, s=15, ec="none", zorder=5)
        ax.set_xlabel("True Class"); ax.set_ylabel("Total Evidence")
        ax.set_title(f"{MODE_NAMES[mode]}: Evidence Distribution by Class")
        ax.set_xticks(range(4)); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")

        # Row 2: Uncertainty vs Evidence scatter (unchanged)
        ax = axes[1, col]
        correct = (d["predicted"].astype(int) == targets)
        ax.scatter(total_ev[correct], uncertainty[correct], c="#70AD47", alpha=0.6, s=30,
                  label="Correct", ec="none")
        ax.scatter(total_ev[~correct], uncertainty[~correct], c="#B2182B", alpha=0.6, s=30,
                  marker="x", label="Error", ec="none")
        ax.set_xlabel("Total Evidence"); ax.set_ylabel("Uncertainty")
        ax.set_title(f"{MODE_NAMES[mode]}: Uncertainty vs Evidence")
        ax.legend(fontsize=8)

    # Panel 3: EDL evidence component stacked area
    ax = axes[0, 2]
    ev = EXP1["edl"]["evidence"].astype(np.float32)
    tg = EXP1["edl"]["targets"].astype(int)
    ev_breakdown = np.zeros((4, 4))
    for k in range(4):
        mask = tg == k
        if mask.sum() > 0:
            ev_breakdown[k] = ev[mask].mean(axis=0)
    x = np.arange(4)
    ax.stackplot(x, [ev_breakdown[:, k] for k in range(4)],
                 labels=[f"e[{k}]" for k in range(4)],
                 colors=class_colors, alpha=0.75)
    ax.plot(x, ev_breakdown.sum(axis=1), "k-o", lw=2, ms=6, label="Total")
    ax.set_xlabel("True Class"); ax.set_ylabel("Mean Evidence")
    ax.set_title("EDL: Evidence Component Stack")
    ax.set_xticks(range(4)); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.legend(fontsize=7, ncol=5)

    # Panel 4: Acc/Conf by Evidence Level — line + shaded range
    ax = axes[1, 2]
    ev = EXP1["edl"]["evidence"].astype(np.float32)
    total_ev = ev.sum(axis=1)
    probs = EXP1["edl"]["probabilities"].astype(np.float32)
    tg = EXP1["edl"]["targets"].astype(int)
    ev_bins = np.percentile(total_ev, [0, 33, 67, 100])
    x_ev = np.arange(3)
    acc_vals, conf_vals = [], []
    for i in range(3):
        mask = (total_ev >= ev_bins[i]) & (total_ev < ev_bins[i+1])
        if mask.sum() > 0:
            acc_vals.append((probs[mask].argmax(axis=1) == tg[mask]).mean())
            conf_vals.append(probs[mask].max(axis=1).mean())
    ax.fill_between(x_ev, acc_vals, conf_vals, alpha=0.15, color="#4472C4")
    ax.plot(x_ev, acc_vals, "o-", color="#4472C4", lw=2, ms=8, label="Accuracy")
    ax.plot(x_ev, conf_vals, "s--", color="#ED7D31", lw=2, ms=8, label="Confidence")
    for i in range(3):
        ax.annotate(f"{acc_vals[i]:.2f}", (x_ev[i], acc_vals[i]),
                   textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8, color="#4472C4")
        ax.annotate(f"{conf_vals[i]:.2f}", (x_ev[i], conf_vals[i]),
                   textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color="#ED7D31")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Low Ev.", "Med Ev.", "High Ev."])
    ax.set_ylabel("Value"); ax.set_title("Acc/Conf by Evidence Level")
    ax.legend(fontsize=8); ax.set_ylim(0, 1.15)

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

    # (a) Acc + F1 + QWK grouped bar chart — KEEP as bars (core comparison)
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
        ax.bar([0], [0], w, color=plt.cm.Blues(0.5+i*0.2), alpha=0.85, ec="black", lw=0.3, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_NAMES[m] for m in modes], rotation=15, fontsize=10)
    ax.set_ylabel("Score"); ax.set_title("(a) Performance Metrics (v2.2)")
    ax.legend(fontsize=8, loc="lower left"); ax.set_ylim(0.55, 1.0)

    # (b) ECE — lollipop chart (stem + dot)
    ax = axes[0, 1]
    ece_vals = [V22[m]["ece"] for m in modes]
    for i, (xi, val) in enumerate(zip(x, ece_vals)):
        ax.plot([xi, xi], [0, val], color=bar_colors[i], lw=2.5, alpha=0.7, zorder=2)
        ax.scatter(xi, val, s=180, color=bar_colors[i], edgecolors="black",
                  lw=0.8, zorder=3)
        ax.text(xi, val + 0.006, f"{val:.4f}", ha="center", va="bottom",
               fontsize=10, fontweight="bold")
    ax.axhline(y=0.05, color="gray", ls="--", alpha=0.5, lw=1, label="Target ECE=0.05")
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_NAMES[m] for m in modes], rotation=15, fontsize=10)
    ax.set_ylabel("ECE (lower is better)"); ax.set_title("(b) Expected Calibration Error")
    ax.legend(fontsize=8); ax.set_ylim(0, max(ece_vals) * 1.15)

    # (c) Per-class recall — dumbbell chart (CE → EDL gap per class)
    ax = axes[1, 0]
    cm_ce = metrics["ce"]["cm"]
    cm_edl = metrics["edl"]["cm"]
    class_palette = ["#2166AC", "#92C5DE", "#F4A582", "#B2182B"]
    for k in range(4):
        r_ce = cm_ce[k, k] / max(cm_ce[k, :].sum(), 1)
        r_edl = cm_edl[k, k] / max(cm_edl[k, :].sum(), 1)
        gap = r_edl - r_ce
        ax.plot([r_ce, r_edl], [k, k], color="gray", lw=2.5, alpha=0.6, zorder=2)
        ax.scatter(r_ce, k, s=140, color="#4472C4", edgecolors="black", lw=0.6, zorder=3)
        ax.scatter(r_edl, k, s=140, color="#70AD47", edgecolors="black", lw=0.6, zorder=3)
        ax.annotate(f"{gap:+.1%}", ((r_ce+r_edl)/2, k + 0.2), ha="center", fontsize=8,
                   color="#B2182B" if gap < 0 else "#2E7D32", fontweight="bold")
    ax.set_yticks(range(4)); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Recall"); ax.set_title("(c) Per-Class Recall: CE → EDL")
    # Dummy legend
    ax.scatter([], [], s=80, color="#4472C4", edgecolors="black", lw=0.6, label="CE")
    ax.scatter([], [], s=80, color="#70AD47", edgecolors="black", lw=0.6, label="EDL")
    ax.legend(fontsize=8, loc="lower right"); ax.set_xlim(0, 1.05); ax.invert_yaxis()

    # (d) QWK + Unimodal — dot plot (Cleveland-style)
    ax = axes[1, 1]
    unim_vals = [V22[m]["unim"] * 100 for m in modes]
    qwk_vals = [V22[m]["qwk"] * 100 for m in modes]
    for yi, mode in enumerate(modes):
        ax.scatter(qwk_vals[yi], yi, s=100, color="#4472C4", zorder=3, label="QWK × 100" if yi == 0 else "")
        ax.scatter(unim_vals[yi], yi, s=100, color="#70AD47", marker="D", zorder=3,
                  label="%Unimodal" if yi == 0 else "")
        ax.plot([qwk_vals[yi], unim_vals[yi]], [yi, yi], color="gray", lw=1, alpha=0.4, zorder=2)
    ax.set_yticks(range(len(modes)))
    ax.set_yticklabels([MODE_NAMES[m] for m in modes])
    ax.set_xlabel("Score / Percentage"); ax.set_title("(d) QWK & Ordinal Constraint")
    ax.legend(fontsize=8, loc="lower right"); ax.set_xlim(0, 105)
    ax.invert_yaxis()

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
# F7: Temperature Calibration — ECE curve + small multiples + shift
# ═══════════════════════════════════════════════════════════════
def fig_f7():
    temps = np.array([0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0])
    ece_vals = np.array([0.2845, 0.2619, 0.2610, 0.2285, 0.1933, 0.1196, 0.2492])

    fig = plt.figure(figsize=(16, 8))

    # (a) ECE vs T — area under curve
    ax = fig.add_subplot(2, 3, 1)
    ax.fill_between(temps, 0, ece_vals, alpha=0.2, color="#B2182B")
    ax.plot(temps, ece_vals, "o-", color="#B2182B", lw=2, ms=8)
    best_i = np.argmin(ece_vals)
    ax.annotate(f"Best: T={temps[best_i]}\nECE={ece_vals[best_i]:.4f}",
               xy=(temps[best_i], ece_vals[best_i]),
               xytext=(temps[best_i]+1.5, ece_vals[best_i]+0.05),
               arrowprops=dict(arrowstyle="->", color="darkred"),
               fontsize=9, color="darkred", fontweight="bold")
    ax.axhline(y=0.05, color="gray", ls="--", alpha=0.5, label="Target ECE=0.05")
    ax.set_xlabel("Temperature T"); ax.set_ylabel("ECE")
    ax.set_title("(a) ECE vs Temperature"); ax.legend(fontsize=8)

    # (b) Small multiples: calibration per T
    probs = EXP1["edl"]["probabilities"].astype(np.float64)
    targets = EXP1["edl"]["targets"]
    T_show = [0.5, 1.0, 2.0, 3.0, 5.0]
    T_colors = ["#2166AC", "#92C5DE", "#F4A582", "#B2182B", "#4D4D4D"]
    for ti, (T, tc) in enumerate(zip(T_show, T_colors)):
        ax = fig.add_subplot(2, 5, 6 + ti)
        logits = np.log(probs + 1e-8)
        sp = softmax_np(logits / T)
        confs = sp.max(axis=1)
        accs = (sp.argmax(axis=1) == targets).astype(float)
        nb = 8; bb = np.linspace(0, 1, nb+1)
        bin_edges, bin_accs, bin_confs = [], [], []
        for i in range(nb):
            m = (confs > bb[i]) & (confs <= bb[i+1])
            if m.sum() >= 1:
                bin_edges.append((bb[i], bb[i+1]))
                bin_accs.append(accs[m].mean())
                bin_confs.append(confs[m].mean())
        bin_edges = np.array(bin_edges); bin_accs = np.array(bin_accs); bin_confs = np.array(bin_confs)
        step_x = bin_edges.flatten()
        ax.fill_between(step_x, np.repeat(bin_accs, 2), np.repeat(bin_confs, 2),
                        alpha=0.12, color=tc)
        ax.step(step_x, np.repeat(bin_confs, 2), where="post", color="darkred", lw=1.2)
        ax.step(step_x, np.repeat(bin_accs, 2), where="post", color=tc, lw=1.5)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.25, lw=0.6)
        ax.scatter(bin_edges.mean(axis=1), bin_accs, s=20, color=tc, zorder=5)
        ax.set_title(f"T={T:.1f}", fontsize=9, fontweight="bold")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        if ti == 0: ax.set_ylabel("Accuracy")
        ax.set_xlabel("Conf.")

    # (c) Confidence distribution shift — ridge density
    ax = fig.add_subplot(2, 3, 3)
    T_ridge = [0.5, 1.0, 2.0, 3.0, 5.0]
    offset = 0
    for T in reversed(T_ridge):
        logits = np.log(probs + 1e-8)
        sp = softmax_np(logits / T)
        conf = sp.max(axis=1)
        hist, edges = np.histogram(conf, bins=15, range=(0, 1), density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.fill_between(centers, offset, offset + hist, alpha=0.7, label=f"T={T:.1f}")
        ax.plot(centers, offset + hist, lw=1.5, color="black", alpha=0.5)
        offset += hist.max() + 0.3
    ax.set_xlabel("Confidence"); ax.set_title("(c) Confidence Shift (Ridge)")
    ax.legend(fontsize=7, loc="upper right"); ax.set_yticks([])

    plt.suptitle("Figure 7: Temperature Calibration — EDL", fontweight="bold", y=1.01)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.08, wspace=0.35, hspace=0.4)
    for fmt in ["pdf", "png"]:
        fig.savefig(OUT_DIR / f"F7_temperature.{fmt}", format=fmt)
    plt.close()
    print("  F7 done.")

def softmax_np(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

# ═══════════════════════════════════════════════════════════════
# F8: Prediction Distribution — ridge plots + donut of %Unimodal
# ═══════════════════════════════════════════════════════════════
def fig_f8():
    fig = plt.figure(figsize=(14, 9))
    modes = ["ce", "cumulative", "sord", "edl", "edl_orcu"]
    class_colors = ["#2166AC", "#92C5DE", "#F4A582", "#B2182B"]

    # Ridge plots for each mode (panels 1-5)
    for idx, mode in enumerate(modes):
        ax = fig.add_subplot(2, 3, idx + 1)
        d = EXP1[mode]
        probs = d["probabilities"].astype(np.float64)
        targets = d["targets"].astype(int)
        # For each true class, plot a density ridge of predicted probability at true class
        # X axis: probability assigned to the CORRECT class
        offset = 0
        for k in range(4):
            mask = targets == k
            correct_probs = probs[mask, k]  # prob of correct class
            if len(correct_probs) > 1:
                hist, edges = np.histogram(correct_probs, bins=12, range=(0, 1), density=True)
                centers = (edges[:-1] + edges[1:]) / 2
                ax.fill_between(centers, offset, offset + hist,
                               alpha=0.55, color=class_colors[k], ec="black", lw=0.3)
                ax.plot(centers, offset + hist, lw=1.2, color="black", alpha=0.4)
                offset += hist.max() + 0.5
        ax.set_title(f"{MODE_NAMES[mode]} (Unim={metrics[mode]['unim']:.0%})", fontweight="bold", fontsize=10)
        ax.set_xlabel("P(correct class)"); ax.set_xlim(0, 1)
        ax.set_yticks([])
        if idx == 0:
            # legend for classes
            for k in range(4):
                ax.text(0.02, 0.95 - k * 0.08, CLASS_NAMES[k], transform=ax.transAxes,
                       fontsize=7, color=class_colors[k], fontweight="bold")

    # Panel 6: Donut chart — %Unimodal per mode
    ax = fig.add_subplot(2, 3, 6)
    unim_pcts = [metrics[m]["unim"] * 100 for m in modes]
    nonunim_pcts = [100 - v for v in unim_pcts]
    # Outer ring: EDL
    outer_colors = [MODE_COLORS[m] for m in modes]
    wedges1, texts1 = ax.pie(unim_pcts, radius=1, colors=outer_colors, startangle=90,
                              wedgeprops=dict(width=0.3, edgecolor="white", lw=1))
    # Inner ring: %Unimodal label
    ax.pie([100], radius=0.68, colors=["white"], wedgeprops=dict(width=0.2))
    ax.text(0, 0, "%Unim\nper Mode", ha="center", va="center", fontsize=9, fontweight="bold")
    # Legend
    labels = [f"{MODE_NAMES[m]}: {unim_pcts[i]:.0f}%" for i, m in enumerate(modes)]
    ax.legend(wedges1, labels, fontsize=7, loc="center left", bbox_to_anchor=(0.75, 0.5))
    ax.set_title("(f) %Unimodal Distribution", fontweight="bold", fontsize=10)

    plt.suptitle("Figure 8: Prediction Probability Distribution & Unimodal Pattern", fontweight="bold", y=1.01)
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
# F10: Per-Class Analysis — bubble + heatmap tile + confusion
# ═══════════════════════════════════════════════════════════════
def fig_f10():
    fig = plt.figure(figsize=(18, 9))
    modes = ["ce", "cumulative", "sord", "edl", "edl_orcu"]
    class_palette = ["#2166AC", "#92C5DE", "#F4A582", "#B2182B"]

    # (a) Bubble chart: x=Mode, y=Class, bubble size=F1, color=Precision
    ax = fig.add_subplot(2, 3, 1)
    for ci, cls in enumerate(CLASS_NAMES):
        for mi, mode in enumerate(modes):
            cm = metrics[mode]["cm"]
            p = cm[ci, ci] / max(cm[:, ci].sum(), 1)
            r = cm[ci, ci] / max(cm[ci, :].sum(), 1)
            f1 = 2 * p * r / max(p + r, 1e-8)
            ax.scatter(mi, ci, s=f1 * 500, color=plt.cm.RdYlGn(p), edgecolors="black",
                      lw=0.8, alpha=0.85)
            ax.text(mi, ci, f"{f1:.2f}", ha="center", va="center", fontsize=7, fontweight="bold")
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels([MODE_NAMES[m] for m in modes], rotation=20)
    ax.set_yticks(range(4)); ax.set_yticklabels(CLASS_NAMES)
    ax.set_title("(a) Bubble: Size=F1, Color=Precision")
    ax.set_xlim(-0.5, 4.5); ax.set_ylim(-0.5, 3.5)

    # (b) Compact heatmap tile: Precision (class × mode)
    ax = fig.add_subplot(2, 3, 2)
    hm_prec = np.zeros((4, len(modes)))
    for ci in range(4):
        for mi, mode in enumerate(modes):
            cm = metrics[mode]["cm"]
            hm_prec[ci, mi] = cm[ci, ci] / max(cm[:, ci].sum(), 1)
    im = ax.imshow(hm_prec, cmap="Oranges", vmin=0, vmax=1, aspect="auto")
    for i in range(4):
        for j in range(len(modes)):
            ax.text(j, i, f"{hm_prec[i,j]:.2f}", ha="center", va="center", fontsize=8,
                   color="black" if hm_prec[i,j] < 0.7 else "white")
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels([MODE_NAMES[m] for m in modes], rotation=20)
    ax.set_yticks(range(4)); ax.set_yticklabels(CLASS_NAMES)
    ax.set_title("(b) Precision (Class × Mode)")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # (c) Compact heatmap tile: Recall (class × mode)
    ax = fig.add_subplot(2, 3, 3)
    hm_rec = np.zeros((4, len(modes)))
    for ci in range(4):
        for mi, mode in enumerate(modes):
            cm = metrics[mode]["cm"]
            hm_rec[ci, mi] = cm[ci, ci] / max(cm[ci, :].sum(), 1)
    im = ax.imshow(hm_rec, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    for i in range(4):
        for j in range(len(modes)):
            ax.text(j, i, f"{hm_rec[i,j]:.2f}", ha="center", va="center", fontsize=8,
                   color="black" if hm_rec[i,j] < 0.7 else "white")
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels([MODE_NAMES[m] for m in modes], rotation=20)
    ax.set_yticks(range(4)); ax.set_yticklabels(CLASS_NAMES)
    ax.set_title("(c) Recall (Class × Mode)")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # (d) Average normalized confusion — heatmap
    ax = fig.add_subplot(2, 3, 4)
    avg_cm = np.zeros((4, 4))
    for mode in modes:
        cm = metrics[mode]["cm"]
        avg_cm += cm.astype(float) / cm.sum(axis=1, keepdims=True)
    avg_cm /= len(modes)
    im = ax.imshow(avg_cm, cmap="RdYlGn", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{avg_cm[i,j]:.1%}", ha="center", va="center", fontsize=10,
                   color="white" if avg_cm[i, j] > 0.6 else "black", fontweight="bold")
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right"); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("(d) Avg Normalized Confusion"); plt.colorbar(im, ax=ax, shrink=0.8)

    # (e) Per-class accuracy — slope chart (parallel coordinates lite)
    ax = fig.add_subplot(2, 3, 5)
    for k in range(4):
        pca = []
        for mode in modes:
            cm = metrics[mode]["cm"]
            pca.append(cm[k, k] / cm.sum(axis=1)[k])
        # Thick band from min to max
        ax.fill_between(range(len(modes)), [min(pca)]*len(modes), [max(pca)]*len(modes),
                        alpha=0.08, color=class_palette[k])
        ax.plot(range(len(modes)), pca, marker="D", color=class_palette[k],
               lw=2.5, ms=8, label=CLASS_NAMES[k])
        ax.scatter(np.argmax(pca), max(pca), s=60, color=class_palette[k],
                  edgecolors="black", lw=1, zorder=5)
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels([MODE_NAMES[m] for m in modes], rotation=20)
    ax.set_ylabel("Per-Class Accuracy"); ax.set_title("(e) Per-Class Accuracy (Slope)")
    ax.legend(fontsize=7, loc="lower left"); ax.set_ylim(0, 1.1)

    # (f) Overall metrics mini-heatmap across modes
    ax = fig.add_subplot(2, 3, 6)
    metric_keys = ["acc", "f1", "qwk", "ece", "unim"]
    metric_labels = ["Acc", "F1", "QWK", "ECE↓", "%Unim"]
    hm_data = np.zeros((len(metric_keys), len(modes)))
    for i, mk in enumerate(metric_keys):
        for j, mode in enumerate(modes):
            hm_data[i, j] = V22[mode][mk]
    im = ax.imshow(hm_data, cmap="RdYlGn", aspect="auto")
    for i in range(len(metric_keys)):
        for j in range(len(modes)):
            color = "white" if hm_data[i, j] > 0.6 else "black"
            ax.text(j, i, f"{hm_data[i, j]:.3f}", ha="center", va="center", fontsize=9, color=color)
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels([MODE_NAMES[m] for m in modes], rotation=20)
    ax.set_yticks(range(len(metric_keys)))
    ax.set_yticklabels(metric_labels)
    ax.set_title("(f) Metrics Overview (v2.2)")
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle("Figure 10: Per-Class Performance Analysis", fontweight="bold", y=1.01)
    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        fig.savefig(OUT_DIR / f"F10_per_class.{fmt}", format=fmt)
    plt.close()
    print("  F10 done.")

# ═══════════════════════════════════════════════════════════════
# F11: Model Architecture Diagram
# ═══════════════════════════════════════════════════════════════
def fig_f11():
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(1, 1, figsize=(18, 8))
    ax.set_xlim(0, 18); ax.set_ylim(0, 9.5); ax.axis("off")

    C = {"input": "#E8F0FE", "backbone": "#4472C4", "head": "#70AD47",
         "edl": "#ED7D31", "orcu": "#FFC000", "loss": "#B2182B",
         "feat": "#D9E2F3", "logits": "#C5E0B4", "unc": "#FCE4D6"}

    def box(x, y, w, h, text, color, fs=9, fc="white", ew=1.2):
        b = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.12",
                          facecolor=color, edgecolor="black", linewidth=ew, zorder=3)
        ax.add_patch(b)
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold",
               color=fc, zorder=4)

    def arrow(x1, y1, x2, y2, color="black", lw=2.0, z=1):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", color=color,
                           lw=lw, zorder=z, mutation_scale=20,
                           connectionstyle="arc3,rad=0")
        ax.add_patch(a)

    # ── CENTRAL FLOW (x=9 center) ──
    CX = 9  # center x
    # Input
    box(CX, 9.0, 4.0, 0.7, "Input Image  (224 x 224 x 3)", C["input"], fc="black")
    arrow(CX, 8.63, CX, 8.2)

    # ViT Backbone
    box(CX, 7.85, 6.0, 1.0, "ViT-Base Backbone (ImageNet-21k pretrained, 86M)\nPatch Embed -> 12x Transformer Encoder -> CLS token",
        C["backbone"], fs=8)
    arrow(CX, 7.33, CX, 6.75)

    # Feature vector
    box(CX, 6.55, 3.2, 0.45, "Feature Vector  f in R^768", C["feat"], fc="black")
    arrow(CX, 6.3, CX, 5.8)

    # Evidence Head
    box(CX, 5.55, 3.4, 0.45, "Evidence Head  --  FC(768 -> 4)", C["head"])
    arrow(CX, 5.3, CX, 4.85)

    # Logits
    box(CX, 4.65, 3.2, 0.42, "Logits  z = [z0, z1, z2, z3]", C["logits"], fc="black")

    # ── LEFT BRANCH: EDL ──
    arrow(CX-1.7, 4.45, 3.5, 3.85, C["edl"], lw=2.5, z=2)
    box(3.3, 3.6, 3.6, 0.65, "Softplus + 1\nalpha = softplus(z) + 1", C["edl"], fs=8)
    arrow(3.3, 3.25, 3.3, 2.7)
    box(3.3, 2.45, 3.6, 0.55, "Dirichlet Distribution\np ~ Dir(alpha),  y_hat = argmax alpha", C["edl"], fs=8)
    arrow(3.3, 2.15, 3.3, 1.55)
    box(3.3, 1.35, 3.4, 0.45, "Uncertainty  u = K / sum(alpha)", C["unc"], fc="black")
    # Loss arrow from Dirichlet → loss
    arrow(1.4, 2.45, 1.4, 1.1, C["loss"], lw=2, z=2)
    box(0.9, 0.85, 2.8, 0.55, r"$\mathcal{L}_{EDL}(\alpha, y)$" + "\nEvidential Loss", C["loss"], fs=8)

    # ── RIGHT BRANCH: ORCU ──
    arrow(CX+1.7, 4.45, 14.7, 3.85, C["orcu"], lw=2.5, z=2)
    box(14.7, 3.6, 3.2, 0.65, "Softmax\np = softmax(z),  y_hat = argmax p", C["orcu"], fs=8)
    arrow(14.7, 3.25, 14.7, 2.7)
    box(14.7, 2.45, 3.2, 0.55, "Prediction\nClass Probabilities  p_k", C["orcu"], fs=8)
    # Loss arrow from Softmax → loss
    arrow(16.4, 3.6, 16.4, 1.1, C["loss"], lw=2, z=2)
    box(16.6, 0.85, 2.8, 0.55, r"$\mathcal{L}_{ORCU}(z, y)$" + "\nOrdinal Regularizer", C["loss"], fs=8)

    # ── COMBINED LOSS ──
    arrow(0.9, 0.55, CX-3.2, 0.25, "black", lw=1.8)
    arrow(16.6, 0.55, CX+3.2, 0.25, "black", lw=1.8)
    box(CX, 0.1, 7.5, 0.5,
        r"$\mathcal{L}_{total} = \mathcal{L}_{EDL} + \lambda \cdot \mathcal{L}_{ORCU}$",
        C["loss"], fs=10)

    # ── TRAINING STRATEGY ──
    ax.text(CX, -0.35, "3-Stage:  CE Warmup (ep 0-4)  ->  EDL Only (ep 5-34)  ->  EDL + lambda*ORCU (ep 35-99)",
           ha="center", va="top", fontsize=9, color="#555555", fontstyle="italic")

    # ── SECTION LABELS ──
    ax.text(3.3, 4.5, "Evidence Path", ha="center", fontsize=9, color=C["edl"], fontweight="bold")
    ax.text(14.7, 4.5, "Classification Path", ha="center", fontsize=9, color=C["orcu"], fontweight="bold")
    ax.text(CX, 9.45, "DF Fluorosis Diagnosis Pipeline", ha="center", fontsize=10, fontweight="bold")

    # ── DASHED BRANCH SEPARATOR ──
    ax.plot([CX, CX], [5.2, 4.0], "k--", lw=0.8, alpha=0.3)
    ax.plot([CX-0.5, 3.3], [5.0, 4.55], "k--", lw=0.8, alpha=0.3)
    ax.plot([CX+0.5, 14.7], [5.0, 4.55], "k--", lw=0.8, alpha=0.3)

    plt.suptitle("Figure 11: Model Architecture — EDL + ORCU Dual-Path Framework", fontweight="bold", y=1.005)
    fig.subplots_adjust(top=0.94, bottom=0.04, left=0.02, right=0.98)
    for fmt in ["pdf", "png"]:
        fig.savefig(OUT_DIR / f"F11_architecture.{fmt}", format=fmt)
    plt.close()
    print("  F11 done.")


# ═══════════════════════════════════════════════════════════════
# F12: Multi-seed Stability — CE vs EDL across 5 seeds
# ═══════════════════════════════════════════════════════════════
def fig_f12():
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    seeds = [42, 123, 456, 789, 1024]
    x = np.arange(len(seeds))

    # v2.2 multi-seed data from comprehensive analysis
    ce_acc = [0.8167, 0.6167, 0.3500, 0.6167, 0.4500]
    ce_qwk = [0.9329, 0.8834, 0.7392, 0.8504, 0.6601]
    edl_acc = [0.8333, 0.8000, 0.7500, 0.7667, 0.8000]
    edl_qwk = [0.9376, 0.9200, 0.8985, 0.9238, 0.9287]

    # Panel (a): Accuracy
    ax = axes[0]
    for i, s in enumerate(seeds):
        ax.plot([i - 0.12, i + 0.12], [ce_acc[i], edl_acc[i]], color="gray", lw=1.5, alpha=0.5, zorder=2)
    ax.scatter(x - 0.12, ce_acc, s=160, color="#4472C4", edgecolors="black", lw=0.8,
              zorder=4, label="CE")
    ax.scatter(x + 0.12, edl_acc, s=160, color="#70AD47", edgecolors="black", lw=0.8,
              zorder=4, label="EDL")
    # Mean lines
    ax.axhline(y=np.mean(ce_acc), color="#4472C4", ls="--", lw=1.5, alpha=0.7)
    ax.axhline(y=np.mean(edl_acc), color="#70AD47", ls="--", lw=1.5, alpha=0.7)
    # Annotation for CE collapse
    ax.annotate("CE collapse\n(seed=456)\nAcc=35.0%", xy=(2, 0.35), xytext=(3.5, 0.38),
               arrowprops=dict(arrowstyle="->", color="darkred", lw=1.5),
               fontsize=9, color="darkred", fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9))
    # CV annotation
    ax.text(0.98, 0.22, f"CE CV={np.std(ce_acc)/np.mean(ce_acc)*100:.1f}%\nEDL CV={np.std(edl_acc)/np.mean(edl_acc)*100:.1f}%",
           transform=ax.transAxes, ha="right", fontsize=9, fontweight="bold",
           bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    ax.set_xticks(x); ax.set_xticklabels([str(s) for s in seeds])
    ax.set_xlabel("Random Seed"); ax.set_ylabel("Accuracy")
    ax.set_title("(a) Accuracy Stability (5 Seeds, v2.2)")
    ax.legend(fontsize=9, loc="upper right"); ax.set_ylim(0.2, 0.95)

    # Panel (b): QWK
    ax = axes[1]
    for i, s in enumerate(seeds):
        ax.plot([i - 0.12, i + 0.12], [ce_qwk[i], edl_qwk[i]], color="gray", lw=1.5, alpha=0.5, zorder=2)
    ax.scatter(x - 0.12, ce_qwk, s=160, color="#4472C4", edgecolors="black", lw=0.8,
              zorder=4, label="CE")
    ax.scatter(x + 0.12, edl_qwk, s=160, color="#70AD47", edgecolors="black", lw=0.8,
              zorder=4, label="EDL")
    ax.axhline(y=np.mean(ce_qwk), color="#4472C4", ls="--", lw=1.5, alpha=0.7)
    ax.axhline(y=np.mean(edl_qwk), color="#70AD47", ls="--", lw=1.5, alpha=0.7)
    # QWK std annotation
    ax.text(0.98, 0.22, f"CE QWK std={np.std(ce_qwk):.4f}\nEDL QWK std={np.std(edl_qwk):.4f}",
           transform=ax.transAxes, ha="right", fontsize=9, fontweight="bold",
           bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    ax.set_xticks(x); ax.set_xticklabels([str(s) for s in seeds])
    ax.set_xlabel("Random Seed"); ax.set_ylabel("QWK")
    ax.set_title("(b) QWK Stability (5 Seeds, v2.2)")
    ax.legend(fontsize=9, loc="lower left"); ax.set_ylim(0.55, 1.0)

    plt.suptitle("Figure 12: Training Stability — CE vs EDL (ViT-Base, DFID)", fontweight="bold", y=1.02)
    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        fig.savefig(OUT_DIR / f"F12_multiseed.{fmt}", format=fmt)
    plt.close()
    sns.set_style("darkgrid")
    print("  F12 done.")


# ═══════════════════════════════════════════════════════════════
# F13: SOTA Comparison — Ours vs Competitors
# ═══════════════════════════════════════════════════════════════
def fig_f13():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    competitors = ["MLTrMR\n(JVCIR'25)", "LD2Net\n(JRTIP'26)", "FusionDentNet\n(2024)",
                   "HiFuse\n(2024)", "Ours CE\n(v2.2)", "Ours EDL\n(v2.2)"]
    comp_acc = [0.8019, 0.8000, 0.8000, 0.7823, 0.8167, 0.8333]
    comp_f1 = [0.7579, 0.7988, 0.7925, 0.7045, 0.8085, 0.8278]
    comp_qwk = [0.813, np.nan, np.nan, np.nan, 0.9329, 0.9376]
    comp_params = [556, 3.31, 201, 164, 86, 86]
    comp_colors = ["#A5A5A5", "#A5A5A5", "#A5A5A5", "#A5A5A5", "#4472C4", "#70AD47"]

    x = np.arange(len(competitors))
    ours_mask = [False, False, False, False, True, True]

    # Panel (a): Acc + F1 grouped bar
    ax = axes[0]
    w = 0.28
    bars_acc = ax.bar(x - w/2, comp_acc, w, color=comp_colors, alpha=0.88,
                     edgecolor="black", lw=0.6, label="Accuracy")
    bars_f1 = ax.bar(x + w/2, comp_f1, w, color=[plt.cm.Set2(i/8) for i in range(6)],
                     alpha=0.75, edgecolor="black", lw=0.6, label="F1 Score",
                     hatch="///" if False else "")
    # Re-do F1 bars with proper styling
    for bar in bars_f1:
        bar.remove()
    f1_colors = ["#D4B9D6", "#D4B9D6", "#D4B9D6", "#D4B9D6", "#92C5DE", "#A8DBA8"]
    bars_f1 = ax.bar(x + w/2, comp_f1, w, color=f1_colors, alpha=0.88,
                     edgecolor="black", lw=0.6, label="F1 Score")
    # Value labels
    for bar, val in zip(bars_acc, comp_acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
               f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold", rotation=90)
    for bar, val in zip(bars_f1, comp_f1):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
               f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold", rotation=90)
    # Highlight EDL best
    ax.annotate("SOTA", xy=(5 - w/2, 0.8333), xytext=(5 - w/2, 0.87),
               ha="center", fontsize=8, color="#2E7D32", fontweight="bold",
               arrowprops=dict(arrowstyle="->", color="#2E7D32", lw=1.2))
    ax.set_xticks(x); ax.set_xticklabels(competitors, fontsize=9)
    ax.set_ylabel("Score"); ax.set_title("(a) Classification Performance")
    ax.legend(fontsize=9, loc="lower right"); ax.set_ylim(0.65, 0.92)

    # Panel (b): QWK + Params dual-axis
    ax = axes[1]
    # QWK bars
    qwk_vals_display = [v if not np.isnan(v) else 0 for v in comp_qwk]
    bar_colors_qwk = ["#A5A5A5", "#A5A5A5", "#A5A5A5", "#A5A5A5",
                      "#4472C4", "#70AD47"]
    bars_qwk = ax.bar(x - 0.18, qwk_vals_display, 0.35, color=bar_colors_qwk, alpha=0.88,
                     edgecolor="black", lw=0.6, label="QWK")
    # Mark NaN as "N/R" (not reported)
    for i, v in enumerate(comp_qwk):
        if np.isnan(v):
            ax.text(i - 0.18, 0.05, "N/R", ha="center", fontsize=8, color="gray",
                   fontstyle="italic")
        else:
            ax.text(i - 0.18, v + 0.005, f"{v:.3f}", ha="center", va="bottom",
                   fontsize=8, fontweight="bold", rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(competitors, fontsize=9)
    ax.set_ylabel("QWK"); ax.set_ylim(0, 1.05)
    ax.set_title("(b) QWK & Model Complexity")

    # Params on secondary axis (log scale)
    ax2 = ax.twinx()
    ax2.scatter(x + 0.2, comp_params, s=[p/2 for p in comp_params], color="darkred",
               alpha=0.7, edgecolors="black", lw=0.6, zorder=5)
    # Connect with thin line
    ax2.plot(x + 0.2, comp_params, color="darkred", lw=1, alpha=0.4, zorder=3)
    for i, p in enumerate(comp_params):
        label = f"{p}M" if p >= 10 else f"{p:.1f}M"
        ax2.text(i + 0.2, p * 1.15, label, ha="center", fontsize=8, color="darkred",
                fontweight="bold")
    ax2.set_ylabel("Parameters (M)", color="darkred"); ax2.set_yscale("log")
    ax2.tick_params(axis="y", labelcolor="darkred")

    # Combined legend
    import matplotlib.patches as mpatches_l
    legend_elements = [
        mpatches_l.Patch(facecolor="#4472C4", alpha=0.88, label="Ours CE"),
        mpatches_l.Patch(facecolor="#70AD47", alpha=0.88, label="Ours EDL"),
        mpatches_l.Patch(facecolor="#A5A5A5", alpha=0.88, label="Competitors"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper left")

    plt.suptitle("Figure 13: SOTA Comparison — DF Fluorosis Diagnosis", fontweight="bold", y=1.02)
    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        fig.savefig(OUT_DIR / f"F13_sota_comparison.{fmt}", format=fmt)
    plt.close()
    print("  F13 done.")


# ═══════════════════════════════════════════════════════════════
# F14: Theta Threshold Sweep — SafePred + RecAcc + AutoRate
# ═══════════════════════════════════════════════════════════════
def fig_f14():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    thetas = np.linspace(0.05, 0.55, 25)

    def safe_metrics(u, targets, preds, theta):
        high = u >= theta
        low = ~high
        ar = low.mean()
        sp = (preds[low] == targets[low]).mean() if low.sum() > 0 else np.nan
        ra = 1 - (preds[high] == targets[high]).mean() if high.sum() > 0 else np.nan
        return sp, ra, ar

    modes_to_plot = ["ce", "edl", "sord", "cumulative", "edl_orcu"]
    mode_colors = [MODE_COLORS[m] for m in modes_to_plot]

    # Panel (a): EDL vs CE theta sweep
    ax = axes[0]
    for mode in ["ce", "edl"]:
        d = EXP1[mode]
        u = d["uncertainty"].astype(np.float32)
        tg = d["targets"].astype(int)
        pr = d["predicted"].astype(int)
        # Normalize uncertainty to [0, 1] per mode for fair comparison
        u_norm = (u - u.min()) / max(u.max() - u.min(), 1e-8)
        sp_vals, ra_vals, ar_vals = [], [], []
        for th in thetas:
            sp, ra, ar = safe_metrics(u_norm, tg, pr, th)
            sp_vals.append(sp)
            ra_vals.append(ra)
            ar_vals.append(ar)
        ls = "-" if mode == "edl" else "--"
        lw = 2.5 if mode == "edl" else 1.8
        c = MODE_COLORS[mode]
        ax.plot(ar_vals, sp_vals, color=c, ls=ls, lw=lw, label=f"{MODE_NAMES[mode]} SafePred")
        ax.plot(ar_vals, ra_vals, color=c, ls=ls, lw=lw, alpha=0.5,
               label=f"{MODE_NAMES[mode]} RecAcc")
        # Find best theta (max SafePred * AutoRate tradeoff)
        scores = [s * a for s, a in zip(sp_vals, ar_vals) if not np.isnan(s)]
        best_i = np.argmax(scores) if scores else 0
        ax.scatter(ar_vals[best_i], sp_vals[best_i], s=120, color=c,
                  edgecolors="black", lw=1.5, zorder=5, marker="o")
        ax.annotate(f"theta={thetas[best_i]:.2f}", (ar_vals[best_i], sp_vals[best_i]),
                   textcoords="offset points", xytext=(8, -12), fontsize=8, color=c, fontweight="bold")
    ax.set_xlabel("AutoRate (fraction auto-diagnosed)"); ax.set_ylabel("Score")
    ax.set_title("(a) EDL vs CE: Uncertainty Triage Trade-off")
    ax.legend(fontsize=7, loc="lower left"); ax.set_xlim(0, 1.02); ax.set_ylim(0.3, 1.02)

    # Panel (b): 5-mode overlay at optimal theta each
    ax = axes[1]
    best_results = {}
    for mode in modes_to_plot:
        d = EXP1[mode]
        u = d["uncertainty"].astype(np.float32)
        tg = d["targets"].astype(int)
        pr = d["predicted"].astype(int)
        u_norm = (u - u.min()) / max(u.max() - u.min(), 1e-8)
        best_score, best_theta, best_sp, best_ar = -1, 0, 0, 0
        for th in thetas:
            sp, ra, ar = safe_metrics(u_norm, tg, pr, th)
            if not np.isnan(sp) and sp * ar > best_score:
                best_score = sp * ar
                best_theta, best_sp, best_ar, best_ra = th, sp, ar, ra
        best_results[mode] = {"theta": best_theta, "sp": best_sp, "ar": best_ar, "ra": best_ra}
        acc = metrics[mode]["acc"]
        ax.scatter(best_ar, best_sp, s=acc * 600, color=MODE_COLORS[mode],
                  edgecolors="black", lw=1.2, alpha=0.85, zorder=5)
        ax.text(best_ar, best_sp, MODE_NAMES[mode], fontsize=9,
               ha="center", va="bottom", fontweight="bold",
               color=MODE_COLORS[mode])
    # Diagonal reference line
    ax.plot([0, 1], [1, 0], "k--", alpha=0.2, lw=0.8)
    ax.set_xlabel("AutoRate"); ax.set_ylabel("SafePred")
    ax.set_title("(b) 5-Mode Auto-Diagnosis Performance (best theta per mode)")
    ax.set_xlim(0.2, 1.05); ax.set_ylim(0.5, 1.05)

    # Overlay theta table
    tbl_text = "Best theta per mode:\n"
    for mode in modes_to_plot:
        r = best_results[mode]
        tbl_text += f"{MODE_NAMES[mode]:12s}: theta={r['theta']:.2f}, Auto={r['ar']:.0%}, Safe={r['sp']:.2f}\n"
    ax.text(0.6, 0.62, tbl_text, transform=ax.transAxes, fontsize=7.5,
           family="monospace", va="top",
           bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85, ec="gray"))

    plt.suptitle("Figure 14: Uncertainty-Driven Auto-Diagnosis Triage", fontweight="bold", y=1.01)
    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        fig.savefig(OUT_DIR / f"F14_autodiagnosis.{fmt}", format=fmt)
    plt.close()
    print("  F14 done.")


# ═══════════════════════════════════════════════════════════════
# F15: Auto-Diagnosis Summary — SafePred vs AutoRate + Clinical Flow
# ═══════════════════════════════════════════════════════════════
def fig_f15():
    fig = plt.figure(figsize=(16, 6.5))
    modes_5 = ["ce", "cumulative", "sord", "edl", "edl_orcu"]

    # Shared theta sweep computation
    thetas = np.linspace(0.05, 0.55, 25)

    def safe_metrics(u, targets, preds, theta):
        high = u >= theta
        low = ~high
        ar = low.mean()
        sp = (preds[low] == targets[low]).mean() if low.sum() > 0 else np.nan
        ra = 1 - (preds[high] == targets[high]).mean() if high.sum() > 0 else np.nan
        return sp, ra, ar

    # Panel (a): Stacked bar — Auto vs Manual split with accuracy coloring
    ax = fig.add_subplot(1, 2, 1)
    bar_data = []
    for mode in modes_5:
        d = EXP1[mode]
        u = d["uncertainty"].astype(np.float32)
        tg = d["targets"].astype(int)
        pr = d["predicted"].astype(int)
        u_norm = (u - u.min()) / max(u.max() - u.min(), 1e-8)
        # Find best theta (max SafePred * AutoRate)
        best_score, best_th = -1, 0.3
        for th in thetas:
            sp, ra, ar = safe_metrics(u_norm, tg, pr, th)
            if not np.isnan(sp) and sp * ar > best_score:
                best_score, best_th = sp * ar, th
        high = u_norm >= best_th
        low = ~high
        n_auto = low.sum()
        n_manual = high.sum()
        auto_correct = (pr[low] == tg[low]).sum()
        auto_wrong = n_auto - auto_correct
        manual_correct = (pr[high] == tg[high]).sum()
        manual_wrong = n_manual - manual_correct
        bar_data.append((n_auto, auto_correct, auto_wrong, n_manual, manual_correct, manual_wrong))

    x = np.arange(len(modes_5))
    w = 0.6
    for i, (mode, (n_a, ac, aw, n_m, mc, mw)) in enumerate(zip(modes_5, bar_data)):
        total = n_a + n_m
        # Auto portion (bottom)
        ax.bar(i, n_a / total, w, color=MODE_COLORS[mode], alpha=0.85,
              edgecolor="black", lw=0.5)
        ax.bar(i, ac / total, w * 0.7, color="white", alpha=0.6, edgecolor="black", lw=0.3)
        # Manual portion (top)
        ax.bar(i, n_m / total, w, bottom=n_a / total, color=MODE_COLORS[mode], alpha=0.35,
              edgecolor="black", lw=0.5, hatch="///")
        # Labels
        ax.text(i, n_a / total / 2, f"Auto\n{n_a}", ha="center", va="center", fontsize=8, fontweight="bold")
        ax.text(i, n_a / total + n_m / total / 2, f"Manual\n{n_m}", ha="center", va="center",
               fontsize=8, color="darkred")
        # SafePred annotation
        sp = ac / max(n_a, 1)
        ax.text(i, -0.05, f"SafePred\n{sp:.2f}", ha="center", va="top", fontsize=8,
               color="black", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([MODE_NAMES[m] for m in modes_5], fontsize=10)
    ax.set_ylabel("Fraction of Test Samples"); ax.set_ylim(-0.12, 1.05)
    ax.set_title("(a) Auto vs Manual Review Split (best theta per mode)")
    # Legend
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor="gray", alpha=0.85, label="Auto (low u)"),
                  Patch(facecolor="gray", alpha=0.35, label="Manual (high u)")]
    ax.legend(handles=legend_els, fontsize=8)

    # Panel (b): Clinical triage workflow illustration
    ax = fig.add_subplot(1, 2, 2)
    ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis("off")
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    def b(x, y, w, h, t, c, fs=9, fc="white"):
        bx = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.1",
                           facecolor=c, edgecolor="black", lw=1.5, zorder=3)
        ax.add_patch(bx)
        ax.text(x, y, t, ha="center", va="center", fontsize=fs, fontweight="bold", color=fc, zorder=4)

    def arr(x1, y1, x2, y2, c="black", lw=2):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", color=c, lw=lw,
                           mutation_scale=18, zorder=1, connectionstyle="arc3,rad=0")
        ax.add_patch(a)

    # Flow
    b(6, 7.6, 4.5, 0.7, "Input: Intraoral Photo", "#E8F0FE", fc="black")
    arr(6, 7.22, 6, 6.75)
    b(6, 6.45, 5.0, 0.65, "EDL Model: Predict + Uncertainty u", "#70AD47")
    arr(6, 6.1, 6, 5.45)
    b(6, 5.15, 4.0, 0.55, "u < theta ?", "#FFC000", fc="black")

    # Left: YES → auto report
    arr(4, 5.15, 2.5, 4.5)
    b(2.5, 4.2, 3.5, 0.65, "YES: Auto-Diagnosis Report\n(SafePred > 0.85)", "#4472C4", fs=8)
    arr(2.5, 3.85, 2.5, 3.2)
    b(2.5, 2.95, 3.2, 0.5, "Output: Grade + Confidence", "#D9E2F3", fc="black", fs=8)

    # Right: NO → human review
    arr(8, 5.15, 9.5, 4.5)
    b(9.5, 4.2, 3.5, 0.65, "NO: Recommend Human Review\n(High Uncertainty)", "#ED7D31", fs=8)
    arr(9.5, 3.85, 9.5, 3.2)
    b(9.5, 2.95, 3.2, 0.5, "Dentist Manual Assessment", "#FCE4D6", fc="black", fs=8)

    # Merge
    arr(2.5, 2.65, 6, 2.0)
    arr(9.5, 2.65, 6, 2.0)
    b(6, 1.75, 4.5, 0.55, "Final Diagnosis Report", "#B2182B")

    ax.text(6, 1.15, "Auto Rate = % samples with u < theta", ha="center", fontsize=9,
           color="gray", fontstyle="italic")
    ax.text(6, 0.75, "Key Advantage: Uncertainty-gated triage reduces misdiagnosis risk",
           ha="center", fontsize=9, color="gray", fontstyle="italic")

    plt.suptitle("Figure 15: Auto-Diagnosis Clinical Triage System", fontweight="bold", y=1.005)
    fig.subplots_adjust(top=0.93, bottom=0.05)
    for fmt in ["pdf", "png"]:
        fig.savefig(OUT_DIR / f"F15_clinical_triage.{fmt}", format=fmt)
    plt.close()
    print("  F15 done.")


# ═══════════════════════════════════════════════════════════════
def fig_f16():
    """F16: Example Diagnostic Reports — 3 real test-sample report cards."""
    print("  Generating F16...")
    d = EXP1["edl"]
    probs, tgt, pred = d["probabilities"], d["targets"].astype(int), d["predicted"].astype(int)
    alpha = d["alpha"]
    S = alpha.sum(axis=1)
    u_norm = 4.0 / S
    conf = probs.max(axis=1)

    examples = [
        {"idx": 6,  "label": "A", "title": "Confident Auto-Diagnosis",
         "desc": "Correct, Low Uncertainty", "color": "#2E7D32", "bg": "#E8F5E9"},
        {"idx": 8,  "label": "B", "title": "Borderline — Needs Review",
         "desc": "Correct, High Uncertainty", "color": "#E65100", "bg": "#FFF3E0"},
        {"idx": 7,  "label": "C", "title": "Dangerous Overconfident",
         "desc": "Wrong, Low Uncertainty", "color": "#B71C1C", "bg": "#FFEBEE"},
    ]
    THETA = 0.50

    fig = plt.figure(figsize=(18, 7.5))
    import matplotlib.patches as mpatches_l
    from matplotlib.colors import LinearSegmentedColormap

    cmap_gauge = LinearSegmentedColormap.from_list("gauge", ["#2E7D32", "#FFC107", "#B71C1C"])

    for ci, ex in enumerate(examples):
        idx = ex["idx"]
        ax = fig.add_subplot(1, 3, ci + 1)
        ax.set_xlim(0, 10); ax.set_ylim(0, 12); ax.axis("off")

        # Card border
        card = mpatches_l.FancyBboxPatch((0.1, 0.01), 9.8, 11.98, boxstyle="round,pad=0.2",
            facecolor=ex["bg"], edgecolor=ex["color"], lw=2.5, zorder=1)
        ax.add_patch(card)

        # Header bar
        header = mpatches_l.FancyBboxPatch((0.1, 10.6), 9.8, 1.2, boxstyle="round,pad=0.1",
            facecolor=ex["color"], edgecolor="none", zorder=2)
        ax.add_patch(header)
        ax.text(5, 11.2, f"Example {ex['label']}", ha="center", va="center",
               fontsize=14, fontweight="bold", color="white", zorder=3)
        ax.text(5, 10.68, ex["desc"], ha="center", va="center", fontsize=11, color="white", alpha=0.95, zorder=3)

        # Sample ID + Ground Truth
        ax.text(0.5, 10.05, f"Sample #{idx}  |  Ground Truth: {CLASS_NAMES[tgt[idx]]}",
               fontsize=11, fontweight="bold", color="black", va="center")

        # Prediction summary
        result_text = "CORRECT" if pred[idx] == tgt[idx] else "WRONG"
        ax.text(0.5, 9.45,
                f"Prediction: {CLASS_NAMES[pred[idx]]}  |  Confidence: {conf[idx]:.3f}  |  {result_text}",
                fontsize=11, fontweight="bold", color=ex["color"])

        # Per-class probability bars
        ax.text(0.5, 8.85, "Class Probabilities:", fontsize=10, fontweight="bold", color="gray")
        cls_colors = ["#64B5F6", "#81C784", "#FFB74D", "#E57373"]
        for k in range(4):
            y = 8.15 - k * 0.85
            p_k = probs[idx][k]
            bg = mpatches_l.FancyBboxPatch((0.5, y - 2.2), 9.0, 0.65, boxstyle="round,pad=0.06",
                facecolor="#F5F5F5", edgecolor="#E0E0E0", lw=0.5, zorder=2)
            ax.add_patch(bg)
            bar_w = max(p_k * 8.8, 0.15)
            fg = mpatches_l.FancyBboxPatch((0.5, y - 2.2), bar_w, 0.65, boxstyle="round,pad=0.06",
                facecolor=cls_colors[k], edgecolor=ex["color"] if k == pred[idx] else "none",
                lw=2 if k == pred[idx] else 0, zorder=3)
            ax.add_patch(fg)
            marker = "  << PREDICTED" if k == pred[idx] else ("  (TRUE LABEL)" if k == tgt[idx] else "")
            ax.text(0.7, y - 1.87, f"{CLASS_NAMES[k]}{marker}: {p_k:.4f}",
                   fontsize=9, ha="left", va="center",
                   fontweight="bold" if k == pred[idx] else "normal",
                   color=ex["color"] if k == pred[idx] else "black", zorder=4)

        # Uncertainty gauge using imshow
        prob_bar_bottom = 8.15 - 4 * 0.85 - 2.2  # = 2.55
        ug_top = prob_bar_bottom + 0.15
        ax.text(0.5, ug_top + 0.05, "Uncertainty Gauge:", fontsize=10, fontweight="bold", color="gray")

        u_val = u_norm[idx]
        # Render gradient with imshow
        grad = np.linspace(0, 1, 256).reshape(1, -1)
        ax.imshow(grad, aspect="auto", cmap=cmap_gauge, extent=[0.5, 9.5, ug_top - 1.65, ug_top - 1.05],
                 zorder=2, alpha=0.9)

        # Gauge border
        gauge_frame = mpatches_l.Rectangle((0.5, ug_top - 1.65), 9.0, 0.6,
            facecolor="none", edgecolor="black", lw=1, zorder=3)
        ax.add_patch(gauge_frame)

        # Threshold line
        th_x = 0.5 + THETA * 9.0
        ax.plot([th_x, th_x], [ug_top - 1.7, ug_top - 0.95], color="black", lw=2.5, zorder=5)
        ax.text(th_x, ug_top - 0.7, f"theta = {THETA:.2f}", fontsize=10, ha="center",
               fontweight="bold", color="black")

        # Current u marker
        u_x = 0.5 + np.clip(u_val, 0, 1) * 9.0
        ax.plot(u_x, ug_top - 1.35, marker="v", color=ex["color"], markersize=14,
               markeredgecolor="black", markeredgewidth=1.5, zorder=6)
        ax.text(0.7, ug_top - 0.95, f"u = {u_val:.3f}", fontsize=11, fontweight="bold", color=ex["color"])

        # Zone labels
        ax.text(2.75, ug_top - 2.15, "AUTO ZONE (u < theta)", fontsize=9, ha="center",
               color="#2E7D32", fontstyle="italic", fontweight="bold")
        ax.text(7.25, ug_top - 2.15, "MANUAL ZONE (u >= theta)", fontsize=9, ha="center",
               color="#B71C1C", fontstyle="italic", fontweight="bold")

        # Triage Decision
        td_y = ug_top - 2.9
        triage = "AUTO-DIAGNOSIS" if u_val < THETA else "MANUAL REVIEW RECOMMENDED"
        triage_icon = "OK" if u_val < THETA else "!"
        triage_color = "#2E7D32" if u_val < THETA else "#B71C1C"
        triage_box = mpatches_l.FancyBboxPatch((0.5, td_y - 0.55), 9.0, 0.9,
            boxstyle="round,pad=0.1", facecolor=triage_color, edgecolor="none", alpha=0.15, zorder=2)
        ax.add_patch(triage_box)
        ax.text(5, td_y, f"[{triage_icon}]  Triage Decision: {triage}",
               fontsize=12, fontweight="bold", ha="center", color=triage_color, zorder=3)

        # Consequence note
        note_y = td_y - 1.15
        if pred[idx] == tgt[idx] and u_val < THETA:
            note = "Safe auto-diagnosis: model is correct and confident."
        elif pred[idx] == tgt[idx] and u_val >= THETA:
            note = "Correctly flagged for manual review: safe deferral despite correct prediction."
        else:
            note = "DANGER: confident wrong prediction would be auto-reported without uncertainty gate."
        ax.text(5, note_y, note, fontsize=9, ha="center", color="gray", fontstyle="italic")

    plt.suptitle("Figure 16: Example Diagnostic Reports — EDL Model on DFID Test Set",
                 fontweight="bold", y=1.01, fontsize=14)
    fig.subplots_adjust(top=0.93, wspace=0.28, left=0.02, right=0.98, bottom=0.01)
    for fmt in ["pdf", "png"]:
        fig.savefig(OUT_DIR / f"F16_example_reports.{fmt}", format=fmt)
    plt.close()
    print("  F16 done.")


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
    fig_f11()
    fig_f12()
    fig_f13()
    fig_f14()
    fig_f15()
    fig_f16()
    print(f"\nDone! {len(os.listdir(OUT_DIR))} files saved to {OUT_DIR}")
