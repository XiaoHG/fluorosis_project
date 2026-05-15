"""
Paper figure generation from EDL+ORCU experiment outputs.

Reads history.json and test_metrics.json produced by train.py,
generates publication-quality figures for the manuscript.

Figures:
  Fig 2: Confusion matrices (CE vs EDL vs EDL+ORCU)
  Fig 3: Reliability diagrams (ECE comparison)
  Fig 4: Uncertainty vs accuracy scatter
  Fig 5: Training curves (3-stage loss)
  Fig 6: Evidence distribution per class
  Fig 7: Ablation bar chart (method comparison)
  Fig 8: RecAcc & SafePred across theta thresholds
"""
import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import (
    COLORS, METHOD_LABELS, GRADE_LABELS,
    set_paper_style, save_figure, plot_confusion_matrix,
)


def load_history(path):
    with open(path) as f:
        return json.load(f)


def load_test_metrics(path):
    with open(path) as f:
        return json.load(f)


def fig_confusion_matrices(cm_dict, save_path):
    """Side-by-side normalized confusion matrix comparison."""
    methods = ["ce", "edl", "edl_orcu"]
    available = [m for m in methods if m in cm_dict]
    if not available:
        print("[SKIP] No confusion matrices found")
        return

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, method in zip(axes, available):
        cm = np.array(cm_dict[method])
        plot_confusion_matrix(ax, cm, title=METHOD_LABELS.get(method, method))

    fig.suptitle("Confusion Matrices: Method Comparison", fontsize=13, fontweight="bold")
    save_figure(fig, save_path)


def fig_reliability_diagrams(ece_data, save_path):
    """Reliability diagram comparing ECE across methods."""
    fig, ax = plt.subplots(figsize=(6, 5))

    for method, data in ece_data.items():
        if "bins" not in data:
            continue
        bins = np.array(data["bins"])
        accs = np.array(data["accs"])
        confs = np.array(data["confs"])
        label = f"{METHOD_LABELS.get(method, method)} (ECE={data['ece']:.4f})"
        ax.plot(confs, accs, "o-", color=COLORS.get(method, "#333"),
                markersize=5, linewidth=1.5, label=label)

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Perfect")
    set_paper_style(ax, title="Reliability Diagrams", xlabel="Confidence", ylabel="Accuracy")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    save_figure(fig, save_path)


def fig_uncertainty_vs_accuracy(alpha_list, z_list, targets_list, method_names, save_path):
    """Scatter: uncertainty vs per-sample correctness for each method."""
    fig, axes = plt.subplots(1, len(method_names), figsize=(5 * len(method_names), 4.5))
    if len(method_names) == 1:
        axes = [axes]

    for ax, name, alpha, targets in zip(axes, method_names, alpha_list, targets_list):
        alpha = np.array(alpha)
        targets = np.array(targets)
        K = alpha.shape[-1]
        S = alpha.sum(axis=-1)
        u = K / S
        probs = alpha / S[:, None]
        preds = probs.argmax(axis=-1)
        correct = (preds == targets).astype(float)

        jitter = np.random.uniform(-0.02, 0.02, len(u))
        ax.scatter(u, correct + jitter, alpha=0.4, s=15,
                   c=["#2e7d32" if c else "#c62828" for c in correct], edgecolors="none")
        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
        ax.axvline(0.3, color="orange", linestyle="--", alpha=0.5, label=r"$\theta$=0.30")

        set_paper_style(ax, title=METHOD_LABELS.get(name, name),
                        xlabel="Uncertainty u", ylabel="Correct")
        ax.set_xlim(0, 1); ax.set_ylim(-0.1, 1.1)

    fig.suptitle("Uncertainty vs Prediction Correctness", fontsize=13, fontweight="bold")
    save_figure(fig, save_path)


def fig_training_curves(history_path, save_path):
    """Three-stage training curves from history.json."""
    history = load_history(history_path)
    epochs = [h["epoch"] for h in history]
    stages = [h["stage"] for h in history]

    train_losses = []
    for h in history:
        tl = h.get("train_loss", {})
        train_losses.append(tl.get("L_total", tl.get("L_ce", tl.get("L_edl", 0))))

    val_metrics = {}
    for key in ["acc", "macro_f1", "qwk", "ece"]:
        val_metrics[key] = [h["val_metrics"].get(key, np.nan) for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, train_losses, color=COLORS["edl_orcu"], linewidth=1.5)
    for s in [1, 2, 3]:
        mask = np.array(stages) == s
        if mask.any():
            ax1.axvline(epochs[np.argmax(mask)], color="gray", linestyle=":", alpha=0.6)
    n = len(epochs)
    stage2_x = sum(1 for s in stages if s == 1) / n
    stage3_x = sum(1 for s in stages if s <= 2) / n
    ax1.text(0.02, 0.95, "CE", transform=ax1.transAxes, fontsize=8, color="gray")
    ax1.text(stage2_x + 0.02, 0.95, "EDL", transform=ax1.transAxes, fontsize=8, color="gray")
    ax1.text(stage3_x + 0.02, 0.95, "Joint", transform=ax1.transAxes, fontsize=8, color="gray")
    set_paper_style(ax1, title="Training Loss", xlabel="Epoch", ylabel="Loss")

    for key, label, color in [("acc", "Acc", "#2e7d32"), ("qwk", "QWK", "#1565c0"),
                                ("ece", "ECE", "#c62828")]:
        ax2.plot(epochs, val_metrics[key], linewidth=1.5, label=label, color=color)
    set_paper_style(ax2, title="Validation Metrics", xlabel="Epoch", ylabel="Score")
    ax2.legend(fontsize=8)

    save_figure(fig, save_path)


def fig_evidence_distribution(evidence_per_class, save_path):
    """Bar chart: mean evidence per grade."""
    keys = sorted(evidence_per_class.keys())
    values = [evidence_per_class[k] for k in keys]
    labels = [GRADE_LABELS[int(k[-1])] for k in keys]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(range(len(labels)), values, color=COLORS["evidence"],
                  edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    set_paper_style(ax, title="Mean Evidence per Class", xlabel="Grade", ylabel="Mean Evidence")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.2f}", ha="center", fontsize=9)
    save_figure(fig, save_path)


def fig_ablation_bars(ablation_results, save_path):
    """Grouped bar chart: Acc/F1/QWK/ECE across methods."""
    methods = list(ablation_results.keys())
    metric_keys = ["acc", "macro_f1", "qwk", "ece"]
    metric_labels = ["Accuracy", "Macro F1", "QWK", "ECE"]
    colors = [COLORS.get(m, "#333") for m in methods]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, mkey, mlabel in zip(axes, metric_keys, metric_labels):
        values = [ablation_results[method].get(mkey, 0) for method in methods]
        ax.bar(range(len(methods)), values, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], rotation=30, ha="right")
        set_paper_style(ax, title=mlabel, ylabel=mlabel)

    fig.suptitle("Ablation Study: Method Comparison", fontsize=13, fontweight="bold")
    save_figure(fig, save_path)


def fig_recacc_safepred(theta_results, save_path):
    """Dual-axis: RecAcc and SafePred across theta thresholds."""
    thetas = [r["theta"] for r in theta_results]
    rec_accs = [r["rec_acc"] for r in theta_results]
    safe_preds = [r["safe_pred_rate"] for r in theta_results]
    unsafe_rates = [r["unsafe_rate"] for r in theta_results]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    ax1.plot(thetas, rec_accs, "o-", color="#c62828", linewidth=2, markersize=6, label="RecAcc")
    ax1.plot(thetas, safe_preds, "s-", color="#2e7d32", linewidth=2, markersize=6, label="SafePred")
    ax1.set_xlabel(r"Uncertainty Threshold $\theta$", fontsize=11)
    ax1.set_ylabel("Rate", fontsize=11)
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.bar(thetas, unsafe_rates, width=0.04, alpha=0.3, color="#7f7f7f", label="Unsafe Rate")
    ax2.set_ylabel("Referral Ratio", fontsize=11)
    ax2.set_ylim(0, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="center right")

    set_paper_style(ax1, title="RecAcc & SafePred across Uncertainty Thresholds")
    ax1.set_xlim(min(thetas) - 0.03, max(thetas) + 0.03)
    save_figure(fig, save_path)


def main():
    p = argparse.ArgumentParser(description="Generate paper figures from experiment outputs")
    p.add_argument("--results_dir", type=str, required=True,
                   help="Training output dir (contains history.json, test_metrics.json)")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Figure output dir (default: results_dir/figures)")
    p.add_argument("--compare", nargs="*", default=[],
                   help="Additional result dirs for method comparison")
    args = p.parse_args()

    results_dir = args.results_dir
    output_dir = args.output_dir or os.path.join(results_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)

    history_path = os.path.join(results_dir, "history.json")
    metrics_path = os.path.join(results_dir, "test_metrics.json")

    if os.path.exists(history_path):
        fig_training_curves(history_path, os.path.join(output_dir, "fig_training_curves.png"))

    if os.path.exists(metrics_path):
        metrics = load_test_metrics(metrics_path)
        ev_keys = [k for k in metrics if k.startswith("evidence_class_")]
        if ev_keys:
            fig_evidence_distribution(
                {k: metrics[k] for k in ev_keys},
                os.path.join(output_dir, "fig_evidence_distribution.png"))

    if args.compare:
        all_metrics = {}
        all_cms = {}
        for comp_dir in args.compare:
            name = os.path.basename(comp_dir)
            mp = os.path.join(comp_dir, "test_metrics.json")
            if os.path.exists(mp):
                m = load_test_metrics(mp)
                all_metrics[name] = m
                if "confusion_matrix" in m:
                    all_cms[name] = m["confusion_matrix"]

        if all_cms:
            fig_confusion_matrices(all_cms, os.path.join(output_dir, "fig_confusion_matrices.png"))

        if len(all_metrics) >= 2:
            fig_ablation_bars(
                {name: {k: m.get(k, 0) for k in ["acc", "macro_f1", "qwk", "ece"]}
                 for name, m in all_metrics.items()},
                os.path.join(output_dir, "fig_ablation_bars.png"))

    print(f"[INFO] Figures saved to {output_dir}/")
    for f in sorted(os.listdir(output_dir)):
        print(f"  {f}")


if __name__ == "__main__":
    main()
