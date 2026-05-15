"""
Shared plotting utilities and consistent styling for paper figures.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "ce": "#7f7f7f",
    "edl": "#1f77b4",
    "orcu": "#ff7f0e",
    "edl_orcu": "#d62728",
    "evidence": ["#e8f5e9", "#a5d6a7", "#66bb6a", "#1b5e20"],
    "uncertainty": ["#fff3e0", "#ffcc80", "#ff9800", "#e65100"],
}

METHOD_LABELS = {
    "ce": "CE (Baseline)",
    "edl": "EDL Only",
    "orcu": "ORCU Only",
    "edl_orcu": "EDL+ORCU (Ours)",
}

GRADE_LABELS = ["Normal", "Mild", "Moderate", "Severe"]


def set_paper_style(ax, title=None, xlabel=None, ylabel=None):
    """Apply consistent paper-ready style to an axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--")


def save_figure(fig, path, dpi=300, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def plot_confusion_matrix(ax, cm, title=None, cmap="YlOrRd"):
    """Plot a single normalized confusion matrix on given axis."""
    n = cm.shape[0]
    im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(GRADE_LABELS[:n], rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(GRADE_LABELS[:n])
    for i in range(n):
        for j in range(n):
            color = "white" if cm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color=color, fontsize=9)
    if title:
        ax.set_title(title, fontsize=11)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    return im
