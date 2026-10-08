"""Figures (English labels, one style)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

BLUE, RED, GREY = "#4c72b0", "#d62728", "#8c8c8c"


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def roc_curves(curves: dict, path, title):
    """curves: label -> (y_true, mean out-of-fold probability, auc)."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for (label, (y, p, auc)), col in zip(curves.items(), (GREY, BLUE, RED, "#2ca02c")):
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=col, lw=2, label=f"{label} (AUC {auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("1 - specificity (false alarms)"); ax.set_ylabel("Sensitivity (cases caught)")
    ax.set_title(title, fontsize=10); ax.legend(fontsize=8, loc="lower right")
    _save(fig, path)


def auc_forest(table, path, title):
    """table: rows with features, model, auc (+ci) pooled and strat_auc (+ci) within image size."""
    t = table.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(t) + 1.7))
    y = np.arange(len(t))[::-1]
    ax.errorbar(t["auc"], y + 0.15, xerr=[t["auc"] - t["auc_ci95_low"], t["auc_ci95_high"] - t["auc"]], fmt="o", color=GREY, capsize=3,
                label="pooled (camera differences help)")
    ax.errorbar(t["strat_auc"], y - 0.15, xerr=[t["strat_auc"] - t["strat_auc_ci95_low"], t["strat_auc_ci95_high"] - t["strat_auc"]], fmt="o",
                color=BLUE, capsize=3, label="within the same image size")
    ax.set_yticks(y); ax.set_yticklabels([f"{f}  ·  {m}" for f, m in zip(t["features"], t["model"])], fontsize=8)
    ax.axvline(0.5, color="k", ls="--", lw=1); ax.set_xlabel("Cross-validated ROC-AUC (95 % interval)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False); ax.set_title(title, fontsize=10)
    _save(fig, path)


def permutation_hist(null, observed, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(null, bins=15, color=GREY, alpha=0.8, label="labels shuffled")
    ax.axvline(observed, color=RED, lw=2, label=f"real labels ({observed:.2f})")
    ax.set_xlabel("Cross-validated ROC-AUC"); ax.set_ylabel("Number of shuffles"); ax.legend(fontsize=8)
    ax.set_title("What does 'no signal' look like with this much data?", fontsize=10)
    _save(fig, path)


def confusion(cm, labels, path, title):
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]:.0f}", ha="center", va="center", color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted grade"); ax.set_ylabel("True grade"); ax.set_title(title, fontsize=10)
    _save(fig, path)


def shap_summary(shap_values, X, names, path, title):
    import shap
    plt.figure(figsize=(7, 5))
    shap.summary_plot(shap_values, X, feature_names=names, show=False, max_display=12)
    plt.title(title, fontsize=10)
    plt.tight_layout(); plt.savefig(path, dpi=140, bbox_inches="tight"); plt.close()
