from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.cluster import hierarchy


def _save_fig(fig: Figure, output_dir: str, file_name: str, dpi: int = 300) -> None:
    """Save a figure as PNG and SVG."""
    fig.savefig(f"{output_dir}/{file_name}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{output_dir}/{file_name}.svg", bbox_inches="tight")


def plot_roc_curve(
    fpr,
    tpr,
    model_idx: int,
    save: bool = False,
    output_dir: str = ".",
    file_name: str = "roc_curve",
) -> Figure:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, label="ROC curve")
    ax.plot([0, 1.0], [0, 1], "--", color="#999999", label="Random classifier")
    ax.set_xlim([-0.03, 1.03])
    ax.set_ylim([0, 1.03])
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title(f"Model {model_idx}: Receiver Operating Characteristic (ROC) Curve")
    ax.legend(loc="lower right")
    if save:
        _save_fig(fig, output_dir, file_name)
    return fig


def plot_precision_recall_curve(
    fpr,
    tpr,
    model_idx: int,
    save: bool = False,
    output_dir: str = ".",
    file_name: str = "precision_recall_curve",
) -> Figure:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, label="Precision-Recall curve")
    ax.set_xlim([-0.03, 1.03])
    ax.set_ylim([0, 1.03])
    ax.set_xlabel("Recall [TP / (TP + FN)]")
    ax.set_ylabel("Precision [TP / (TP + FP)]")
    ax.set_title(f"Model {model_idx}: Precision-Recall Curve")
    ax.legend(loc="lower right")
    if save:
        _save_fig(fig, output_dir, file_name)
    return fig


def plot_permutation_importances(
    result,
    X_eval,
    model_idx: int,
    save: bool = False,
    output_dir: str = ".",
    file_name: str = "permutation_importances",
) -> Figure:
    sorted_importances_idx = result.importances_mean.argsort()

    importances = pd.DataFrame(
        result.importances[sorted_importances_idx].T,
        columns=X_eval.columns[sorted_importances_idx],
    )
    x_size = max(4, min(6, 0.3 * importances.shape[1]))
    ax = importances.plot.box(vert=False, whis=10, figsize=(6, x_size))
    ax.set_title(f"Model {model_idx}: Permutation importances")
    ax.axvline(x=0, color="k", linestyle="--")
    ax.set_xlabel("Decrease in average precision")
    ax.set_ylabel("Predictor variable")

    fig = ax.figure
    if save:
        _save_fig(fig, output_dir, file_name)
    return fig


def plot_dendrogram(
    linkage,
    labels,
    model_idx: int,
    save: bool = False,
    output_dir: str = ".",
    file_name: str = "dendrogram",
) -> tuple[Figure, dict]:
    """Draw a right-oriented dendrogram.

    Returns both the figure and the scipy ``dendrogram`` dict (needed by
    ``plot_correlation_heatmap`` for its leaf ordering).
    """
    xsize = 6
    ysize = max(6, min(15, 0.25 * len(labels)))

    fig, ax = plt.subplots(figsize=(xsize, ysize))

    dendro = hierarchy.dendrogram(
        linkage,
        labels=labels,
        orientation="right",
        ax=ax,
    )

    ax.vlines(0.5, 0, 500, linestyle="--", color="#b2b4549f", linewidth=2)
    ax.set_xlabel("Linkage distance (increase in within-cluster variance)")
    ax.set_ylabel("Predictors")
    ax.set_title(f"Model {model_idx}: Hierarchical clustering of predictors")

    if save:
        _save_fig(fig, output_dir, file_name)
    return fig, dendro


def plot_correlation_heatmap(
    corr,
    dendro,
    label_threshold: float = 0.30,
    model_idx: int | Any = None,
    save: bool = False,
    output_dir: str = ".",
    file_name: str = "correlation_heatmap",
) -> Figure:
    C = corr[dendro["leaves"], :][:, dendro["leaves"]]
    labels = dendro["ivl"]
    dendro_idx = np.arange(len(labels))

    ysize = max(6, min(15, 0.4 * C.shape[0]))
    xsize = ysize

    with plt.rc_context(
        {"ytick.labelsize": 12, "xtick.labelsize": 12, "axes.titlesize": 12}
    ):
        fig, ax = plt.subplots(figsize=(xsize, ysize))
        im = ax.imshow(C, cmap="viridis")

        ax.set_title(f"Model {model_idx}: Correlation heatmap of predictors")
        ax.set_xticks(dendro_idx)
        ax.set_yticks(dendro_idx)
        ax.set_xticklabels(labels, rotation="vertical")
        ax.set_yticklabels(labels)

        n = C.shape[0]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if abs(C[i, j]) < label_threshold:
                    continue

                ax.text(
                    j,
                    i,
                    f"{C[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if abs(C[i, j]) < 0.6 else "black",
                )

        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.025)

        if save:
            _save_fig(fig, output_dir, file_name)
    return fig


def plot_shap_bar(
    explanation,
    model_idx: int,
    max_display: int = 45,
    save: bool = False,
    output_dir: str = ".",
    file_name: str | None = None,
) -> Figure:
    import shap

    file_name = file_name or f"model_{model_idx}_shap_bar"
    with plt.rc_context({"axes.titlesize": 12}):
        fig, ax = plt.subplots(figsize=(8, 12))
        ax.set_title(f"Model {model_idx}: SHAP values for predictor variables")
        shap.plots.bar(explanation, max_display=max_display, ax=ax, show=False)
        if save:
            _save_fig(fig, output_dir, file_name)
    return fig


def plot_shap_beeswarm(
    explanation,
    model_idx: int,
    max_display: int = 45,
    plot_size: tuple[int, int] = (8, 10),
    save: bool = False,
    output_dir: str = ".",
    file_name: str | None = None,
) -> Figure:
    import shap

    file_name = file_name or f"model_{model_idx}_shap_beeswarm"
    with plt.rc_context({"axes.titlesize": 12}):
        shap.plots.beeswarm(
            explanation, max_display=max_display, plot_size=plot_size, show=False
        )
        fig = plt.gcf()
        fig.suptitle(f"Model {model_idx}: SHAP values for predictor variables")
        if save:
            _save_fig(fig, output_dir, file_name)
    return fig


def plot_shap_scatter(
    explanation,
    feature_x: str,
    feature_color: str,
    model_idx: int,
    save: bool = False,
    output_dir: str = ".",
    file_name: str | None = None,
) -> Figure:
    import shap

    file_name = file_name or f"model_{model_idx}_shap_{feature_x}_vs_{feature_color}"
    shap.plots.scatter(
        explanation[:, feature_x], color=explanation[:, feature_color], show=False
    )
    fig = plt.gcf()
    if save:
        _save_fig(fig, output_dir, file_name)
    return fig
