from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.cluster import hierarchy


def save_fig(
    fig: Figure,
    output_dir: str,
    file_name: str,
    *,
    dpi: int = 300,
    data: pd.DataFrame | None = None,
) -> None:
    """Save a figure as PNG + SVG, plus a companion CSV of the plotted data.

    The CSV is written to ``{output_dir}/{file_name}.csv`` and carries the
    arrays actually rendered on the plot. Keeping image and data at the
    same stem lets readers re-plot or validate the figure without re-running
    the upstream pipeline.
    """
    fig.savefig(f"{output_dir}/{file_name}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{output_dir}/{file_name}.svg", bbox_inches="tight")
    if data is not None:
        data.to_csv(f"{output_dir}/{file_name}.csv", index=False)


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
        data = pd.DataFrame({"fpr": np.asarray(fpr), "tpr": np.asarray(tpr)})
        save_fig(fig, output_dir, file_name, data=data)
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
        # Callers pass (recall, precision) as the positional args — keep the
        # historical parameter names but label the CSV columns correctly.
        data = pd.DataFrame({"recall": np.asarray(fpr), "precision": np.asarray(tpr)})
        save_fig(fig, output_dir, file_name, data=data)
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
        # Long format so each (feature, repeat) pair is one row — reproduces
        # the boxplot inputs exactly.
        data = importances.melt(var_name="feature", value_name="importance")
        data["repeat"] = data.groupby("feature").cumcount()
        data = data[["feature", "repeat", "importance"]]
        save_fig(fig, output_dir, file_name, data=data)
    return fig


def plot_grouped_permutation_importances(
    grouped_importance: pd.DataFrame,
    model_idx: int,
    *,
    max_display: int = 25,
    save: bool = False,
    output_dir: str = ".",
    file_name: str = "grouped_permutation_importances",
) -> Figure:
    """Plot grouped permutation importance as a horizontal bar chart."""
    display = (
        grouped_importance.sort_values("importance_mean", ascending=False)
        .head(max_display)
        .copy()
    )
    display = display.iloc[::-1]
    labels = display["group"].astype(str)
    if "dimension_hint" in display:
        labels = labels + " (" + display["dimension_hint"].astype(str) + ")"

    y_size = max(4, min(10, 0.35 * max(len(display), 1)))
    fig, ax = plt.subplots(figsize=(8, y_size))
    ax.barh(labels, display["importance_mean"])
    ax.axvline(x=0, color="k", linestyle="--")
    ax.set_title(f"Model {model_idx}: grouped permutation importances")
    ax.set_xlabel("Decrease in average precision")
    ax.set_ylabel("Feature group")

    if save:
        save_fig(fig, output_dir, file_name, data=grouped_importance)
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
        # Leaf order is the main thing a downstream reader wants; the full
        # linkage matrix is also useful for reconstruction.
        leaf_data = pd.DataFrame(
            {
                "position": np.arange(len(dendro["ivl"])),
                "label": dendro["ivl"],
                "leaf_index": dendro["leaves"],
            }
        )
        linkage_arr = np.asarray(linkage)
        linkage_frame = pd.DataFrame(
            linkage_arr,
            columns=["left", "right", "distance", "n_obs"][: linkage_arr.shape[1]],
        )
        linkage_frame.insert(0, "merge_index", np.arange(len(linkage_frame)))
        data = pd.concat(
            [
                leaf_data.assign(kind="leaf"),
                linkage_frame.assign(kind="merge"),
            ],
            ignore_index=True,
        )
        save_fig(fig, output_dir, file_name, data=data)
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
            # Wide CSV with the reordered correlation matrix — columns and
            # rows in the order shown on the heatmap.
            data = pd.DataFrame(C, index=labels, columns=labels)
            data.index.name = "feature"
            data = data.reset_index()
            save_fig(fig, output_dir, file_name, data=data)
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
            data = _shap_bar_data(explanation)
            save_fig(fig, output_dir, file_name, data=data)
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
            data = _shap_beeswarm_data(explanation)
            save_fig(fig, output_dir, file_name, data=data)
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
        data = _shap_scatter_data(explanation, feature_x, feature_color)
        save_fig(fig, output_dir, file_name, data=data)
    return fig


# ---------------------------------------------------------------------------
# SHAP → DataFrame helpers
# ---------------------------------------------------------------------------


def _shap_feature_names(explanation) -> list[str]:
    names = getattr(explanation, "feature_names", None)
    if names is None:
        values = np.asarray(explanation.values)
        return [f"f_{i}" for i in range(values.shape[-1])]
    return list(names)


def _shap_bar_data(explanation) -> pd.DataFrame:
    values = np.asarray(explanation.values)
    if values.ndim != 2:  # collapse to (n, features) if shap returned 1-D
        values = values.reshape(-1, values.shape[-1])
    mean_abs = np.mean(np.abs(values), axis=0)
    names = _shap_feature_names(explanation)
    return (
        pd.DataFrame({"feature": names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def _shap_beeswarm_data(explanation) -> pd.DataFrame:
    values = np.asarray(explanation.values)
    data_vals = getattr(explanation, "data", None)
    data_arr = np.asarray(data_vals) if data_vals is not None else None
    names = _shap_feature_names(explanation)
    if values.ndim != 2:
        values = values.reshape(-1, values.shape[-1])
    rows, cols = values.shape
    sample_idx = np.repeat(np.arange(rows), cols)
    feature_idx = np.tile(np.arange(cols), rows)
    frame = pd.DataFrame(
        {
            "sample_index": sample_idx,
            "feature": [names[k] for k in feature_idx],
            "shap_value": values.ravel(),
        }
    )
    if data_arr is not None and data_arr.shape == values.shape:
        frame["feature_value"] = data_arr.ravel()
    return frame


def _shap_scatter_data(explanation, feature_x: str, feature_color: str) -> pd.DataFrame:
    names = _shap_feature_names(explanation)
    values = np.asarray(explanation.values)
    data_vals = getattr(explanation, "data", None)
    data_arr = np.asarray(data_vals) if data_vals is not None else None
    if feature_x not in names:
        raise ValueError(f"feature_x {feature_x!r} not in explanation feature_names")
    if feature_color not in names:
        raise ValueError(
            f"feature_color {feature_color!r} not in explanation feature_names"
        )
    ix = names.index(feature_x)
    ic = names.index(feature_color)
    frame = pd.DataFrame(
        {
            "sample_index": np.arange(values.shape[0]),
            f"shap_{feature_x}": values[:, ix],
        }
    )
    if data_arr is not None:
        frame[feature_x] = data_arr[:, ix]
        frame[feature_color] = data_arr[:, ic]
    return frame
