"""SHAP TreeExplainer helpers.

Wraps the repetitive ``shap.TreeExplainer`` + ``shap.plots.bar /
beeswarm / scatter`` calls from notebooks and scripts into reusable
helpers with simple, explicit signatures (booster + X_eval).

A thin ``ModelFitContext``-aware layer will be added in step 4 once the
pipeline lands.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

if TYPE_CHECKING:
    import lightgbm as lgb


def compute_explanation(booster: lgb.Booster, X_eval: pd.DataFrame) -> shap.Explanation:
    """Return a ``shap.Explanation`` for ``X_eval`` using a LightGBM booster.

    Uses tree-path-dependent feature perturbation and raw model output, matching
    the convention used in the existing notebooks.
    """
    explainer = shap.TreeExplainer(
        booster, feature_perturbation="tree_path_dependent", model_output="raw"
    )
    return explainer(X_eval)


def shap_importance(explanation: shap.Explanation, feature_names) -> pd.DataFrame:
    """Mean absolute SHAP value per feature, sorted descending."""
    shap_values = explanation.values
    return pd.DataFrame(
        {
            "feature": list(feature_names),
            "mean_abs_shap": np.mean(np.abs(shap_values), axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)


def _maybe_save(output_dir: str, file_stem: str, save: bool) -> None:
    if not save:
        return
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{file_stem}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_dir}/{file_stem}.svg", bbox_inches="tight")


def plot_bar(
    explanation: shap.Explanation,
    *,
    model_idx: int = 0,
    max_display: int = 40,
    save: bool = False,
    output_dir: str = ".",
    file_stem: str = "shap_bar",
    show: bool = True,
    figsize: tuple[float, float] | None = None,
) -> None:
    """SHAP bar plot of mean absolute values."""
    fig_size = figsize or (8, max(6, max_display * 0.28))
    with plt.rc_context({"axes.titlesize": 12}):
        fig = plt.figure(figsize=fig_size)
        ax = fig.subplots()
        ax.set_title(f"Model {model_idx}: SHAP values for predictor variables")
        shap.plots.bar(explanation, max_display=max_display, ax=ax)
        _maybe_save(output_dir, file_stem, save)
        if show:
            plt.show()
        else:
            plt.close(fig)


def plot_beeswarm(
    explanation: shap.Explanation,
    *,
    model_idx: int = 0,
    max_display: int = 40,
    save: bool = False,
    output_dir: str = ".",
    file_stem: str = "shap_beeswarm",
    show: bool = True,
    plot_size: tuple[float, float] = (8, 9),
) -> None:
    """SHAP beeswarm plot of per-row SHAP values."""
    with plt.rc_context({"axes.titlesize": 12}):
        fig = plt.figure()
        ax = fig.subplots()
        ax.set_title(f"Model {model_idx}: SHAP values for predictor variables")
        shap.plots.beeswarm(
            explanation, max_display=max_display, plot_size=plot_size
        )
        _maybe_save(output_dir, file_stem, save)
        if show:
            plt.show()
        else:
            plt.close(fig)


def plot_scatter(
    explanation: shap.Explanation,
    x_feature: str,
    colour_feature: str | None = None,
    *,
    model_idx: int = 0,
    save: bool = False,
    output_dir: str = ".",
    file_stem: str | None = None,
    show: bool = True,
) -> None:
    """SHAP scatter for ``x_feature``, optionally coloured by ``colour_feature``.

    ``file_stem`` defaults to ``shap_<x>_vs_<colour>`` (or ``shap_<x>`` when no
    colour feature is given).
    """
    if file_stem is None:
        file_stem = (
            f"shap_{x_feature}_vs_{colour_feature}"
            if colour_feature
            else f"shap_{x_feature}"
        )
    colour = explanation[:, colour_feature] if colour_feature else None
    shap.plots.scatter(explanation[:, x_feature], color=colour, show=False)
    plt.title(f"Model {model_idx}: SHAP scatter — {x_feature}")
    _maybe_save(output_dir, file_stem, save)
    if show:
        plt.show()
    else:
        plt.close()
