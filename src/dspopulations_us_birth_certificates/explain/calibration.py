"""Classification calibration and top-K evaluation helpers.

Dependency-light (numpy + pandas only) so they can be reused from
notebooks, scripts, or tests without pulling in LightGBM.
"""

from __future__ import annotations

import os
from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_KS: tuple[int, ...] = (100, 500, 1000, 5000, 10000, 20000, 50000)
DEFAULT_TAIL_FRACS: tuple[float, ...] = (1e-2, 1e-3, 1e-4, 1e-5)


def precision_recall_at_k(
    y_true,
    y_score,
    ks: Iterable[int] = DEFAULT_KS,
) -> pd.DataFrame:
    """Precision@K and Recall@K for a binary classifier ranking.

    Returns a frame with columns ``K, tp, precision_at_k, recall_at_k``.
    Ties are broken by a stable mergesort on ``-y_score``.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_score_arr = np.asarray(y_score, dtype=float)

    if (
        y_true_arr.ndim != 1
        or y_score_arr.ndim != 1
        or y_true_arr.shape[0] != y_score_arr.shape[0]
    ):
        raise ValueError("y_true and y_score must be 1D arrays of the same length.")

    n = y_true_arr.shape[0]
    pos_total = int(y_true_arr.sum())
    if pos_total == 0:
        raise ValueError("y_true contains no positives; recall is undefined.")

    order = np.argsort(-y_score_arr, kind="mergesort")
    y_sorted = y_true_arr[order]
    ctp = np.cumsum(y_sorted)

    rows = []
    for K in ks:
        k = int(min(max(K, 1), n))
        tp = int(ctp[k - 1])
        rows.append((k, tp, tp / k, tp / pos_total))

    return pd.DataFrame(rows, columns=["K", "tp", "precision_at_k", "recall_at_k"])


def tail_calibration_table(
    y_true,
    y_score,
    fracs: Iterable[float] = DEFAULT_TAIL_FRACS,
) -> pd.DataFrame:
    """Compare predicted-vs-observed event rates in the top ``fracs`` of scores.

    Returns a frame with columns ``top_frac, k, pred_rate_mean, obs_rate, tp,
    fp, pred_minus_obs, ratio_pred_to_obs``.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_score, dtype=float)

    if y.ndim != 1 or p.ndim != 1 or y.shape[0] != p.shape[0]:
        raise ValueError("y_true and y_score must be 1D arrays of the same length.")

    order = np.argsort(-p)
    y_sorted = y[order]
    p_sorted = p[order]

    n = len(y)
    rows = []
    for f in fracs:
        k = max(1, int(round(n * f)))
        y_top = y_sorted[:k]
        p_top = p_sorted[:k]
        obs_rate = float(y_top.mean())
        pred_rate = float(p_top.mean())
        tp = int(y_top.sum())
        fp = int(k - tp)
        rows.append(
            {
                "top_frac": float(f),
                "k": k,
                "pred_rate_mean": pred_rate,
                "obs_rate": obs_rate,
                "tp": tp,
                "fp": fp,
                "pred_minus_obs": pred_rate - obs_rate,
                "ratio_pred_to_obs": (pred_rate / obs_rate) if obs_rate > 0 else np.nan,
            }
        )

    return pd.DataFrame(rows)


def plot_precision_recall_at_k_curve(
    pr_df: pd.DataFrame,
    title_prefix: str = "Validation",
    *,
    save: bool = False,
    output_dir: str = ".",
    file_stem: str = "precision_recall_at_k",
    show: bool = True,
) -> None:
    """Plot Precision@K and Recall@K vs K (log x-axis).

    Two figures are produced (precision, recall). When ``save`` is True,
    writes PNG + SVG to ``output_dir`` using ``file_stem`` as a basename
    (``_precision`` / ``_recall`` suffixes).
    """
    ks = pr_df["K"].to_numpy()
    prec = pr_df["precision_at_k"].to_numpy()
    rec = pr_df["recall_at_k"].to_numpy()

    if save:
        os.makedirs(output_dir, exist_ok=True)

    for metric_name, values, suffix in (
        ("Precision@K", prec, "precision"),
        ("Recall@K", rec, "recall"),
    ):
        plt.figure(figsize=(9, 5))
        plt.plot(ks, values, marker="o")
        plt.xscale("log")
        plt.xlabel("K (top-K flagged; log scale)")
        plt.ylabel(metric_name)
        plt.title(f"{title_prefix}: {metric_name}")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        if save:
            plt.savefig(
                f"{output_dir}/{file_stem}_{suffix}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.savefig(
                f"{output_dir}/{file_stem}_{suffix}.svg", bbox_inches="tight"
            )
        if show:
            plt.show()
        else:
            plt.close()
