"""Classification calibration and top-K evaluation helpers.

These functions are lifted verbatim from
``notebooks/00010-predictors-10-c.py`` in refactor step 2 and unit-tested.
Kept dependency-light (numpy + pandas only) so they can be reused from
notebooks, scripts, or tests without pulling in LightGBM.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


def precision_recall_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    ks: Iterable[int] = (100, 500, 1000, 5000, 10000, 20000, 50000),
) -> pd.DataFrame:
    """Precision@K and Recall@K for a binary classifier ranking.

    Returns a frame with columns ``K, tp, precision_at_k, recall_at_k``.
    Ties are broken by a stable mergesort on ``-y_score``.
    """
    raise NotImplementedError("populated in refactor step 2")


def tail_calibration_table(
    y_true: np.ndarray,
    y_score: np.ndarray,
    fracs: Iterable[float] = (1e-2, 1e-3, 1e-4, 1e-5),
) -> pd.DataFrame:
    """Compare predicted-vs-observed event rates in the top ``fracs`` of scores.

    Returns a frame with columns
    ``top_frac, k, pred_rate_mean, obs_rate, tp, fp, pred_minus_obs,
    ratio_pred_to_obs``.
    """
    raise NotImplementedError("populated in refactor step 2")
