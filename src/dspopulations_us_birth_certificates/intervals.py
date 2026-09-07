"""Project-wide posterior interval conventions.

The reduction itself is delegated to
``dse_research_utils.statistics.array_intervals.equal_tail_interval``; this
module keeps the project's own decisions on top of it — the coverage
restriction, the default NaN policy, the column labels and the *mean*-based
posterior summary.
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
from dse_research_utils.statistics.array_intervals import (
    equal_tail_interval as _shared_equal_tail_interval,
)

DEFAULT_INTERVAL_PROB = 0.89
DEFAULT_HPDI_PROB = DEFAULT_INTERVAL_PROB
DEFAULT_HDI_PROB = DEFAULT_INTERVAL_PROB
DEFAULT_ETI_PROB = DEFAULT_INTERVAL_PROB


def interval_label(prob: float = DEFAULT_INTERVAL_PROB) -> str:
    """Human-readable credible-interval probability label."""
    pct = prob * 100.0
    if abs(pct - round(pct)) < 1e-9:
        return f"{round(pct):.0f}%"
    return f"{pct:g}%"


def interval_percent(prob: float = DEFAULT_INTERVAL_PROB) -> int:
    """Rounded integer interval percentage for column names."""
    return int(round(prob * 100.0))


def interval_tail_probability(prob: float = DEFAULT_INTERVAL_PROB) -> float:
    """One-sided tail probability for an equal-tail interval."""
    if not 0.0 < prob < 1.0:
        raise ValueError(f"interval probability must lie in (0, 1), got {prob!r}")
    return (1.0 - prob) / 2.0


def eti_quantiles(prob: float = DEFAULT_ETI_PROB) -> tuple[float, float]:
    """Lower and upper quantiles for an equal-tail interval."""
    lo = interval_tail_probability(prob)
    return lo, 1.0 - lo


def equal_tail_interval(
    draws,
    *,
    prob: float = DEFAULT_ETI_PROB,
    axis=None,
    nan: bool = False,
):
    """Return lower/upper bounds for an equal-tail interval.

    Thin wrapper over the shared axis-aware reduction. What stays local:

    - the coverage restriction ``0 < prob < 1`` (the shared helper also
      accepts ``prob=1``, which this project has never used);
    - the NaN policy — ``nan=False`` propagates NaNs into the bounds (the
      default), ``nan=True`` omits them, matching the previous
      ``np.quantile``/``np.nanquantile`` pair. Infinities are retained by
      both, as before;
    - the scalar return for a full reduction, so ``axis=None`` still yields
      ``np.float64`` rather than a 0-d array.

    Two contracts come from the shared helper. Samples are converted to
    float64, so a float32 input no longer returns float32 bounds; and an
    empty (or entirely omitted) slice returns NaN bounds instead of raising.
    """
    # Keeps this project's coverage restriction; the shared helper allows 1.0.
    interval_tail_probability(prob)
    lo, hi = _shared_equal_tail_interval(
        draws,
        prob=prob,
        axis=axis,
        nonfinite="omit_nan" if nan else "propagate",
    )
    if lo.ndim == 0:
        return lo[()], hi[()]
    return lo, hi


def posterior_mean_eti(
    draws,
    *,
    prob: float = DEFAULT_ETI_PROB,
    nan: bool = False,
) -> dict[str, float]:
    """Mean and equal-tail interval for a flattened posterior draw array."""
    flat = np.asarray(draws, dtype=float).ravel()
    lo, hi = equal_tail_interval(flat, prob=prob, nan=nan)
    mean = np.nanmean(flat) if nan else np.mean(flat)
    return {"mean": float(mean), "lo": float(lo), "hi": float(hi)}


def normal_interval_z(prob: float = DEFAULT_ETI_PROB) -> float:
    """Normal-distribution z cutoff for a central interval."""
    alpha = interval_tail_probability(prob)
    return NormalDist().inv_cdf(1.0 - alpha)


__all__ = [
    "DEFAULT_ETI_PROB",
    "DEFAULT_HDI_PROB",
    "DEFAULT_HPDI_PROB",
    "DEFAULT_INTERVAL_PROB",
    "equal_tail_interval",
    "eti_quantiles",
    "interval_label",
    "interval_percent",
    "interval_tail_probability",
    "normal_interval_z",
    "posterior_mean_eti",
]
