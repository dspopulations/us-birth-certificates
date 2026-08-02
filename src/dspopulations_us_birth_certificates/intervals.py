"""Project-wide posterior interval conventions."""

from __future__ import annotations

from statistics import NormalDist

import numpy as np

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
    """Return lower/upper bounds for an equal-tail interval."""
    lo_q, hi_q = eti_quantiles(prob)
    quantile = np.nanquantile if nan else np.quantile
    return quantile(draws, lo_q, axis=axis), quantile(draws, hi_q, axis=axis)


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
