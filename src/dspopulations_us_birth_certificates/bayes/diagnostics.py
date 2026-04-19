"""Convergence diagnostics and summary tables."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import arviz as az


DEFAULT_SUMMARY_VAR_NAMES = ("alpha", "ls_t", "eta_t", "ls_age", "eta_age")

PRIOR_PREDICTIVE_COLUMNS = ("unit", "median", "hdi_lo", "hdi_hi")


def summary_table(
    idata: az.InferenceData,
    *,
    var_names: tuple[str, ...] | None = None,
    hdi_prob: float = 0.94,
) -> pd.DataFrame:
    """Return ``az.summary`` as a DataFrame keyed on the variable name.

    If ``var_names`` is None, the full summary is returned (may be large
    for models with per-cell deterministics).
    """
    import arviz as az

    if var_names is None:
        return az.summary(idata, hdi_prob=hdi_prob)
    available = [
        n
        for n in var_names
        if n in idata.posterior.data_vars  # type: ignore[attr-defined]
    ]
    if not available:
        return az.summary(idata, hdi_prob=hdi_prob)
    return az.summary(idata, var_names=list(available), hdi_prob=hdi_prob)


def convergence_health(
    summary: pd.DataFrame,
    *,
    rhat_threshold: float = 1.01,
    ess_threshold: float = 400.0,
) -> dict[str, Any]:
    """Roll up Rhat/ESS from a summary table into a pass/fail verdict."""
    rhat_col = "r_hat" if "r_hat" in summary.columns else "rhat"
    ess_cols = [c for c in ("ess_bulk", "ess_tail") if c in summary.columns]

    max_rhat = (
        float(summary[rhat_col].max()) if rhat_col in summary.columns else float("nan")
    )
    min_ess = float(summary[ess_cols].min().min()) if ess_cols else float("nan")

    rhat_ok = max_rhat < rhat_threshold if max_rhat == max_rhat else False
    ess_ok = min_ess >= ess_threshold if min_ess == min_ess else False

    return {
        "max_rhat": max_rhat,
        "min_ess": min_ess,
        "rhat_threshold": rhat_threshold,
        "ess_threshold": ess_threshold,
        "rhat_ok": rhat_ok,
        "ess_ok": ess_ok,
        "all_ok": rhat_ok and ess_ok,
    }


def _hdi_of(flat: np.ndarray, hdi_prob: float) -> tuple[float, float, float]:
    """Return (median, hdi_lo, hdi_hi) for a 1-D array of prior samples."""
    import arviz as az

    flat = np.asarray(flat).reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return float("nan"), float("nan"), float("nan")
    interval = az.hdi(flat, hdi_prob=hdi_prob)
    lo, hi = float(interval[0]), float(interval[1])
    return float(np.median(flat)), lo, hi


def prior_predictive_summary(
    idata: az.InferenceData,
    cells: pd.DataFrame,
    *,
    time_coord: str = "t",
    age_coord: str = "age",
    hdi_prob: float = 0.94,
) -> pd.DataFrame:
    """Summarise prior-implied quantities on rate / ratio scales.

    Rows (indexed by ``check``):
        - ``baseline_rate``: ``sigmoid(alpha)`` — prior on the intercept rate.
        - ``cell_rate_mean``: exposure-weighted mean prior rate across cells.
        - ``y_cell_min_exposure`` / ``y_cell_max_exposure``: prior-predictive
          counts for the smallest and largest cell.
        - ``time_trend_rate_ratio``: ``exp(max(f_t) - min(f_t))`` — the rate
          multiplier the prior admits across the time range.
        - ``age_gradient_rate_ratio``: same for ``f_age``.
        - ``ls_t_coord_units`` / ``ls_age_coord_units``: HSGP length-scales
          translated back to original coord units (years).

    Rows are skipped silently if the corresponding variable is absent from
    the InferenceData — callers can rely on the returned DataFrame matching
    whatever the model actually produced.

    Args:
        idata: Must carry ``prior`` and ``prior_predictive`` groups.
        cells: Cell frame (must contain the coord columns).
        time_coord / age_coord: Column names in ``cells`` for each coord.
        hdi_prob: Credible-interval width for the HDI columns.

    Returns:
        DataFrame with columns ``unit, median, hdi_lo, hdi_hi``, indexed by
        ``check``.
    """
    rows: list[dict[str, Any]] = []
    prior = idata.prior  # type: ignore[attr-defined]
    pp = getattr(idata, "prior_predictive", None)

    if "alpha" in prior.data_vars:
        alpha = np.asarray(prior["alpha"].values)
        baseline = 1.0 / (1.0 + np.exp(-alpha))
        med, lo, hi = _hdi_of(baseline, hdi_prob)
        rows.append(
            {"check": "baseline_rate", "unit": "rate", "median": med, "hdi_lo": lo, "hdi_hi": hi}
        )

    if "p" in prior.data_vars and "n_cell" in cells.columns:
        p = np.asarray(prior["p"].values)  # (chain, draw, cell)
        n = cells["n_cell"].to_numpy(dtype=float)
        denom = float(n.sum()) or 1.0
        weighted = (p * n[None, None, :]).sum(axis=-1) / denom
        med, lo, hi = _hdi_of(weighted, hdi_prob)
        rows.append(
            {
                "check": "cell_rate_mean",
                "unit": "rate",
                "median": med,
                "hdi_lo": lo,
                "hdi_hi": hi,
            }
        )

    if pp is not None and "y_obs" in pp.data_vars and "n_cell" in cells.columns:
        y = np.asarray(pp["y_obs"].values)  # (chain, draw, cell)
        n = cells["n_cell"].to_numpy()
        if n.size:
            i_min = int(np.argmin(n))
            i_max = int(np.argmax(n))
            for label, idx in (
                ("y_cell_min_exposure", i_min),
                ("y_cell_max_exposure", i_max),
            ):
                med, lo, hi = _hdi_of(y[..., idx], hdi_prob)
                rows.append(
                    {
                        "check": label,
                        "unit": f"count (n_cell={int(n[idx])})",
                        "median": med,
                        "hdi_lo": lo,
                        "hdi_hi": hi,
                    }
                )

    for smooth_name, label in (
        ("f_t", "time_trend_rate_ratio"),
        ("f_age", "age_gradient_rate_ratio"),
    ):
        if smooth_name in prior.data_vars:
            f = np.asarray(prior[smooth_name].values)  # (chain, draw, coord)
            if f.ndim >= 3:
                rng = np.exp(f.max(axis=-1) - f.min(axis=-1))
                med, lo, hi = _hdi_of(rng, hdi_prob)
                rows.append(
                    {
                        "check": label,
                        "unit": "rate ratio",
                        "median": med,
                        "hdi_lo": lo,
                        "hdi_hi": hi,
                    }
                )

    for var_name, coord_col, label in (
        ("ls_t", time_coord, "ls_t_coord_units"),
        ("ls_age", age_coord, "ls_age_coord_units"),
    ):
        if var_name in prior.data_vars and coord_col in cells.columns:
            coord = cells[coord_col].to_numpy(dtype=float)
            std = float(coord.std())
            scale = std if std > 0 else 1.0
            ls_scaled = np.asarray(prior[var_name].values) * scale
            med, lo, hi = _hdi_of(ls_scaled, hdi_prob)
            rows.append(
                {
                    "check": label,
                    "unit": f"{coord_col} units",
                    "median": med,
                    "hdi_lo": lo,
                    "hdi_hi": hi,
                }
            )

    return pd.DataFrame(rows, columns=["check", *PRIOR_PREDICTIVE_COLUMNS]).set_index(
        "check"
    )


def loo_table(idata: az.InferenceData) -> pd.DataFrame:
    """Leave-one-out CV summary as a 1-row DataFrame."""
    import arviz as az

    loo = az.loo(idata, pointwise=False)
    return pd.DataFrame(
        {
            "elpd_loo": [float(loo.elpd_loo)],
            "se": [float(loo.se)],
            "p_loo": [float(loo.p_loo)],
            "n_eff": [float(getattr(loo, "n_samples", float("nan")))],
        }
    )
