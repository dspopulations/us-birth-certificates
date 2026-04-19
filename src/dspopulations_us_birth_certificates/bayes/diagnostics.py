"""Convergence diagnostics and summary tables."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    import arviz as az


DEFAULT_SUMMARY_VAR_NAMES = ("alpha", "ls_year", "eta_year", "ls_age", "eta_age")


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
