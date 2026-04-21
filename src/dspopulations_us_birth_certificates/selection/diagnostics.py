"""Posterior diagnostics for the three-stage selection model.

Each function takes a fitted ``az.InferenceData`` plus (where relevant)
the aggregated ``cells`` frame that produced it, and returns a
:class:`matplotlib.figure.Figure`. Callers wanting paired CSVs /
publication-quality artefacts should use
:mod:`dspopulations_us_birth_certificates.selection.render_diagnostics`,
which wraps these with a ``_save`` helper that writes figures alongside
their tidy-DataFrame companions.

Functions
---------
- :func:`identifiability_pairplot` — posterior pair-plot of race effects
  on ``eta_term`` vs ``s``. Correlation ``|r| > 0.7`` indicates the
  decomposition is prior-driven rather than data-identified (plan §4.3,
  §10 #4).
- :func:`eta_term_year_trajectory_plot` — posterior trajectory of
  ``eta_term_year`` by year. Drift across the window is a residual
  year-over-year effect on termination rates.
- :func:`cchd_consistency_check` — posterior CCHD co-occurrence among
  true DS livebirths vs the EUROCAT published prevalence (~22.5%).
- :func:`posterior_predictive_by_stratum` — recorded-count PPC plot
  aggregated by a chosen stratum (year / race / age).
- :func:`decomposition_by_race` — posterior stacked estimate of true
  DS livebirths, recorded, prenatally terminated, and missed, by race.
- :func:`age_curve_check` — posterior ``theta_LB`` by age band vs the
  Morris/de Graaf prior means (sanity-check that Stage 1 is not being
  pulled around by data fitting).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from dspopulations_us_birth_certificates.selection.priors import (
    AGE_LEVELS,
    MORRIS_THETA_LB_PER_1000,
    RACE_LEVELS,
    inv_logit,
)

if TYPE_CHECKING:
    import arviz as az
    from matplotlib.figure import Figure


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _styles():
    """Return the shared plot-style module (lazy import for test speed)."""
    import dse_research_utils.plot.styles as plot_styles

    return plot_styles


def _quantile(arr: np.ndarray, q: float) -> np.ndarray:
    """Chain+draw quantile along the leading two axes."""
    return np.quantile(arr.reshape(-1, *arr.shape[2:]), q, axis=0)


# --------------------------------------------------------------------------- #
# 1. Identifiability pair-plot                                                #
# --------------------------------------------------------------------------- #


def identifiability_pairplot(
    idata: az.InferenceData,
) -> Figure:
    """Per-race pair-plot of ``eta_term_race`` vs ``s_race`` draws.

    A high negative correlation means the data cannot distinguish
    lower termination from lower sensitivity — the decomposition is
    prior-driven. Plot title reports ``|r|`` per race so readers can
    read severity directly off the figure.
    """
    import matplotlib.pyplot as plt

    styles = _styles()
    post = idata.posterior
    if "eta_term_race" not in post.data_vars or "s_race" not in post.data_vars:
        raise ValueError(
            "InferenceData must carry 'eta_term_race' and 's_race'. "
            "Re-fit with spec='full' or spec='single_eta'."
        )

    eta = np.asarray(post["eta_term_race"].values)  # (chain, draw, race)
    s = np.asarray(post["s_race"].values)
    n_race = eta.shape[-1]
    labels = RACE_LEVELS[:n_race]

    n_cols = min(3, n_race)
    n_rows = int(np.ceil(n_race / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(
            styles.FIGSIZE_LG[0] * n_cols / 2.0,
            styles.FIGSIZE_MD[1] * n_rows / 1.2,
        ),
        squeeze=False,
    )

    for idx in range(n_race):
        ax = axes.flat[idx]
        x = eta[..., idx].ravel()
        y = s[..., idx].ravel()
        ax.scatter(x, y, s=2, alpha=0.2, color=styles.COLOUR_BLUE)
        r = float(np.corrcoef(x, y)[0, 1])
        prior_driven = "prior-driven" if abs(r) > 0.7 else "data-informed"
        ax.set_title(f"{labels[idx]}\n|r|={abs(r):.2f} ({prior_driven})")
        ax.set_xlabel(r"$\eta_{term}$ race effect")
        ax.set_ylabel("s race effect")
        ax.axhline(0, color=styles.TEXT_COLOUR, lw=0.5, alpha=0.5)
        ax.axvline(0, color=styles.TEXT_COLOUR, lw=0.5, alpha=0.5)
    for idx in range(n_race, n_rows * n_cols):
        axes.flat[idx].set_visible(False)

    fig.suptitle(
        r"Posterior identifiability: $\eta_{term}$ vs $s$ race effects",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def identifiability_table(idata: az.InferenceData) -> pd.DataFrame:
    """Per-race posterior correlations between ``eta_term_race`` and ``s_race``.

    Returned as a tidy DataFrame so the rendering CLI can save it
    alongside the figure.
    """
    post = idata.posterior
    eta = np.asarray(post["eta_term_race"].values)
    s = np.asarray(post["s_race"].values)
    n_race = eta.shape[-1]
    rows = []
    for idx in range(n_race):
        x = eta[..., idx].ravel()
        y = s[..., idx].ravel()
        r = float(np.corrcoef(x, y)[0, 1])
        rows.append(
            {
                "race_idx": idx,
                "race": RACE_LEVELS[idx] if idx < len(RACE_LEVELS) else f"idx_{idx}",
                "correlation": r,
                "abs_correlation": abs(r),
                "interpretation": (
                    "prior-driven" if abs(r) > 0.7 else "data-informed"
                ),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 2. Year-trajectory plot for eta_term                                        #
# --------------------------------------------------------------------------- #


def eta_term_year_trajectory_plot(
    idata: az.InferenceData,
    *,
    hdi_prob: float = 0.94,
) -> Figure:
    """Posterior trajectory of ``eta_term_year`` by year.

    Year effects on termination are modelled with a single
    homoscedastic sigma; this plot shows the per-year posterior means
    with credible intervals so any residual year drift is visible.
    """
    import matplotlib.pyplot as plt

    styles = _styles()
    post = idata.posterior
    if "eta_term_year" not in post.data_vars:
        raise ValueError(
            "InferenceData is missing 'eta_term_year' — fit with spec='full'."
        )

    year_arr = np.asarray(post["eta_term_year"].values)  # (chain, draw, year)
    n_year = year_arr.shape[-1]

    mean = year_arr.mean(axis=(0, 1))
    lo = _quantile(year_arr, (1 - hdi_prob) / 2)
    hi = _quantile(year_arr, 1 - (1 - hdi_prob) / 2)

    fig, ax = plt.subplots(figsize=styles.FIGSIZE_MD)
    x = np.arange(n_year)
    ax.errorbar(
        x,
        mean,
        yerr=[mean - lo, hi - mean],
        fmt="o",
        color=styles.COLOUR_BLUE,
        ecolor=styles.TEXT_COLOUR,
        capsize=3,
    )
    ax.axhline(0, color=styles.TEXT_COLOUR, lw=0.8)
    ax.set_xticks(x)
    ax.set_xlabel("year_idx")
    ax.set_ylabel(r"$\eta_{term}$ year effect (logit)")
    ax.set_title("Termination year effect posterior trajectory")
    fig.tight_layout()
    return fig


def eta_term_year_trajectory_table(
    idata: az.InferenceData,
    *,
    hdi_prob: float = 0.94,
) -> pd.DataFrame:
    """Per-year ``eta_term_year`` posterior summary."""
    post = idata.posterior
    year_arr = np.asarray(post["eta_term_year"].values)
    mean = year_arr.mean(axis=(0, 1))
    lo = _quantile(year_arr, (1 - hdi_prob) / 2)
    hi = _quantile(year_arr, 1 - (1 - hdi_prob) / 2)

    rows = [
        {
            "year_idx": int(i),
            "posterior_mean": float(mean[i]),
            "lo": float(lo[i]),
            "hi": float(hi[i]),
            "hdi_prob": hdi_prob,
        }
        for i in range(year_arr.shape[-1])
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 3. CCHD consistency                                                         #
# --------------------------------------------------------------------------- #


def cchd_consistency_check(
    idata: az.InferenceData,
    cells: pd.DataFrame,
    *,
    published_cchd_prevalence: float = 0.225,
) -> Figure:
    """Posterior CCHD co-occurrence among true DS livebirths vs EUROCAT target.

    ``true DS livebirths`` = ``theta_LB . eta . N_cell``. CCHD prevalence
    among them is a weighted average of ``cells['cchd']`` using those
    counts as weights. Overlay the published prevalence and its tolerance
    band as a reference line.
    """
    import matplotlib.pyplot as plt

    styles = _styles()
    p_ds_lb = np.asarray(idata.posterior["p_ds_lb"].values)  # (c, d, cell)
    N = cells["N_cell"].to_numpy(dtype=float)
    cchd = cells["cchd"].to_numpy(dtype=float)

    true_counts = p_ds_lb * N[None, None, :]  # (c, d, cell)
    numerator = (true_counts * cchd[None, None, :]).sum(axis=-1)
    denominator = true_counts.sum(axis=-1)
    prevalence = numerator / np.clip(denominator, 1e-12, None)

    flat = prevalence.ravel()
    mean = float(flat.mean())
    lo = float(np.quantile(flat, 0.025))
    hi = float(np.quantile(flat, 0.975))

    fig, ax = plt.subplots(figsize=styles.FIGSIZE_MD)
    ax.hist(flat, bins=40, color=styles.COLOUR_BLUE, alpha=0.75)
    ax.axvline(
        published_cchd_prevalence,
        color=styles.COLOUR_RED,
        lw=1.5,
        label=f"EUROCAT ~{published_cchd_prevalence:.0%}",
    )
    ax.axvline(mean, color=styles.TEXT_COLOUR, lw=1.0, ls="--", label="Posterior mean")
    ax.set_xlabel("CCHD prevalence among posterior true DS livebirths")
    ax.set_ylabel("Posterior draws")
    ax.set_title(
        f"CCHD co-occurrence: posterior mean {mean:.1%} "
        f"[{lo:.1%}, {hi:.1%}] vs {published_cchd_prevalence:.1%}"
    )
    ax.legend()
    fig.tight_layout()
    return fig


def cchd_consistency_summary(
    idata: az.InferenceData,
    cells: pd.DataFrame,
    *,
    published_cchd_prevalence: float = 0.225,
) -> pd.DataFrame:
    """Numeric summary row for the CCHD consistency check."""
    p_ds_lb = np.asarray(idata.posterior["p_ds_lb"].values)
    N = cells["N_cell"].to_numpy(dtype=float)
    cchd = cells["cchd"].to_numpy(dtype=float)
    true_counts = p_ds_lb * N[None, None, :]
    numerator = (true_counts * cchd[None, None, :]).sum(axis=-1)
    denominator = true_counts.sum(axis=-1)
    prevalence = numerator / np.clip(denominator, 1e-12, None)
    flat = prevalence.ravel()
    return pd.DataFrame(
        {
            "posterior_mean": [float(flat.mean())],
            "lo_95": [float(np.quantile(flat, 0.025))],
            "hi_95": [float(np.quantile(flat, 0.975))],
            "target": [float(published_cchd_prevalence)],
            "target_in_95_ci": [
                bool(
                    float(np.quantile(flat, 0.025))
                    <= published_cchd_prevalence
                    <= float(np.quantile(flat, 0.975))
                )
            ],
        }
    )


# --------------------------------------------------------------------------- #
# 4. Posterior predictive by stratum                                          #
# --------------------------------------------------------------------------- #


def posterior_predictive_by_stratum(
    idata: az.InferenceData,
    cells: pd.DataFrame,
    *,
    stratum_col: str,
    hdi_prob: float = 0.94,
) -> Figure:
    """Observed vs posterior-predicted recorded counts, aggregated by stratum.

    Uses ``p_recorded`` to compute the posterior-predictive expected
    count per cell, then sums within each stratum value. Compares with
    the observed ``R_cell`` sum.
    """
    import matplotlib.pyplot as plt

    styles = _styles()
    if stratum_col not in cells.columns:
        raise KeyError(f"{stratum_col!r} not in cells frame")
    p_rec = np.asarray(idata.posterior["p_recorded"].values)  # (c, d, cell)
    N = cells["N_cell"].to_numpy(dtype=float)
    R = cells["R_cell"].to_numpy(dtype=float)
    strata = cells[stratum_col].to_numpy()
    unique = np.sort(np.unique(strata))

    pred_counts = p_rec * N[None, None, :]

    mean = np.zeros(len(unique))
    lo = np.zeros(len(unique))
    hi = np.zeros(len(unique))
    observed = np.zeros(len(unique))
    for i, v in enumerate(unique):
        mask = strata == v
        cell_sum = pred_counts[..., mask].sum(axis=-1)
        flat = cell_sum.ravel()
        mean[i] = float(flat.mean())
        lo[i] = float(np.quantile(flat, (1 - hdi_prob) / 2))
        hi[i] = float(np.quantile(flat, 1 - (1 - hdi_prob) / 2))
        observed[i] = float(R[mask].sum())

    fig, ax = plt.subplots(figsize=styles.FIGSIZE_MD)
    x = np.arange(len(unique))
    ax.bar(
        x,
        mean,
        yerr=[mean - lo, hi - mean],
        color=styles.COLOUR_BLUE,
        alpha=0.7,
        capsize=3,
        label="Posterior mean",
    )
    ax.plot(
        x,
        observed,
        "o",
        color=styles.COLOUR_ORANGE,
        label="Observed",
        markersize=5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in unique], rotation=30, ha="right")
    ax.set_xlabel(stratum_col)
    ax.set_ylabel("Recorded DS count")
    ax.set_title(
        f"Posterior predictive check by {stratum_col} "
        f"({int(hdi_prob * 100)}% CI)"
    )
    ax.legend()
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 5. Decomposition by race                                                    #
# --------------------------------------------------------------------------- #


def decomposition_by_race(
    idata: az.InferenceData,
    cells: pd.DataFrame,
) -> Figure:
    """Stacked bar of true DS livebirths by race: recorded / missed.

    Also reports the implied number of prenatally terminated
    pregnancies per race via the identity

        theta_LB * eta_detect * eta_term = theta_LB - (theta_LB * eta)
                                         = theta_LB - p_ds_lb

    so we never need the per-cell eta_detect / eta_term Deterministics,
    which are dropped from the saved InferenceData for size reasons
    (see docstring of ``selection.build_model``). ``theta_lb`` per cell
    is reconstructed from the saved ``theta_lb_age[age_idx]``.
    """
    import matplotlib.pyplot as plt

    styles = _styles()
    if "race_idx" not in cells.columns:
        raise KeyError("'race_idx' not in cells")
    if "age_idx" not in cells.columns:
        raise KeyError("'age_idx' not in cells — cannot reconstruct theta_lb")

    post = idata.posterior
    p_ds_lb = np.asarray(post["p_ds_lb"].values)  # (chain, draw, cell)
    p_rec = np.asarray(post["p_recorded"].values)

    # Reconstruct per-cell theta_lb from theta_lb_age[age_idx].
    theta_lb_age_logit = np.asarray(
        post["theta_lb_age"].values
    )  # (chain, draw, N_AGE)
    age_idx = cells["age_idx"].to_numpy()
    theta_lb_cell = inv_logit(theta_lb_age_logit[..., age_idx])  # (c, d, cell)

    N = cells["N_cell"].to_numpy(dtype=float)
    races = cells["race_idx"].to_numpy()
    unique = np.sort(np.unique(races))
    labels = [
        RACE_LEVELS[int(v)] if 0 <= int(v) < len(RACE_LEVELS) else f"idx_{v}"
        for v in unique
    ]

    # If the fit was theta_only or theta_s (no η), eta=1 so terminated=0.
    # Detect this by whether eta_detect_int is a named RV in the posterior
    # — proxy for "spec='full'".
    has_full_eta = "eta_detect_int" in post.data_vars

    rows = []
    for v in unique:
        mask = races == v
        true_count = (p_ds_lb[..., mask] * N[mask]).sum(axis=-1).mean().item()
        recorded_count = (p_rec[..., mask] * N[mask]).sum(axis=-1).mean().item()
        missed_count = max(true_count - recorded_count, 0.0)
        if has_full_eta:
            terminated_draws = (
                (theta_lb_cell[..., mask] * N[mask]).sum(axis=-1)
                - (p_ds_lb[..., mask] * N[mask]).sum(axis=-1)
            )
            terminated = float(terminated_draws.mean())
        else:
            terminated = float("nan")
        rows.append(
            {
                "race_idx": int(v),
                "race": labels[list(unique).index(v)],
                "true_livebirths": true_count,
                "recorded": recorded_count,
                "missed": missed_count,
                "prenatally_terminated": terminated,
            }
        )
    summary = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=styles.FIGSIZE_LG)
    x = np.arange(len(unique))
    ax.bar(x, summary["recorded"], color=styles.COLOUR_BLUE, label="Recorded")
    ax.bar(
        x,
        summary["missed"],
        bottom=summary["recorded"],
        color=styles.COLOUR_ORANGE,
        label="Missed (posterior)",
    )
    if has_full_eta:
        ax.plot(
            x,
            summary["prenatally_terminated"],
            "v",
            color=styles.COLOUR_RED,
            markersize=8,
            label="Prenatally terminated (implied)",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(summary["race"], rotation=30, ha="right")
    ax.set_ylabel("Posterior mean count")
    ax.set_title("DS livebirth decomposition by race (posterior means)")
    ax.legend()
    fig.tight_layout()
    # Attach the tidy data as an attribute for the rendering CLI to save.
    fig._selection_data = summary  # type: ignore[attr-defined]
    return fig


# --------------------------------------------------------------------------- #
# 6. Age-curve sanity check                                                   #
# --------------------------------------------------------------------------- #


def age_curve_check(
    idata: az.InferenceData,
    cells: pd.DataFrame | None = None,  # noqa: ARG001 — reserved for future use
    *,
    hdi_prob: float = 0.94,
) -> Figure:
    """Posterior ``theta_LB`` by age band vs Morris/de Graaf prior means.

    ``cells`` is accepted for signature symmetry with the other
    diagnostics but not used — ``theta_lb_age`` already carries the
    full posterior over age bands.
    """
    import matplotlib.pyplot as plt

    styles = _styles()
    post = idata.posterior
    if "theta_lb_age" not in post.data_vars:
        raise ValueError("theta_lb_age missing from posterior")
    theta_logit = np.asarray(post["theta_lb_age"].values)  # (c, d, age)
    theta = inv_logit(theta_logit) * 1000.0  # per 1,000 livebirths
    n_age = theta.shape[-1]
    labels = AGE_LEVELS[:n_age]
    mean = theta.mean(axis=(0, 1))
    lo = _quantile(theta, (1 - hdi_prob) / 2)
    hi = _quantile(theta, 1 - (1 - hdi_prob) / 2)
    morris = MORRIS_THETA_LB_PER_1000[:n_age]

    fig, ax = plt.subplots(figsize=styles.FIGSIZE_MD)
    x = np.arange(n_age)
    ax.errorbar(
        x,
        mean,
        yerr=[mean - lo, hi - mean],
        fmt="o",
        color=styles.COLOUR_BLUE,
        capsize=3,
        label="Posterior",
    )
    ax.plot(x, morris, "s", color=styles.COLOUR_ORANGE, label="Morris / de Graaf")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(r"$\theta_{LB}$ per 1,000 livebirths")
    ax.set_title("Age curve: posterior vs Morris prior")
    ax.legend()
    fig.tight_layout()
    return fig


def age_curve_table(
    idata: az.InferenceData,
    *,
    hdi_prob: float = 0.94,
) -> pd.DataFrame:
    """Tidy per-age-band summary of ``theta_LB`` posterior vs Morris."""
    theta_logit = np.asarray(idata.posterior["theta_lb_age"].values)
    theta = inv_logit(theta_logit) * 1000.0
    n_age = theta.shape[-1]
    return pd.DataFrame(
        {
            "age_band": AGE_LEVELS[:n_age],
            "posterior_mean_per_1000": theta.mean(axis=(0, 1)),
            "lo": _quantile(theta, (1 - hdi_prob) / 2),
            "hi": _quantile(theta, 1 - (1 - hdi_prob) / 2),
            "morris_per_1000": MORRIS_THETA_LB_PER_1000[:n_age],
        }
    )


# --------------------------------------------------------------------------- #
# Convergence summary                                                          #
# --------------------------------------------------------------------------- #


def summary_table(
    idata: az.InferenceData,
    *,
    var_names: tuple[str, ...] | None = None,
    hdi_prob: float = 0.94,
) -> pd.DataFrame:
    """Return ``az.summary`` as a DataFrame (optionally filtered)."""
    import arviz as az

    if var_names is None:
        return az.summary(idata, hdi_prob=hdi_prob)
    available = [n for n in var_names if n in idata.posterior.data_vars]
    if not available:
        return az.summary(idata, hdi_prob=hdi_prob)
    return az.summary(idata, var_names=list(available), hdi_prob=hdi_prob)


def convergence_health(
    summary: pd.DataFrame,
    *,
    rhat_threshold: float = 1.01,
    ess_threshold: float = 400.0,
) -> dict:
    """Roll up Rhat / ESS from an ``az.summary`` frame to a pass/fail dict."""
    rhat_col = "r_hat" if "r_hat" in summary.columns else "rhat"
    ess_cols = [c for c in ("ess_bulk", "ess_tail") if c in summary.columns]
    max_rhat = (
        float(summary[rhat_col].max())
        if rhat_col in summary.columns
        else float("nan")
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
