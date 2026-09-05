"""Posterior diagnostics for the three-stage selection model.

Each function takes a fitted ``xr.DataTree`` plus (where relevant)
the aggregated ``cells`` frame that produced it, and returns a
:class:`matplotlib.figure.Figure`. Callers wanting paired CSVs /
publication-quality artefacts should use
:mod:`dspopulations_us_birth_certificates.selection.render`,
which wraps these with a ``_save`` helper that writes figures alongside
their tidy-DataFrame companions.

Functions
---------
- :func:`identifiability_pairplot` — posterior pair-plot of race effects
  on ``eta_term`` vs ``s``. The correlation is now a ridge-correlation
  warning, not a stand-alone identification test, because ``s`` is
  externally anchored.
- :func:`s_anchor_shrinkage_plot` — prior-to-posterior readout for
  ``s_race_year`` showing whether the recording surface was estimated
  from the birth-certificate likelihood or mostly carried in by the
  anchor.
- :func:`eta_term_year_trajectory_plot` — posterior trajectory of
  ``eta_term_year`` by year. Drift across the window is a residual
  year-over-year effect on termination rates.
- :func:`cchd_consistency_check` — posterior CCHD co-occurrence among
  true DS livebirths vs the EUROCAT published prevalence (~22.5%),
  interpreted as a structural stress check rather than calibration.
- :func:`posterior_predictive_by_stratum` — recorded-count PPC plot
  aggregated by a chosen stratum (year / race / age).
- :func:`decomposition_by_race` — posterior stacked estimate of true
  DS livebirths, recorded, prenatally terminated, and missed, by race.
- :func:`age_curve_check` — posterior ``theta_LB`` by age band vs the
  pinned Morris/de Graaf prior means.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from dspopulations_us_birth_certificates.intervals import (
    DEFAULT_ETI_PROB,
    DEFAULT_HPDI_PROB,
    interval_label,
    interval_percent,
    posterior_mean_eti,
)
from dspopulations_us_birth_certificates.selection.priors import (
    AGE_LEVELS,
    MORRIS_THETA_LB_PER_1000,
    RACE_LEVELS,
    inv_logit,
    variant_C_default,
)
from dspopulations_us_birth_certificates.selection.recording_anchor import ANCHOR_YEARS

if TYPE_CHECKING:
    import xarray as xr
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


def _draw_summary(arr: np.ndarray) -> dict[str, float]:
    """Posterior mean + equal-tail interval for a flattened draw array."""
    return posterior_mean_eti(arr, nan=True)


def _year_labels(n_year: int, year_range: tuple[int, int] | None) -> list[int]:
    """Actual years when available, otherwise zero-based year indices."""
    if year_range is not None:
        return list(range(int(year_range[0]), int(year_range[0]) + n_year))
    return list(range(n_year))


def _prior_year_slice(
    prior: np.ndarray,
    *,
    n_year: int,
    year_range: tuple[int, int] | None,
) -> np.ndarray:
    """Slice a full anchor-year prior matrix to the posterior year window."""
    if prior.shape[-1] == n_year:
        return prior
    if year_range is not None:
        start_year = int(year_range[0])
        offset = start_year - ANCHOR_YEARS[0]
        if 0 <= offset and offset + n_year <= prior.shape[-1]:
            return prior[..., offset : offset + n_year]
    return prior[..., :n_year]


def _s_prior_arrays(
    priors_config: Mapping[str, object] | None,
    *,
    n_year: int,
    year_range: tuple[int, int] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``s_race_year`` prior mean/sigma aligned to the posterior."""
    if priors_config is None:
        priors = variant_C_default()
        mean = priors.s_race_year_logit
        sigma = priors.s_race_year_sigma
    else:
        try:
            mean = np.asarray(priors_config["s_race_year_logit"], dtype=float)
            sigma = np.asarray(priors_config["s_race_year_sigma"], dtype=float)
        except KeyError as exc:
            raise ValueError(
                "priors_config must include s_race_year_logit and s_race_year_sigma"
            ) from exc
    return (
        _prior_year_slice(mean, n_year=n_year, year_range=year_range),
        _prior_year_slice(sigma, n_year=n_year, year_range=year_range),
    )


def _s_anchor_interpretation(
    *,
    race_idx: int,
    prior_sd: float,
    sd_ratio: float,
    shift_in_prior_sd: float,
) -> str:
    """Short, conservative label for an ``s_race_year`` shrinkage row."""
    if prior_sd <= 0.005:
        return "effectively fixed by prior"
    if race_idx >= 5:
        return "weak fallback; no de Graaf anchor"
    if abs(shift_in_prior_sd) >= 1.0:
        return "anchor tension"
    if sd_ratio >= 0.8 and abs(shift_in_prior_sd) < 0.5:
        return "mostly anchor-carried"
    if sd_ratio < 0.5:
        return "posterior narrowed"
    return "partly updated"


def _identifiability_correlations(
    idata: xr.DataTree,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Per-race ``eta_term_race``/``s_race`` draws, labels, and correlation array.

    Shared by :func:`identifiability_pairplot` and :func:`identifiability_table`
    so both read the same posterior arrays and compute ``r`` once per race.
    """
    post = idata.posterior
    if "eta_term_race" not in post.data_vars or "s_race" not in post.data_vars:
        raise ValueError(
            "InferenceData must carry 'eta_term_race' and 's_race'. "
            "Re-fit with spec='full' or spec='single_eta'."
        )
    eta = np.asarray(post["eta_term_race"].values)  # (chain, draw, race)
    s = np.asarray(post["s_race"].values)
    n_race = eta.shape[-1]
    labels = [
        RACE_LEVELS[i] if i < len(RACE_LEVELS) else f"idx_{i}" for i in range(n_race)
    ]
    corr = np.array(
        [
            float(np.corrcoef(eta[..., i].ravel(), s[..., i].ravel())[0, 1])
            for i in range(n_race)
        ]
    )
    return eta, s, labels, corr


def _eta_term_year_stats(
    idata: xr.DataTree, *, hdi_prob: float = DEFAULT_ETI_PROB
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-year ``eta_term_year`` posterior mean/lo/hi.

    Shared by :func:`eta_term_year_trajectory_plot` and
    :func:`eta_term_year_trajectory_table`.
    """
    post = idata.posterior
    if "eta_term_year" not in post.data_vars:
        raise ValueError(
            "InferenceData is missing 'eta_term_year' — fit with spec='full'."
        )
    year_arr = np.asarray(post["eta_term_year"].values)  # (chain, draw, year)
    mean = year_arr.mean(axis=(0, 1))
    lo = _quantile(year_arr, (1 - hdi_prob) / 2)
    hi = _quantile(year_arr, 1 - (1 - hdi_prob) / 2)
    return mean, lo, hi


def _cchd_prevalence_draws(idata: xr.DataTree, cells: pd.DataFrame) -> np.ndarray:
    """Flattened posterior draws of CCHD prevalence among true DS livebirths.

    Shared by :func:`cchd_consistency_check` and :func:`cchd_consistency_summary`.
    """
    p_ds_lb = np.asarray(idata.posterior["p_ds_lb"].values)  # (c, d, cell)
    N = cells["N_cell"].to_numpy(dtype=float)
    cchd = cells["cchd"].to_numpy(dtype=float)
    true_counts = p_ds_lb * N[None, None, :]  # (c, d, cell)
    numerator = (true_counts * cchd[None, None, :]).sum(axis=-1)
    denominator = true_counts.sum(axis=-1)
    prevalence = numerator / np.clip(denominator, 1e-12, None)
    return prevalence.ravel()


def _age_curve_stats(
    idata: xr.DataTree, *, hdi_prob: float = DEFAULT_ETI_PROB
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``theta_LB`` per 1,000 livebirths: posterior mean/lo/hi + Morris reference.

    Shared by :func:`age_curve_check` and :func:`age_curve_table`.
    """
    theta_logit = np.asarray(idata.posterior["theta_lb_age"].values)  # (c, d, age)
    theta = inv_logit(theta_logit) * 1000.0  # per 1,000 livebirths
    n_age = theta.shape[-1]
    mean = theta.mean(axis=(0, 1))
    lo = _quantile(theta, (1 - hdi_prob) / 2)
    hi = _quantile(theta, 1 - (1 - hdi_prob) / 2)
    morris = MORRIS_THETA_LB_PER_1000[:n_age]
    return mean, lo, hi, morris


# --------------------------------------------------------------------------- #
# 1. Identifiability pair-plot                                                #
# --------------------------------------------------------------------------- #


def identifiability_pairplot(
    idata: xr.DataTree,
) -> Figure:
    """Per-race pair-plot of ``eta_term_race`` vs ``s_race`` draws.

    A high absolute correlation remains useful evidence of posterior
    trade-off along the eta/s ridge. A low correlation is no longer
    evidence of identification by itself, because the de Graaf anchor can
    keep ``s`` nearly fixed and mechanically suppress the covariance.
    """
    import matplotlib.pyplot as plt

    styles = _styles()
    eta, s, labels, corr = _identifiability_correlations(idata)
    n_race = eta.shape[-1]

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
        r = corr[idx]
        label = "ridge warning" if abs(r) > 0.7 else "low covariance"
        ax.set_title(f"{labels[idx]}\n|r|={abs(r):.2f} ({label})")
        ax.set_xlabel(r"$\eta_{term}$ race effect")
        ax.set_ylabel("s race effect")
        ax.axhline(0, color=styles.TEXT_COLOUR, lw=0.5, alpha=0.5)
        ax.axvline(0, color=styles.TEXT_COLOUR, lw=0.5, alpha=0.5)
    for idx in range(n_race, n_rows * n_cols):
        axes.flat[idx].set_visible(False)

    fig.suptitle(
        r"Posterior ridge correlation: $\eta_{term}$ vs $s$ race effects",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def identifiability_table(idata: xr.DataTree) -> pd.DataFrame:
    """Per-race posterior correlations between ``eta_term_race`` and ``s_race``.

    This is a ridge-correlation diagnostic only. Low ``|r|`` should be
    read with the ``s_anchor_shrinkage`` diagnostic before making any
    identification claim.
    """
    _eta, _s, labels, corr = _identifiability_correlations(idata)
    return pd.DataFrame(
        [
            {
                "race_idx": idx,
                "race": labels[idx],
                "correlation": r,
                "abs_correlation": abs(r),
                "interpretation": (
                    "eta/s ridge warning"
                    if abs(r) > 0.7
                    else "low covariance; inspect s_anchor_shrinkage"
                ),
            }
            for idx, r in enumerate(corr)
        ]
    )


def s_anchor_shrinkage_table(
    idata: xr.DataTree,
    *,
    priors_config: Mapping[str, object] | None = None,
    year_range: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Prior-to-posterior readout for the ``s_race_year`` recording anchor.

    The key columns are ``sd_ratio`` (posterior SD divided by prior SD)
    and ``shift_in_prior_sd`` (posterior mean minus prior mean, measured
    in prior standard deviations). If an anchored row has ``sd_ratio``
    near 1 and a small shift, the posterior is mostly carrying forward
    the anchor rather than estimating a new recording level from the
    birth-certificate likelihood.
    """
    post = idata.posterior
    if "s_race_year" not in post.data_vars:
        raise ValueError("InferenceData is missing 's_race_year'.")
    s = np.asarray(post["s_race_year"].values)  # (chain, draw, race, year)
    n_race = s.shape[-2]
    n_year = s.shape[-1]
    prior_mean, prior_sd = _s_prior_arrays(
        priors_config, n_year=n_year, year_range=year_range
    )
    prior_mean = prior_mean[:n_race, :n_year]
    prior_sd = prior_sd[:n_race, :n_year]
    posterior_mean = s.mean(axis=(0, 1))
    posterior_sd = s.reshape(-1, n_race, n_year).std(axis=0, ddof=1)
    years = _year_labels(n_year, year_range)

    rows: list[dict[str, object]] = []
    for r in range(n_race):
        race = RACE_LEVELS[r] if r < len(RACE_LEVELS) else f"idx_{r}"
        for y in range(n_year):
            psd = float(prior_sd[r, y])
            anchor_source = (
                "fixed_variant_prior"
                if psd <= 0.005
                else "de_graaf"
                if r < 5
                else "weak_fallback"
            )
            sd_ratio = float(posterior_sd[r, y] / psd) if psd > 0 else float("nan")
            shift = (
                float((posterior_mean[r, y] - prior_mean[r, y]) / psd)
                if psd > 0
                else float("nan")
            )
            rows.append(
                {
                    "race_idx": r,
                    "race": race,
                    "year_idx": y,
                    "year": years[y],
                    "anchor_source": anchor_source,
                    "prior_mean_logit": float(prior_mean[r, y]),
                    "prior_sd_logit": psd,
                    "posterior_mean_logit": float(posterior_mean[r, y]),
                    "posterior_sd_logit": float(posterior_sd[r, y]),
                    "sd_ratio": sd_ratio,
                    "shift_in_prior_sd": shift,
                    "interpretation": _s_anchor_interpretation(
                        race_idx=r,
                        prior_sd=psd,
                        sd_ratio=sd_ratio,
                        shift_in_prior_sd=shift,
                    ),
                }
            )
    return pd.DataFrame(rows)


def s_anchor_shrinkage_plot(
    idata: xr.DataTree,
    *,
    priors_config: Mapping[str, object] | None = None,
    year_range: tuple[int, int] | None = None,
) -> Figure:
    """Plot ``s_race_year`` posterior/prior SD ratio and mean shift."""
    import matplotlib.pyplot as plt

    styles = _styles()
    table = s_anchor_shrinkage_table(
        idata, priors_config=priors_config, year_range=year_range
    )
    fig, axes = plt.subplots(1, 2, figsize=styles.FIGSIZE_XL, sharex=True)
    for race, sub in table.groupby("race", sort=False):
        axes[0].plot(sub["year"], sub["sd_ratio"], marker="o", ms=3, label=race)
        axes[1].plot(
            sub["year"],
            sub["shift_in_prior_sd"],
            marker="o",
            ms=3,
            label=race,
        )
    axes[0].axhline(1.0, color=styles.TEXT_COLOUR, lw=0.8, ls="--")
    axes[0].set_ylabel("posterior SD / prior SD")
    axes[0].set_title("Uncertainty retained from s prior")
    axes[1].axhline(0.0, color=styles.TEXT_COLOUR, lw=0.8, ls="--")
    axes[1].axhline(1.0, color=styles.TEXT_COLOUR, lw=0.6, alpha=0.5)
    axes[1].axhline(-1.0, color=styles.TEXT_COLOUR, lw=0.6, alpha=0.5)
    axes[1].set_ylabel("posterior mean shift / prior SD")
    axes[1].set_title("Posterior movement away from s prior")
    xlabel = "year" if year_range is not None else "year_idx"
    for ax in axes:
        ax.set_xlabel(xlabel)
        ax.tick_params(axis="x", rotation=30)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.suptitle("Recording-anchor shrinkage: s(race, year)")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 2. Year-trajectory plot for eta_term                                        #
# --------------------------------------------------------------------------- #


def eta_term_year_trajectory_plot(
    idata: xr.DataTree,
    *,
    hdi_prob: float = DEFAULT_ETI_PROB,
) -> Figure:
    """Posterior trajectory of ``eta_term_year`` by year.

    Year effects on termination are modelled with a single
    homoscedastic sigma; this plot shows the per-year posterior means
    with 89% ETIs so any residual year drift is visible.
    """
    import matplotlib.pyplot as plt

    styles = _styles()
    mean, lo, hi = _eta_term_year_stats(idata, hdi_prob=hdi_prob)
    n_year = mean.shape[-1]

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
    idata: xr.DataTree,
    *,
    hdi_prob: float = DEFAULT_ETI_PROB,
) -> pd.DataFrame:
    """Per-year ``eta_term_year`` posterior summary."""
    mean, lo, hi = _eta_term_year_stats(idata, hdi_prob=hdi_prob)
    rows = [
        {
            "year_idx": int(i),
            "posterior_mean": float(mean[i]),
            "lo": float(lo[i]),
            "hi": float(hi[i]),
            "hdi_prob": hdi_prob,
        }
        for i in range(mean.shape[-1])
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 3. CCHD consistency                                                         #
# --------------------------------------------------------------------------- #


def cchd_consistency_check(
    idata: xr.DataTree,
    cells: pd.DataFrame,
    *,
    published_cchd_prevalence: float = 0.225,
) -> Figure:
    """CCHD structural stress check against an external reference.

    ``true DS livebirths`` = ``theta_LB . eta . N_cell``. CCHD prevalence
    among them is a weighted average of ``cells['cchd']`` using those
    counts as weights. Because CCHD is not a model stage or recording
    covariate, this check should not be interpreted as calibration of
    ``s``; it shows the consequence of the current structural
    independence assumption for clinical co-occurrence.
    """
    import matplotlib.pyplot as plt

    styles = _styles()
    flat = _cchd_prevalence_draws(idata, cells)
    stats = posterior_mean_eti(flat)
    mean = stats["mean"]
    lo = stats["lo"]
    hi = stats["hi"]

    fig, ax = plt.subplots(figsize=styles.FIGSIZE_MD)
    ax.hist(flat, bins=40, color=styles.COLOUR_BLUE, alpha=0.75)
    ax.axvline(
        published_cchd_prevalence,
        color=styles.COLOUR_RED,
        lw=1.5,
        label=f"External reference ~{published_cchd_prevalence:.0%}",
    )
    ax.axvline(mean, color=styles.TEXT_COLOUR, lw=1.0, ls="--", label="Posterior mean")
    ax.set_xlabel("CCHD prevalence implied by posterior true DS weights")
    ax.set_ylabel("Posterior draws")
    ax.set_title(
        f"CCHD structural stress check: posterior mean {mean:.1%}, "
        f"{interval_label()} ETI "
        f"[{lo:.1%}, {hi:.1%}] vs {published_cchd_prevalence:.1%}"
    )
    ax.legend()
    fig.tight_layout()
    return fig


def cchd_consistency_summary(
    idata: xr.DataTree,
    cells: pd.DataFrame,
    *,
    published_cchd_prevalence: float = 0.225,
) -> pd.DataFrame:
    """Numeric summary row for the CCHD structural stress check."""
    flat = _cchd_prevalence_draws(idata, cells)
    stats = posterior_mean_eti(flat)
    interval_pct = interval_percent()
    return pd.DataFrame(
        {
            "posterior_mean": [stats["mean"]],
            f"lo_{interval_pct}": [stats["lo"]],
            f"hi_{interval_pct}": [stats["hi"]],
            "interval_prob": [DEFAULT_ETI_PROB],
            "target": [float(published_cchd_prevalence)],
            "target_in_interval": [
                bool(stats["lo"] <= published_cchd_prevalence <= stats["hi"])
            ],
            "diagnostic_role": ["structural_stress_check"],
            "interpretation": [
                "not s calibration; CCHD is not a selection-model covariate"
            ],
        }
    )


# --------------------------------------------------------------------------- #
# 4. Posterior predictive by stratum                                          #
# --------------------------------------------------------------------------- #


def posterior_predictive_by_stratum(
    idata: xr.DataTree,
    cells: pd.DataFrame,
    *,
    stratum_col: str,
    hdi_prob: float = DEFAULT_ETI_PROB,
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
        f"Posterior predictive check by {stratum_col} ({interval_label(hdi_prob)} ETI)"
    )
    ax.legend()
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 5. Decomposition by race                                                    #
# --------------------------------------------------------------------------- #


def decomposition_by_race(
    idata: xr.DataTree,
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
    theta_lb_age_logit = np.asarray(post["theta_lb_age"].values)  # (chain, draw, N_AGE)
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
        true_draws = (p_ds_lb[..., mask] * N[mask]).sum(axis=-1)
        recorded_draws = (p_rec[..., mask] * N[mask]).sum(axis=-1)
        missed_draws = np.clip(true_draws - recorded_draws, 0.0, None)
        true_summary = _draw_summary(true_draws)
        recorded_summary = _draw_summary(recorded_draws)
        missed_summary = _draw_summary(missed_draws)
        if has_full_eta:
            terminated_draws = (theta_lb_cell[..., mask] * N[mask]).sum(axis=-1) - (
                p_ds_lb[..., mask] * N[mask]
            ).sum(axis=-1)
            terminated_summary = _draw_summary(terminated_draws)
        else:
            terminated_summary = {
                "mean": float("nan"),
                "lo": float("nan"),
                "hi": float("nan"),
            }
        rows.append(
            {
                "race_idx": int(v),
                "race": labels[list(unique).index(v)],
                "true_livebirths": true_summary["mean"],
                "true_livebirths_lo": true_summary["lo"],
                "true_livebirths_hi": true_summary["hi"],
                "recorded": recorded_summary["mean"],
                "recorded_lo": recorded_summary["lo"],
                "recorded_hi": recorded_summary["hi"],
                "missed": missed_summary["mean"],
                "missed_lo": missed_summary["lo"],
                "missed_hi": missed_summary["hi"],
                "prenatally_terminated": terminated_summary["mean"],
                "prenatally_terminated_lo": terminated_summary["lo"],
                "prenatally_terminated_hi": terminated_summary["hi"],
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
    ax.errorbar(
        x,
        summary["true_livebirths"],
        yerr=[
            summary["true_livebirths"] - summary["true_livebirths_lo"],
            summary["true_livebirths_hi"] - summary["true_livebirths"],
        ],
        fmt="none",
        ecolor=styles.TEXT_COLOUR,
        capsize=3,
        label=f"True livebirths {interval_label()} ETI",
    )
    if has_full_eta:
        ax.errorbar(
            x,
            summary["prenatally_terminated"],
            yerr=[
                summary["prenatally_terminated"] - summary["prenatally_terminated_lo"],
                summary["prenatally_terminated_hi"] - summary["prenatally_terminated"],
            ],
            fmt="v",
            color=styles.COLOUR_RED,
            ecolor=styles.COLOUR_RED,
            capsize=3,
            markersize=8,
            label="Prenatally terminated (implied)",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(summary["race"], rotation=30, ha="right")
    ax.set_ylabel("Posterior mean count")
    ax.set_title(
        f"DS livebirth decomposition by race (means with {interval_label()} ETIs)"
    )
    ax.legend()
    fig.tight_layout()
    # Attach the tidy data as an attribute for the rendering CLI to save.
    fig._selection_data = summary  # type: ignore[attr-defined]
    return fig


# --------------------------------------------------------------------------- #
# 6. Age-curve sanity check                                                   #
# --------------------------------------------------------------------------- #


def age_curve_check(
    idata: xr.DataTree,
    cells: pd.DataFrame | None = None,  # noqa: ARG001 — reserved for future use
    *,
    hdi_prob: float = DEFAULT_ETI_PROB,
) -> Figure:
    """Pinned ``theta_LB`` age curve vs Morris/de Graaf prior means.

    ``cells`` is accepted for signature symmetry with the other
    diagnostics but not used — ``theta_lb_age`` already carries the
    full posterior over age bands. Since ``theta_LB`` is pinned tightly,
    this is a coding/anchor propagation check, not evidence that the data
    estimated the age curve.
    """
    import matplotlib.pyplot as plt

    styles = _styles()
    if "theta_lb_age" not in idata.posterior.data_vars:
        raise ValueError("theta_lb_age missing from posterior")
    mean, lo, hi, morris = _age_curve_stats(idata, hdi_prob=hdi_prob)
    n_age = mean.shape[-1]
    labels = AGE_LEVELS[:n_age]

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
    ax.set_title("Pinned age curve: posterior vs Morris prior")
    ax.legend()
    fig.tight_layout()
    return fig


def age_curve_table(
    idata: xr.DataTree,
    *,
    hdi_prob: float = DEFAULT_ETI_PROB,
) -> pd.DataFrame:
    """Tidy per-age-band summary of pinned ``theta_LB`` vs Morris."""
    mean, lo, hi, morris = _age_curve_stats(idata, hdi_prob=hdi_prob)
    n_age = mean.shape[-1]
    relative_diff = (mean - morris) / morris
    return pd.DataFrame(
        {
            "age_band": AGE_LEVELS[:n_age],
            "posterior_mean_per_1000": mean,
            "lo": lo,
            "hi": hi,
            "morris_per_1000": morris,
            "relative_diff": relative_diff,
            "interpretation": "pinned_anchor_propagation",
        }
    )


# --------------------------------------------------------------------------- #
# Convergence summary                                                          #
# --------------------------------------------------------------------------- #


def summary_table(
    idata: xr.DataTree,
    *,
    var_names: tuple[str, ...] | None = None,
    hdi_prob: float = DEFAULT_HPDI_PROB,
) -> pd.DataFrame:
    """Return ``az.summary`` as a DataFrame (optionally filtered)."""
    import arviz as az

    summary_kwargs = {
        "ci_prob": hdi_prob,
        "ci_kind": "hdi",
        # Convergence gates use this frame directly.  ArviZ's display-oriented
        # default can move R-hat or ESS across a strict threshold.  The literal
        # string is required: Python ``None`` falls back to configured rounding.
        "round_to": "none",
    }
    if var_names is None:
        return az.summary(idata, **summary_kwargs)
    available = [n for n in var_names if n in idata.posterior.data_vars]
    if not available:
        return az.summary(idata, **summary_kwargs)
    return az.summary(idata, var_names=list(available), **summary_kwargs)


def convergence_health(
    summary: pd.DataFrame,
    *,
    rhat_threshold: float = 1.01,
    ess_threshold: float = 400.0,
    constant_names: tuple[str, ...] = (),
) -> dict:
    """Roll up Rhat / ESS from an ``az.summary`` frame to a pass/fail dict."""
    summary = summary.loc[
        [str(name).split("[")[0] not in constant_names for name in summary.index]
    ]
    rhat_col = "r_hat" if "r_hat" in summary.columns else "rhat"
    ess_cols = [c for c in ("ess_bulk", "ess_tail") if c in summary.columns]
    columns = [rhat_col, "ess_bulk", "ess_tail"]
    finite = bool(len(summary) and all(c in summary for c in columns))
    finite = finite and bool(np.isfinite(summary[columns].to_numpy(dtype=float)).all())
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
        "finite_diagnostics": finite,
        "all_ok": finite and rhat_ok and ess_ok,
    }
