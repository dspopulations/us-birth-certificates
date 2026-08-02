"""Reporting outputs for the core reduction-recording model."""

from __future__ import annotations

from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from dspopulations_us_birth_certificates.selection.priors import AGE_LEVELS, logit

DEFAULT_HDI_PROB = 0.95


def _inv_logit(x: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def _summary(
    draws: np.ndarray, *, hdi_prob: float = DEFAULT_HDI_PROB
) -> dict[str, float]:
    """Mean and equal-tail interval for a draw array."""
    flat = np.asarray(draws, dtype=float).reshape(-1)
    alpha = (1.0 - hdi_prob) / 2.0
    return {
        "mean": float(np.mean(flat)),
        "lo": float(np.quantile(flat, alpha)),
        "hi": float(np.quantile(flat, 1.0 - alpha)),
    }


def _normal_interval_z(prob: float) -> float:
    alpha = (1.0 - prob) / 2.0
    return NormalDist().inv_cdf(1.0 - alpha)


def _year_labels(cells: pd.DataFrame, year_range: tuple[int, int] | None) -> list[int]:
    n_year = int(cells["year_idx"].max()) + 1 if len(cells) else 0
    if year_range is not None:
        return list(range(int(year_range[0]), int(year_range[0]) + n_year))
    return list(range(n_year))


def _require_posterior_predictive(idata: Any):
    try:
        return idata.posterior_predictive["R_obs"]
    except (AttributeError, KeyError) as exc:
        raise ValueError(
            "Core reporting requires posterior predictive draws for R_obs."
        ) from exc


def _table_path(out_dir: Path, stem: str) -> Path:
    return out_dir / "tables" / f"{stem}.csv"


def _plot_path(out_dir: Path, stem: str, suffix: str) -> Path:
    return out_dir / "plots" / f"{stem}.{suffix}"


def _save_figure(fig, out_dir: Path, stem: str) -> None:
    import matplotlib.pyplot as plt

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(_plot_path(out_dir, stem, "png"), dpi=150, bbox_inches="tight")
    fig.savefig(_plot_path(out_dir, stem, "svg"), bbox_inches="tight")
    plt.close(fig)


def _write_table(df: pd.DataFrame, out_dir: Path, stem: str) -> None:
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(_table_path(out_dir, stem), index=False)


def _natural_expected_by_year(
    cells: pd.DataFrame,
    theta_lb_age: np.ndarray,
    *,
    n_year: int,
) -> np.ndarray:
    out = np.zeros(n_year, dtype=float)
    age_idx = cells["age_idx"].to_numpy()
    year_idx = cells["year_idx"].to_numpy()
    n_cell = cells["N_cell"].to_numpy(dtype=float)
    for y in range(n_year):
        mask = year_idx == y
        out[y] = float((n_cell[mask] * theta_lb_age[age_idx[mask]]).sum())
    return out


def _prior_rho_table(
    priors_config: dict[str, object],
    *,
    hdi_prob: float,
) -> pd.DataFrame:
    mu = np.asarray(priors_config["reduction_logit"], dtype=float)
    sigma = np.asarray(priors_config["reduction_sigma"], dtype=float)
    mean = np.asarray(priors_config["reduction_mean"], dtype=float)
    z = _normal_interval_z(hdi_prob)
    return pd.DataFrame(
        {
            "rho_prior_mean": mean,
            "rho_prior_lo": _inv_logit(mu - z * sigma),
            "rho_prior_hi": _inv_logit(mu + z * sigma),
            "rho_prior_sigma_logit": sigma,
            "rho_prior_source": priors_config.get("reduction_source", ""),
            "extrapolated": np.arange(len(mean))
            >= int(priors_config.get("extrapolated_reduction_start", 10**9))
            - int(priors_config.get("year_start", 0)),
        }
    )


def accounting_by_year_table(
    idata: Any,
    cells: pd.DataFrame,
    priors_config: dict[str, object],
    *,
    year_range: tuple[int, int] | None,
    hdi_prob: float = DEFAULT_HDI_PROB,
) -> pd.DataFrame:
    """Build the main by-year accounting table."""
    years = _year_labels(cells, year_range)
    n_year = len(years)
    theta = np.asarray(priors_config["theta_lb_age"], dtype=float)
    natural = _natural_expected_by_year(cells, theta, n_year=n_year)
    observed = cells.groupby("year_idx", observed=True)["R_cell"].sum()
    births = cells.groupby("year_idx", observed=True)["N_cell"].sum()
    prior = _prior_rho_table(
        {**priors_config, "year_start": years[0] if years else 0},
        hdi_prob=hdi_prob,
    )

    rows = []
    for y, year in enumerate(years):
        row: dict[str, float | int | str | bool] = {
            "year": year,
            "births": int(births.get(y, 0)),
            "recorded_ds": int(observed.get(y, 0)),
            "natural_expected_ds": natural[y],
        }
        for var in (
            "rho_year",
            "eta_year",
            "true_count_year",
            "recorded_count_year_mu",
        ):
            stats = _summary(idata.posterior[var].sel(year=y).values, hdi_prob=hdi_prob)
            row[f"{var}_mean"] = stats["mean"]
            row[f"{var}_lo"] = stats["lo"]
            row[f"{var}_hi"] = stats["hi"]
        for col in prior.columns:
            row[col] = prior.iloc[y][col]
        rows.append(row)
    return pd.DataFrame(rows)


def headline_table(
    idata: Any,
    cells: pd.DataFrame,
    accounting: pd.DataFrame,
    priors_config: dict[str, object],
    *,
    hdi_prob: float = DEFAULT_HDI_PROB,
) -> pd.DataFrame:
    """One-row-per-headline summary for the report top section."""
    total_true = _summary(idata.posterior["true_count_total"].values, hdi_prob=hdi_prob)
    recording_s = _summary(idata.posterior["recording_s"].values, hdi_prob=hdi_prob)
    total_births = int(cells["N_cell"].sum())
    total_recorded = int(cells["R_cell"].sum())
    natural_total = float(accounting["natural_expected_ds"].sum())
    rho_agg = 1.0 - total_true["mean"] / natural_total
    recording_prior_mean = _inv_logit(
        float(priors_config.get("recording_s_logit", logit(0.5)))
    ).item()
    return pd.DataFrame(
        [
            {
                "metric": "livebirths",
                "mean": total_births,
                "lo": np.nan,
                "hi": np.nan,
                "notes": "Total livebirths in modelled cells",
            },
            {
                "metric": "recorded_ds",
                "mean": total_recorded,
                "lo": np.nan,
                "hi": np.nan,
                "notes": "Births with DS recorded on certificate",
            },
            {
                "metric": "natural_expected_ds",
                "mean": natural_total,
                "lo": np.nan,
                "hi": np.nan,
                "notes": "Morris/de Graaf age-expected DS livebirths before reduction",
            },
            {
                "metric": "true_ds_livebirths",
                **total_true,
                "notes": "Posterior true DS livebirth total",
            },
            {
                "metric": "aggregate_reduction",
                "mean": rho_agg,
                "lo": np.nan,
                "hi": np.nan,
                "notes": "1 - true / natural, using posterior mean true total",
            },
            {
                "metric": "recording_s",
                **recording_s,
                "notes": (
                    "Overall certificate recording sensitivity; prior mean "
                    f"{recording_prior_mean:.3f}"
                ),
            },
        ]
    )


def reduction_prior_posterior_table(
    accounting: pd.DataFrame,
) -> pd.DataFrame:
    """Table comparing rho prior and posterior by year."""
    cols = [
        "year",
        "rho_prior_mean",
        "rho_prior_lo",
        "rho_prior_hi",
        "rho_prior_sigma_logit",
        "rho_year_mean",
        "rho_year_lo",
        "rho_year_hi",
        "eta_year_mean",
        "extrapolated",
    ]
    return accounting[cols].copy()


def recording_s_table(
    idata: Any,
    priors_config: dict[str, object],
    *,
    hdi_prob: float = DEFAULT_HDI_PROB,
) -> pd.DataFrame:
    """Prior/posterior summary for the single recording sensitivity parameter."""
    post = _summary(idata.posterior["recording_s"].values, hdi_prob=hdi_prob)
    mu = float(priors_config.get("recording_s_logit", logit(0.5)))
    sigma = float(priors_config.get("recording_s_sigma", 1.0))
    z = _normal_interval_z(hdi_prob)
    return pd.DataFrame(
        [
            {
                "parameter": "recording_s",
                "prior_mean": float(_inv_logit(mu)),
                "prior_lo": float(_inv_logit(mu - z * sigma)),
                "prior_hi": float(_inv_logit(mu + z * sigma)),
                "prior_sigma_logit": sigma,
                "posterior_mean": post["mean"],
                "posterior_lo": post["lo"],
                "posterior_hi": post["hi"],
            }
        ]
    )


def posterior_predictive_table(
    idata: Any,
    cells: pd.DataFrame,
    *,
    group_col: str,
    labels: list[str | int],
    hdi_prob: float = DEFAULT_HDI_PROB,
) -> pd.DataFrame:
    """Aggregate posterior predictive recorded DS counts by one cell column."""
    ppc = _require_posterior_predictive(idata)
    groups = cells[group_col].to_numpy()
    observed = cells.groupby(group_col, observed=True)["R_cell"].sum()
    rows = []
    for idx, label in enumerate(labels):
        mask = groups == idx
        if not np.any(mask):
            continue
        draws = ppc[:, :, mask].sum(dim="cell").values
        stats = _summary(draws, hdi_prob=hdi_prob)
        rows.append(
            {
                group_col: idx,
                "label": label,
                "observed": int(observed.get(idx, 0)),
                "predicted_mean": stats["mean"],
                "predicted_lo": stats["lo"],
                "predicted_hi": stats["hi"],
                "observed_in_interval": bool(
                    stats["lo"] <= observed.get(idx, 0) <= stats["hi"]
                ),
            }
        )
    return pd.DataFrame(rows)


def _errorbar_plot(
    df: pd.DataFrame,
    *,
    x: str,
    observed: str,
    mean: str,
    lo: str,
    hi: str,
    ylabel: str,
    title: str,
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    xpos = np.arange(len(df))
    y = df[mean].to_numpy(dtype=float)
    yerr = np.vstack(
        (y - df[lo].to_numpy(dtype=float), df[hi].to_numpy(dtype=float) - y)
    )
    ax.errorbar(xpos, y, yerr=yerr, fmt="o-", capsize=4, label="posterior predictive")
    ax.scatter(
        xpos,
        df[observed].to_numpy(dtype=float),
        marker="x",
        s=70,
        color="black",
        label="observed",
        zorder=3,
    )
    ax.set_xticks(xpos)
    ax.set_xticklabels(df[x].astype(str), rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def _accounting_plot(df: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(df))
    ax.plot(
        x, df["natural_expected_ds"], marker="o", label="age-expected before reduction"
    )
    true_mean = df["true_count_year_mean"].to_numpy(dtype=float)
    true_lo = df["true_count_year_lo"].to_numpy(dtype=float)
    true_hi = df["true_count_year_hi"].to_numpy(dtype=float)
    ax.fill_between(x, true_lo, true_hi, alpha=0.18, label="true DS 95% interval")
    ax.plot(x, true_mean, marker="o", label="true DS posterior mean")
    ax.bar(x, df["recorded_ds"], width=0.45, alpha=0.45, label="recorded DS")
    ax.set_xticks(x)
    ax.set_xticklabels(df["year"].astype(str), rotation=35, ha="right")
    ax.set_ylabel("DS births")
    ax.set_title("Core accounting by year")
    ax.legend()
    fig.tight_layout()
    return fig


def _reduction_plot(df: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(df))
    ax.fill_between(
        x,
        df["rho_prior_lo"],
        df["rho_prior_hi"],
        alpha=0.16,
        label="prior 95% interval",
    )
    ax.plot(x, df["rho_prior_mean"], "--", label="prior mean")
    post = df["rho_year_mean"].to_numpy(dtype=float)
    yerr = np.vstack(
        (
            post - df["rho_year_lo"].to_numpy(dtype=float),
            df["rho_year_hi"].to_numpy(dtype=float) - post,
        )
    )
    ax.errorbar(x, post, yerr=yerr, fmt="o-", capsize=4, label="posterior")
    ax.set_xticks(x)
    ax.set_xticklabels(df["year"].astype(str), rotation=35, ha="right")
    ax.set_ylim(0, max(0.65, float(df["rho_year_hi"].max()) + 0.05))
    ax.set_ylabel("combined reduction before livebirth")
    ax.set_title("Reduction prior vs posterior")
    ax.legend()
    fig.tight_layout()
    return fig


def _recording_s_plot(
    idata: Any, table: pd.DataFrame, priors_config: dict[str, object]
):
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(47)
    mu = float(priors_config.get("recording_s_logit", logit(0.5)))
    sigma = float(priors_config.get("recording_s_sigma", 1.0))
    prior_draws = _inv_logit(rng.normal(mu, sigma, size=10_000))
    posterior_draws = idata.posterior["recording_s"].values.reshape(-1)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bins = np.linspace(0, 1, 50)
    ax.hist(prior_draws, bins=bins, density=True, alpha=0.25, label="prior")
    ax.hist(posterior_draws, bins=bins, density=True, alpha=0.55, label="posterior")
    ax.axvline(
        table["posterior_mean"].iloc[0],
        color="black",
        linestyle="-",
        label="posterior mean",
    )
    ax.set_xlabel("certificate recording sensitivity")
    ax.set_ylabel("density")
    ax.set_title("Overall recording sensitivity")
    ax.legend()
    fig.tight_layout()
    return fig


def render_core_all(
    idata: Any,
    cells: pd.DataFrame,
    out_dir: Path,
    *,
    priors_config: dict[str, object],
    year_range: tuple[int, int] | None = None,
    hdi_prob: float = DEFAULT_HDI_PROB,
) -> dict[str, pd.DataFrame]:
    """Write all prespecified core-model report figures and tables."""
    out_dir = Path(out_dir)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    accounting = accounting_by_year_table(
        idata,
        cells,
        priors_config,
        year_range=year_range,
        hdi_prob=hdi_prob,
    )
    headlines = headline_table(
        idata,
        cells,
        accounting,
        priors_config,
        hdi_prob=hdi_prob,
    )
    reduction = reduction_prior_posterior_table(accounting)
    recording = recording_s_table(idata, priors_config, hdi_prob=hdi_prob)
    ppc_year = posterior_predictive_table(
        idata,
        cells,
        group_col="year_idx",
        labels=list(accounting["year"]),
        hdi_prob=hdi_prob,
    )
    ppc_age = posterior_predictive_table(
        idata,
        cells,
        group_col="age_idx",
        labels=AGE_LEVELS,
        hdi_prob=hdi_prob,
    )

    tables = {
        "core_headlines": headlines,
        "core_accounting_by_year": accounting,
        "core_reduction_prior_posterior": reduction,
        "core_recording_s": recording,
        "core_ppc_by_year": ppc_year,
        "core_ppc_by_age": ppc_age,
    }
    for stem, df in tables.items():
        _write_table(df, out_dir, stem)

    _save_figure(_accounting_plot(accounting), out_dir, "core_accounting_by_year")
    _save_figure(_reduction_plot(reduction), out_dir, "core_reduction_prior_posterior")
    _save_figure(
        _recording_s_plot(idata, recording, priors_config), out_dir, "core_recording_s"
    )
    _save_figure(
        _errorbar_plot(
            ppc_year,
            x="label",
            observed="observed",
            mean="predicted_mean",
            lo="predicted_lo",
            hi="predicted_hi",
            ylabel="recorded DS births",
            title="Posterior predictive check by year",
        ),
        out_dir,
        "core_ppc_by_year",
    )
    _save_figure(
        _errorbar_plot(
            ppc_age,
            x="label",
            observed="observed",
            mean="predicted_mean",
            lo="predicted_lo",
            hi="predicted_hi",
            ylabel="recorded DS births",
            title="Posterior predictive check by maternal age band",
        ),
        out_dir,
        "core_ppc_by_age",
    )
    return tables


__all__ = [
    "accounting_by_year_table",
    "headline_table",
    "posterior_predictive_table",
    "recording_s_table",
    "reduction_prior_posterior_table",
    "render_core_all",
]
