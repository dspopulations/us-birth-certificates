"""Reporting outputs for the core reduction-recording model."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dspopulations_us_birth_certificates.chance import (
    get_ds_lb_nt_probability_array,
)
from dspopulations_us_birth_certificates.intervals import (
    DEFAULT_ETI_PROB,
    interval_label,
    normal_interval_z,
    posterior_mean_eti,
)
from dspopulations_us_birth_certificates.selection.priors import AGE_LEVELS, logit

DEFAULT_INTERVAL_PROB = DEFAULT_ETI_PROB


def _inv_logit(x: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def _summary(
    draws: np.ndarray, *, interval_prob: float = DEFAULT_INTERVAL_PROB
) -> dict[str, float]:
    """Mean and equal-tail interval for a draw array."""
    return posterior_mean_eti(draws, prob=interval_prob)


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


def _interval_yerr(mean: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Matplotlib-safe asymmetric error bars from interval bounds."""
    return np.vstack((np.maximum(mean - lo, 0.0), np.maximum(hi - mean, 0.0)))


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


def _natural_expected_by_year_age(
    cells: pd.DataFrame,
    theta_lb_age: np.ndarray,
    *,
    n_year: int,
    n_age: int,
) -> np.ndarray:
    """Return natural-DS expected counts for every modelled year and age."""
    out = np.zeros((n_year, n_age), dtype=float)
    age_idx = cells["age_idx"].to_numpy(dtype=int)
    year_idx = cells["year_idx"].to_numpy(dtype=int)
    natural_cell = cells["N_cell"].to_numpy(dtype=float) * theta_lb_age[age_idx]
    np.add.at(out, (year_idx, age_idx), natural_cell)
    return out


def _theta_lb_age_for_report(
    idata: Any,
    cells: pd.DataFrame,
    priors_config: dict[str, object],
    *,
    n_age: int,
) -> np.ndarray:
    """Resolve the Morris age curve actually used by the fitted model."""
    constant_data = getattr(idata, "constant_data", None)
    if constant_data is not None and "theta_lb_age" in constant_data:
        theta = np.asarray(constant_data["theta_lb_age"].values, dtype=float)
    elif "maternal_age" in cells:
        age_table = (
            cells[["age_idx", "maternal_age"]].drop_duplicates().sort_values("age_idx")
        )
        theta = np.asarray(
            get_ds_lb_nt_probability_array(
                age_table["maternal_age"].to_numpy(dtype=int)
            ),
            dtype=float,
        )
    else:
        theta = np.asarray(priors_config["theta_lb_age"], dtype=float)
    if len(theta) != n_age:
        raise ValueError(
            f"Resolved theta_lb_age has length {len(theta)}, but the cells use "
            f"{n_age} ages."
        )
    return theta


def _age_reduction_variable(idata: Any) -> str | None:
    """Return the supported age-specific reduction variable, when present."""
    for name in ("rho_year_age", "rho_age_year"):
        if name not in idata.posterior:
            continue
        dims = set(idata.posterior[name].dims)
        if not {"year", "age"}.issubset(dims):
            raise ValueError(
                f"Posterior variable {name!r} must have 'year' and 'age' dimensions."
            )
        return name
    return None


def _age_labels(cells: pd.DataFrame, rho_year_age: Any | None = None) -> list[object]:
    """Resolve report labels for age bands or exact maternal ages."""
    if "maternal_age" in cells:
        n_age = int(cells["age_idx"].max()) + 1 if len(cells) else 0
        label_column = (
            "maternal_age_label" if "maternal_age_label" in cells else "maternal_age"
        )
        label_columns = ["age_idx", "maternal_age"]
        if label_column != "maternal_age":
            label_columns.append(label_column)
        labels = cells[label_columns].drop_duplicates().sort_values("age_idx")
        counts = labels.groupby("age_idx", observed=True)[label_column].nunique()
        if len(labels) != n_age or (counts != 1).any():
            raise ValueError(
                f"{label_column} must map one-to-one to the age_idx model dimension."
            )
        return labels[label_column].tolist()

    if rho_year_age is None:
        return list(AGE_LEVELS)
    n_age = int(rho_year_age.sizes["age"])
    coord = np.asarray(rho_year_age.coords["age"].values)
    if not np.array_equal(coord, np.arange(n_age)):
        return [value.item() if hasattr(value, "item") else value for value in coord]
    return [AGE_LEVELS[idx] if idx < len(AGE_LEVELS) else idx for idx in range(n_age)]


def _maternal_age_band_index(maternal_age: np.ndarray) -> np.ndarray:
    """Map exact maternal-age codes to the established seven reporting bands."""
    age = np.asarray(maternal_age, dtype=int)
    return np.select(
        [age < 20, age < 25, age < 30, age < 35, age < 40, age < 45],
        [0, 1, 2, 3, 4, 5],
        default=6,
    ).astype(int)


def _prior_rho_table(
    priors_config: dict[str, object],
    *,
    interval_prob: float,
) -> pd.DataFrame:
    mu = np.asarray(priors_config["reduction_logit"], dtype=float)
    sigma = np.asarray(priors_config["reduction_sigma"], dtype=float)
    mean = np.asarray(priors_config["reduction_mean"], dtype=float)
    z = normal_interval_z(interval_prob)
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
    interval_prob: float = DEFAULT_INTERVAL_PROB,
) -> pd.DataFrame:
    """Build the main by-year accounting table."""
    years = _year_labels(cells, year_range)
    n_year = len(years)
    if "maternal_age" in cells:
        n_age = int(cells["age_idx"].max()) + 1 if len(cells) else 0
    else:
        n_age = len(priors_config["theta_lb_age"])
    theta = _theta_lb_age_for_report(
        idata,
        cells,
        priors_config,
        n_age=n_age,
    )
    natural = _natural_expected_by_year(cells, theta, n_year=n_year)
    observed = cells.groupby("year_idx", observed=True)["R_cell"].sum()
    births = cells.groupby("year_idx", observed=True)["N_cell"].sum()
    prior = _prior_rho_table(
        {**priors_config, "year_start": years[0] if years else 0},
        interval_prob=interval_prob,
    )

    rows = []
    for y, year in enumerate(years):
        row: dict[str, float | int | str | bool] = {
            "year": year,
            "births": int(births.get(y, 0)),
            "recorded_ds": int(observed.get(y, 0)),
            "natural_expected_ds": natural[y],
            "interval_prob": interval_prob,
        }
        for var in (
            "rho_year",
            "eta_year",
            "recording_s_year",
            "true_count_year",
            "recorded_count_year_mu",
        ):
            stats = _summary(
                idata.posterior[var].sel(year=y).values,
                interval_prob=interval_prob,
            )
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
    interval_prob: float = DEFAULT_INTERVAL_PROB,
) -> pd.DataFrame:
    """One-row-per-headline summary for the report top section."""
    total_true = _summary(
        idata.posterior["true_count_total"].values,
        interval_prob=interval_prob,
    )
    recording_s = _summary(
        idata.posterior["recording_s"].values,
        interval_prob=interval_prob,
    )
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
                "notes": (
                    f"Posterior true DS livebirth total ({interval_label(interval_prob)}"
                    " ETI)"
                ),
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
                    "Global certificate recording-sensitivity centre; "
                    "equals overall sensitivity in constant-s models; prior mean "
                    f"{recording_prior_mean:.3f}; "
                    f"{interval_label(interval_prob)} ETI"
                ),
            },
        ]
    )


def reduction_prior_posterior_table(
    accounting: pd.DataFrame,
) -> pd.DataFrame:
    """Table comparing the marginal rho prior and posterior by year."""
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
    out = accounting[cols].copy()
    out["rho_year_definition"] = (
        "natural-DS-weighted marginal reduction across maternal-age cells"
    )
    return out


def age_reduction_by_year_table(
    idata: Any,
    cells: pd.DataFrame,
    priors_config: dict[str, object],
    *,
    year_range: tuple[int, int] | None,
    interval_prob: float = DEFAULT_INTERVAL_PROB,
) -> pd.DataFrame:
    """Summarise DSP003 age-specific rho and its weighted yearly marginal.

    ``rho_year`` is the model's calibrated marginal. ``rho_year_recomputed``
    independently recomputes the same quantity from ``rho_year_age`` using
    natural expected DS births, ``N * theta_age``, as the weights.
    """
    variable = _age_reduction_variable(idata)
    if variable is None:
        raise ValueError(
            "InferenceData has no age-specific reduction variable; expected "
            "'rho_year_age' or 'rho_age_year'."
        )
    if "rho_year" not in idata.posterior:
        raise ValueError("InferenceData is missing the marginal 'rho_year'.")

    rho_year_age = idata.posterior[variable]
    years = _year_labels(cells, year_range)
    n_year = len(years)
    n_age = int(rho_year_age.sizes["age"])
    if int(rho_year_age.sizes["year"]) != n_year:
        raise ValueError(
            "Age-specific reduction year dimension does not match the modelled cells."
        )

    theta = _theta_lb_age_for_report(
        idata,
        cells,
        priors_config,
        n_age=n_age,
    )
    natural = _natural_expected_by_year_age(
        cells,
        theta,
        n_year=n_year,
        n_age=n_age,
    )
    natural_total = natural.sum(axis=1)
    if np.any(natural_total <= 0.0):
        bad = [years[idx] for idx in np.flatnonzero(natural_total <= 0.0)]
        raise ValueError(f"Natural expected DS total is zero for years: {bad!r}.")
    weight_share = natural / natural_total[:, None]
    age_labels = _age_labels(cells, rho_year_age)

    rows: list[dict[str, float | int | str]] = []
    for y, year in enumerate(years):
        marginal_draws = np.asarray(
            idata.posterior["rho_year"].isel(year=y).values,
            dtype=float,
        ).reshape(-1)
        age_draws = np.stack(
            [
                np.asarray(
                    rho_year_age.isel(year=y, age=a).values,
                    dtype=float,
                ).reshape(-1)
                for a in range(n_age)
            ],
            axis=1,
        )
        recomputed_draws = age_draws @ weight_share[y]
        if recomputed_draws.shape != marginal_draws.shape:
            raise ValueError(
                "rho_year and the age-specific reduction variable have "
                "incompatible sample dimensions."
            )
        marginal_stats = _summary(
            marginal_draws,
            interval_prob=interval_prob,
        )
        recomputed_stats = _summary(
            recomputed_draws,
            interval_prob=interval_prob,
        )
        difference_draws = recomputed_draws - marginal_draws
        difference_stats = _summary(
            difference_draws,
            interval_prob=interval_prob,
        )
        anchor_stats = None
        anchor_difference_stats = None
        if "rho_year_anchor" in idata.posterior:
            anchor_draws = np.asarray(
                idata.posterior["rho_year_anchor"].isel(year=y).values,
                dtype=float,
            ).reshape(-1)
            if anchor_draws.shape != marginal_draws.shape:
                raise ValueError(
                    "rho_year_anchor and rho_year have incompatible sample dimensions."
                )
            anchor_stats = _summary(anchor_draws, interval_prob=interval_prob)
            anchor_difference_stats = _summary(
                anchor_draws - marginal_draws,
                interval_prob=interval_prob,
            )

        for age in range(n_age):
            age_stats = _summary(
                age_draws[:, age],
                interval_prob=interval_prob,
            )
            row: dict[str, float | int | str] = {
                "year": year,
                "age_idx": age,
                "age": age_labels[age],
                "natural_expected_ds": natural[y, age],
                "natural_ds_weight_share": weight_share[y, age],
                "rho_year_age_mean": age_stats["mean"],
                "rho_year_age_lo": age_stats["lo"],
                "rho_year_age_hi": age_stats["hi"],
                "rho_year_marginal_mean": marginal_stats["mean"],
                "rho_year_marginal_lo": marginal_stats["lo"],
                "rho_year_marginal_hi": marginal_stats["hi"],
                "rho_year_recomputed_mean": recomputed_stats["mean"],
                "rho_year_recomputed_lo": recomputed_stats["lo"],
                "rho_year_recomputed_hi": recomputed_stats["hi"],
                "rho_year_recomputed_minus_marginal_mean": difference_stats["mean"],
                "rho_year_recomputed_minus_marginal_lo": difference_stats["lo"],
                "rho_year_recomputed_minus_marginal_hi": difference_stats["hi"],
                "rho_year_marginal_max_abs_draw_difference": float(
                    np.max(np.abs(difference_draws))
                ),
                "rho_year_definition": (
                    "natural-DS-weighted marginal reduction across maternal-age cells"
                ),
                "interval_prob": interval_prob,
            }
            if "maternal_age" in cells:
                maternal_age = int(
                    cells.loc[cells["age_idx"] == age, "maternal_age"].iloc[0]
                )
                boundary = maternal_age <= 14 or maternal_age >= 48
                row.update(
                    {
                        "maternal_age": maternal_age,
                        "sparse_boundary_age": boundary,
                        "maternal_age_endpoint_capped": maternal_age in {12, 50},
                    }
                )
            if anchor_stats is not None and anchor_difference_stats is not None:
                row.update(
                    {
                        "rho_year_anchor_mean": anchor_stats["mean"],
                        "rho_year_anchor_lo": anchor_stats["lo"],
                        "rho_year_anchor_hi": anchor_stats["hi"],
                        "rho_year_anchor_minus_marginal_mean": (
                            anchor_difference_stats["mean"]
                        ),
                        "rho_year_anchor_minus_marginal_lo": (
                            anchor_difference_stats["lo"]
                        ),
                        "rho_year_anchor_minus_marginal_hi": (
                            anchor_difference_stats["hi"]
                        ),
                    }
                )
            if "rho_age_offset" in idata.posterior:
                offset = idata.posterior["rho_age_offset"]
                if "age" not in offset.dims:
                    raise ValueError(
                        "Posterior variable 'rho_age_offset' must have an 'age' "
                        "dimension."
                    )
                offset_stats = _summary(
                    offset.isel(age=age).values,
                    interval_prob=interval_prob,
                )
                row.update(
                    {
                        "rho_age_offset_mean": offset_stats["mean"],
                        "rho_age_offset_lo": offset_stats["lo"],
                        "rho_age_offset_hi": offset_stats["hi"],
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def recording_s_table(
    idata: Any,
    priors_config: dict[str, object],
    *,
    interval_prob: float = DEFAULT_INTERVAL_PROB,
) -> pd.DataFrame:
    """Prior/posterior summary for the global recording sensitivity parameter."""
    post = _summary(
        idata.posterior["recording_s"].values,
        interval_prob=interval_prob,
    )
    mu = float(priors_config.get("recording_s_logit", logit(0.5)))
    sigma = float(priors_config.get("recording_s_sigma", 1.0))
    z = normal_interval_z(interval_prob)
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
                "interval_prob": interval_prob,
            }
        ]
    )


def recording_s_by_year_table(
    idata: Any,
    priors_config: dict[str, object],
    *,
    years: list[int],
    recording_model: str = "constant",
    interval_prob: float = DEFAULT_INTERVAL_PROB,
) -> pd.DataFrame:
    """Prior/posterior summary for recording sensitivity by year."""
    if "recording_s_year" not in idata.posterior:
        raise ValueError(
            "InferenceData is missing 'recording_s_year'. Re-fit the core model."
        )

    mu = float(priors_config.get("recording_s_logit", logit(0.5)))
    global_sigma = float(priors_config.get("recording_s_sigma", 1.0))
    year_sigma = (
        float(priors_config.get("recording_s_year_sigma", 0.0))
        if recording_model == "year"
        else 0.0
    )
    centered_offset_sigma = (
        year_sigma * math.sqrt((len(years) - 1.0) / len(years)) if years else 0.0
    )
    marginal_sigma = math.sqrt(global_sigma**2 + centered_offset_sigma**2)
    z = normal_interval_z(interval_prob)

    rows = []
    for idx, year in enumerate(years):
        post = _summary(
            idata.posterior["recording_s_year"].sel(year=idx).values,
            interval_prob=interval_prob,
        )
        rows.append(
            {
                "year": year,
                "parameter": "recording_s_year",
                "recording_model": recording_model,
                "prior_mean": float(_inv_logit(mu)),
                "prior_lo": float(_inv_logit(mu - z * marginal_sigma)),
                "prior_hi": float(_inv_logit(mu + z * marginal_sigma)),
                "prior_sigma_logit": marginal_sigma,
                "prior_global_sigma_logit": global_sigma,
                "prior_year_offset_sigma_logit": year_sigma,
                "posterior_mean": post["mean"],
                "posterior_lo": post["lo"],
                "posterior_hi": post["hi"],
                "interval_prob": interval_prob,
            }
        )
    return pd.DataFrame(rows)


def posterior_predictive_table(
    idata: Any,
    cells: pd.DataFrame,
    *,
    group_col: str,
    labels: list[str | int],
    interval_prob: float = DEFAULT_INTERVAL_PROB,
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
        stats = _summary(draws, interval_prob=interval_prob)
        rows.append(
            {
                group_col: idx,
                "label": label,
                "observed": int(observed.get(idx, 0)),
                "predicted_mean": stats["mean"],
                "predicted_lo": stats["lo"],
                "predicted_hi": stats["hi"],
                "interval_prob": interval_prob,
                "observed_in_interval": bool(
                    stats["lo"] <= observed.get(idx, 0) <= stats["hi"]
                ),
            }
        )
    return pd.DataFrame(rows)


def posterior_predictive_age_year_table(
    idata: Any,
    cells: pd.DataFrame,
    *,
    years: list[int],
    age_labels: list[object],
    interval_prob: float = DEFAULT_INTERVAL_PROB,
) -> pd.DataFrame:
    """Summarise cell-level PPC residuals across the complete age-year grid."""
    ppc = _require_posterior_predictive(idata)
    year_idx = cells["year_idx"].to_numpy(dtype=int)
    age_idx = cells["age_idx"].to_numpy(dtype=int)
    rows = []
    for y, year in enumerate(years):
        for age, age_label in enumerate(age_labels):
            mask = (year_idx == y) & (age_idx == age)
            observed = int(cells.loc[mask, "R_cell"].sum())
            births = int(cells.loc[mask, "N_cell"].sum())
            if np.any(mask):
                draws = ppc.isel(cell=np.flatnonzero(mask)).sum(dim="cell").values
            else:
                draws = np.zeros(
                    tuple(ppc.sizes[dim] for dim in ppc.dims if dim != "cell"),
                    dtype=float,
                )
            stats = _summary(draws, interval_prob=interval_prob)
            predictive_sd = float(np.std(draws, ddof=1)) if draws.size > 1 else 0.0
            residual = observed - stats["mean"]
            rows.append(
                {
                    "year_idx": y,
                    "year": year,
                    "age_idx": age,
                    "age": age_label,
                    "births": births,
                    "observed": observed,
                    "predicted_mean": stats["mean"],
                    "predicted_lo": stats["lo"],
                    "predicted_hi": stats["hi"],
                    "posterior_predictive_sd": predictive_sd,
                    "residual_observed_minus_predicted": residual,
                    "standardized_residual": (
                        residual / predictive_sd if predictive_sd > 0.0 else np.nan
                    ),
                    "relative_residual": (
                        residual / stats["mean"] if stats["mean"] > 0.0 else np.nan
                    ),
                    "interval_prob": interval_prob,
                    "observed_in_interval": bool(
                        stats["lo"] <= observed <= stats["hi"]
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

    # Keep the seven-band plots compact while giving the 39 represented ages
    # enough horizontal space for legible endpoint and single-year labels.
    width = min(14.0, max(8.0, 0.32 * len(df)))
    fig, ax = plt.subplots(figsize=(width, 4.5))
    xpos = np.arange(len(df))
    y = df[mean].to_numpy(dtype=float)
    yerr = _interval_yerr(
        y,
        df[lo].to_numpy(dtype=float),
        df[hi].to_numpy(dtype=float),
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


def _age_year_ppc_residual_plot(df: pd.DataFrame):
    """Heatmap of age-year posterior-predictive standardized residuals."""
    import matplotlib.pyplot as plt

    years = list(df["year"].drop_duplicates())
    ages = list(df.sort_values("age_idx")["age"].drop_duplicates())
    matrix = (
        df.pivot(index="age", columns="year", values="standardized_residual")
        .reindex(index=ages, columns=years)
        .to_numpy(dtype=float)
    )
    finite = np.abs(matrix[np.isfinite(matrix)])
    limit = max(3.0, float(finite.max()) if finite.size else 3.0)
    fig, ax = plt.subplots(figsize=(10, 6.0), layout="constrained")
    image = ax.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )
    colourbar = fig.colorbar(image, ax=ax, pad=0.015)
    colourbar.set_label(
        "standardized residual (observed minus posterior-predictive mean)"
    )

    failures = df.loc[~df["observed_in_interval"]]
    if len(failures):
        year_position = {year: idx for idx, year in enumerate(years)}
        age_position = {age: idx for idx, age in enumerate(ages)}
        ax.scatter(
            failures["year"].map(year_position),
            failures["age"].map(age_position),
            marker="x",
            color="black",
            s=30,
            linewidth=1.0,
            label="observed outside posterior-predictive ETI",
        )
        ax.legend(loc="upper left", fontsize="small")

    age_ticks = np.unique(
        np.linspace(0, max(len(ages) - 1, 0), min(len(ages), 12), dtype=int)
    )
    year_ticks = np.unique(
        np.linspace(0, max(len(years) - 1, 0), min(len(years), 12), dtype=int)
    )
    ax.set_yticks(age_ticks)
    ax.set_yticklabels([str(ages[idx]) for idx in age_ticks])
    ax.set_xticks(year_ticks)
    ax.set_xticklabels(
        [str(years[idx]) for idx in year_ticks],
        rotation=35,
        ha="right",
    )
    ax.set_ylabel("maternal age")
    ax.set_xlabel("year")
    ax.set_title("Posterior-predictive residuals by maternal age and year")
    return fig


def _accounting_plot(df: pd.DataFrame, *, interval_prob: float = DEFAULT_INTERVAL_PROB):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(df))
    ax.plot(
        x, df["natural_expected_ds"], marker="o", label="age-expected before reduction"
    )
    true_mean = df["true_count_year_mean"].to_numpy(dtype=float)
    true_lo = df["true_count_year_lo"].to_numpy(dtype=float)
    true_hi = df["true_count_year_hi"].to_numpy(dtype=float)
    ax.fill_between(
        x,
        true_lo,
        true_hi,
        alpha=0.18,
        label=f"true DS {interval_label(interval_prob)} ETI",
    )
    ax.plot(x, true_mean, marker="o", label="true DS posterior mean")
    ax.bar(x, df["recorded_ds"], width=0.45, alpha=0.45, label="recorded DS")
    ax.set_xticks(x)
    ax.set_xticklabels(df["year"].astype(str), rotation=35, ha="right")
    ax.set_ylabel("DS births")
    ax.set_title("Core accounting by year")
    ax.legend()
    fig.tight_layout()
    return fig


def _reduction_plot(df: pd.DataFrame, *, interval_prob: float = DEFAULT_INTERVAL_PROB):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(df))
    ax.fill_between(
        x,
        df["rho_prior_lo"],
        df["rho_prior_hi"],
        alpha=0.16,
        label=f"prior {interval_label(interval_prob)} ETI",
    )
    ax.plot(x, df["rho_prior_mean"], "--", label="prior mean")
    post = df["rho_year_mean"].to_numpy(dtype=float)
    yerr = _interval_yerr(
        post,
        df["rho_year_lo"].to_numpy(dtype=float),
        df["rho_year_hi"].to_numpy(dtype=float),
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


def _age_reduction_plot(
    df: pd.DataFrame, *, interval_prob: float = DEFAULT_INTERVAL_PROB
):
    """Plot age-specific reduction means and the calibrated marginal."""
    import matplotlib.pyplot as plt

    years = list(df["year"].drop_duplicates())
    ages = list(df.sort_values("age_idx")["age"].drop_duplicates())
    matrix = (
        df.pivot(index="age", columns="year", values="rho_year_age_mean")
        .reindex(index=ages, columns=years)
        .to_numpy(dtype=float)
    )
    fig, (ax_heatmap, ax_marginal) = plt.subplots(
        2,
        1,
        figsize=(10, 7.2),
        height_ratios=(3.2, 1.4),
        sharex=True,
        layout="constrained",
    )
    image = ax_heatmap.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        vmin=0.0,
        vmax=min(1.0, max(0.65, float(np.nanmax(matrix)) + 0.05)),
    )
    colourbar = fig.colorbar(image, ax=ax_heatmap, pad=0.015)
    colourbar.set_label("posterior mean combined reduction")
    age_ticks = np.unique(
        np.linspace(0, max(len(ages) - 1, 0), min(len(ages), 12), dtype=int)
    )
    ax_heatmap.set_yticks(age_ticks)
    ax_heatmap.set_yticklabels([str(ages[idx]) for idx in age_ticks])
    ax_heatmap.set_ylabel("maternal age")
    ax_heatmap.set_title("Age-specific combined reduction")

    marginal = df.drop_duplicates("year").set_index("year").loc[years]
    x = np.arange(len(years))
    marginal_mean = marginal["rho_year_marginal_mean"].to_numpy(dtype=float)
    ax_marginal.fill_between(
        x,
        marginal["rho_year_marginal_lo"].to_numpy(dtype=float),
        marginal["rho_year_marginal_hi"].to_numpy(dtype=float),
        color="black",
        alpha=0.12,
        label=f"weighted marginal {interval_label(interval_prob)} ETI",
    )
    ax_marginal.plot(
        x,
        marginal_mean,
        color="black",
        linewidth=2.3,
        marker="o",
        label="natural-DS-weighted marginal",
    )
    ax_marginal.set_xticks(x)
    ax_marginal.set_xticklabels(
        [str(year) for year in years],
        rotation=35,
        ha="right",
    )
    upper = float(marginal["rho_year_marginal_hi"].max())
    ax_marginal.set_ylim(0, min(1.0, max(0.65, upper + 0.05)))
    ax_marginal.set_ylabel("marginal reduction")
    ax_marginal.set_xlabel("year")
    ax_marginal.legend(fontsize="small")
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


def _recording_s_by_year_plot(
    table: pd.DataFrame, *, interval_prob: float = DEFAULT_INTERVAL_PROB
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(table))
    ax.fill_between(
        x,
        table["prior_lo"],
        table["prior_hi"],
        alpha=0.16,
        label=f"prior {interval_label(interval_prob)} ETI",
    )
    ax.plot(x, table["prior_mean"], "--", label="prior mean")
    post = table["posterior_mean"].to_numpy(dtype=float)
    yerr = _interval_yerr(
        post,
        table["posterior_lo"].to_numpy(dtype=float),
        table["posterior_hi"].to_numpy(dtype=float),
    )
    ax.errorbar(x, post, yerr=yerr, fmt="o-", capsize=4, label="posterior")
    ax.set_xticks(x)
    ax.set_xticklabels(table["year"].astype(str), rotation=35, ha="right")
    ax.set_ylim(0, min(1.0, max(0.8, float(table["posterior_hi"].max()) + 0.05)))
    ax.set_ylabel("certificate recording sensitivity")
    ax.set_title("Recording sensitivity by year")
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
    recording_model: str = "constant",
    interval_prob: float = DEFAULT_INTERVAL_PROB,
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
        interval_prob=interval_prob,
    )
    headlines = headline_table(
        idata,
        cells,
        accounting,
        priors_config,
        interval_prob=interval_prob,
    )
    reduction = reduction_prior_posterior_table(accounting)
    recording = recording_s_table(idata, priors_config, interval_prob=interval_prob)
    recording_by_year = recording_s_by_year_table(
        idata,
        priors_config,
        years=list(accounting["year"]),
        recording_model=recording_model,
        interval_prob=interval_prob,
    )
    ppc_year = posterior_predictive_table(
        idata,
        cells,
        group_col="year_idx",
        labels=list(accounting["year"]),
        interval_prob=interval_prob,
    )
    age_reduction_variable = _age_reduction_variable(idata)
    age_labels: list[object] = AGE_LEVELS
    exact_age_model = "maternal_age" in cells
    if exact_age_model:
        age_labels = _age_labels(cells)
    elif age_reduction_variable is not None:
        age_labels = _age_labels(
            cells,
            idata.posterior[age_reduction_variable],
        )
    ppc_age = posterior_predictive_table(
        idata,
        cells,
        group_col="age_idx",
        labels=age_labels,
        interval_prob=interval_prob,
    )
    ppc_age_band = None
    if exact_age_model:
        age_band_cells = cells.copy()
        age_band_cells["age_band_idx"] = _maternal_age_band_index(
            age_band_cells["maternal_age"].to_numpy(dtype=int)
        )
        ppc_age_band = posterior_predictive_table(
            idata,
            age_band_cells,
            group_col="age_band_idx",
            labels=AGE_LEVELS,
            interval_prob=interval_prob,
        )
    age_reduction = None
    ppc_age_year = None
    if age_reduction_variable is not None:
        age_reduction = age_reduction_by_year_table(
            idata,
            cells,
            priors_config,
            year_range=year_range,
            interval_prob=interval_prob,
        )
    if exact_age_model:
        ppc_age_year = posterior_predictive_age_year_table(
            idata,
            cells,
            years=list(accounting["year"]),
            age_labels=age_labels,
            interval_prob=interval_prob,
        )

    tables = {
        "core_headlines": headlines,
        "core_accounting_by_year": accounting,
        "core_reduction_prior_posterior": reduction,
        "core_recording_s": recording,
        "core_recording_s_by_year": recording_by_year,
        "core_ppc_by_year": ppc_year,
        "core_ppc_by_age": ppc_age,
    }
    if age_reduction is not None:
        tables["core_reduction_by_age_year"] = age_reduction
    if ppc_age_year is not None:
        tables["core_ppc_by_age_year"] = ppc_age_year
    if ppc_age_band is not None:
        tables["core_ppc_by_age_band"] = ppc_age_band
    for stem, df in tables.items():
        _write_table(df, out_dir, stem)

    _save_figure(
        _accounting_plot(accounting, interval_prob=interval_prob),
        out_dir,
        "core_accounting_by_year",
    )
    _save_figure(
        _reduction_plot(reduction, interval_prob=interval_prob),
        out_dir,
        "core_reduction_prior_posterior",
    )
    if age_reduction is not None:
        _save_figure(
            _age_reduction_plot(age_reduction, interval_prob=interval_prob),
            out_dir,
            "core_reduction_by_age_year",
        )
    if ppc_age_year is not None:
        _save_figure(
            _age_year_ppc_residual_plot(ppc_age_year),
            out_dir,
            "core_ppc_by_age_year",
        )
    if ppc_age_band is not None:
        _save_figure(
            _errorbar_plot(
                ppc_age_band,
                x="label",
                observed="observed",
                mean="predicted_mean",
                lo="predicted_lo",
                hi="predicted_hi",
                ylabel="recorded DS births",
                title="Posterior predictive check by maternal-age band",
            ),
            out_dir,
            "core_ppc_by_age_band",
        )
    _save_figure(
        _recording_s_plot(idata, recording, priors_config), out_dir, "core_recording_s"
    )
    _save_figure(
        _recording_s_by_year_plot(recording_by_year, interval_prob=interval_prob),
        out_dir,
        "core_recording_s_by_year",
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
            title="Posterior predictive check by maternal age",
        ),
        out_dir,
        "core_ppc_by_age",
    )
    return tables


__all__ = [
    "accounting_by_year_table",
    "age_reduction_by_year_table",
    "headline_table",
    "posterior_predictive_table",
    "posterior_predictive_age_year_table",
    "recording_s_by_year_table",
    "recording_s_table",
    "reduction_prior_posterior_table",
    "render_core_all",
]
