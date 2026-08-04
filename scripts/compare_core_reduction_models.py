"""Compare two fitted core reduction-recording model outputs."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import arviz as az
import dse_research_utils.environment.setup as setup
import numpy as np
import pandas as pd

from dspopulations_us_birth_certificates import cli_output
from dspopulations_us_birth_certificates.chance import get_ds_lb_nt_probability_array
from dspopulations_us_birth_certificates.intervals import DEFAULT_ETI_PROB
from dspopulations_us_birth_certificates.selection.core_reduction import (
    CORE_REDUCTION_MODEL_ID,
)
from dspopulations_us_birth_certificates.selection.priors import (
    AGE_LEVELS,
    FALSE_POSITIVE_RATE,
)

HEADLINE_METRICS = (
    "true_ds_livebirths",
    "aggregate_reduction",
    "recording_s",
)
COMMON_GRID_RANDOM_SEED = 47_113
GRID_KEY_COLUMNS = ("year_idx", "age_idx", "maternal_age", "N_cell", "R_cell")


@dataclass(frozen=True)
class _CommonGridPrediction:
    """Posterior-predictive reconstruction for one model on an exact-age grid."""

    table: pd.DataFrame
    draws: np.ndarray
    revision_pooled: bool


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _run_label(run_dir: Path) -> str:
    config = _read_json(_require_file(run_dir / "config.json"))
    return str(config.get("model_id", run_dir.name))


def _headlines(run_dir: Path) -> pd.DataFrame:
    path = _require_file(run_dir / "tables" / "core_headlines.csv")
    return pd.read_csv(path).set_index("metric")


def _recording_by_year(run_dir: Path) -> pd.DataFrame:
    path = _require_file(run_dir / "tables" / "core_recording_s_by_year.csv")
    return pd.read_csv(path)


def _reduction_by_year(run_dir: Path) -> pd.DataFrame:
    path = _require_file(run_dir / "tables" / "core_reduction_prior_posterior.csv")
    return pd.read_csv(path)


def _age_reduction_by_year(run_dir: Path) -> pd.DataFrame | None:
    path = run_dir / "tables" / "core_reduction_by_age_year.csv"
    return pd.read_csv(path) if path.is_file() else None


def _run_config(run_dir: Path) -> dict[str, Any]:
    return _read_json(_require_file(run_dir / "config.json"))


def _run_cells(run_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(_require_file(run_dir / "cells.parquet"))


def _age_model(config: dict[str, Any]) -> str:
    return str(config.get("age_model", "band"))


def _false_positive_rate(config: dict[str, Any]) -> float:
    priors = config.get("priors", {})
    return float(priors.get("false_positive_rate", FALSE_POSITIVE_RATE))


def _recording_model(config: dict[str, Any]) -> str:
    return str(config.get("recording_model", "constant"))


def _maternal_age_band_index(maternal_age: np.ndarray) -> np.ndarray:
    age = np.asarray(maternal_age, dtype=int)
    return np.select(
        [age < 20, age < 25, age < 30, age < 35, age < 40, age < 45],
        [0, 1, 2, 3, 4, 5],
        default=6,
    ).astype(int)


def _pool_cells_over_revision(cells: pd.DataFrame) -> pd.DataFrame:
    """Sum revised and unrevised cells back into one cell per age-year.

    ``prepare_core_age_year_cells(..., split_revision=True)`` emits one cell per
    certificate version, so a revision fit's grid carries two rows per
    (year_idx, age_idx). Reporting keys on the pooled cell that the two versions
    jointly describe, matching ``posterior_predictive_age_year_table`` in the
    single-run reports. Cells without a ``revised`` column pass through.
    """
    if "revised" not in cells:
        return cells.reset_index(drop=True)
    keys = [
        column
        for column in ("year_idx", "age_idx", "maternal_age", "maternal_age_label")
        if column in cells
    ]
    return (
        cells.groupby(keys, as_index=False, observed=True)[["N_cell", "R_cell"]]
        .sum()
        .sort_values(["year_idx", "age_idx"], ignore_index=True)
    )


def _revision_pooling_matrix(
    cells: pd.DataFrame,
    pooled: pd.DataFrame,
) -> np.ndarray:
    """Return the 0/1 matrix that sums split cells into their pooled cell."""
    keys = ["year_idx", "age_idx"]
    target = pd.MultiIndex.from_arrays(
        [pooled[key].to_numpy(dtype="int64") for key in keys]
    ).get_indexer(
        pd.MultiIndex.from_arrays([cells[key].to_numpy(dtype="int64") for key in keys])
    )
    if np.any(target < 0):
        raise ValueError("Revision-split cells do not map onto the pooled grid.")
    matrix = np.zeros((len(cells), len(pooled)), dtype="int64")
    matrix[np.arange(len(cells)), target] = 1
    return matrix


def _sorted_exact_cells(run_dir: Path) -> pd.DataFrame:
    cells = _run_cells(run_dir)
    sort_columns = ["year_idx", "age_idx"]
    if "revised" in cells:
        sort_columns.append("revised")
    return cells.sort_values(sort_columns, ignore_index=True)


def _grid_signature(cells: pd.DataFrame, *, split: bool) -> pd.DataFrame:
    """Return a dtype-normalised grid identity for cross-run comparison."""
    frame = cells if split else _pool_cells_over_revision(cells)
    columns = [*GRID_KEY_COLUMNS, *(["revised"] if split else [])]
    return pd.DataFrame(
        {column: frame[column].to_numpy(dtype="int64") for column in columns}
    )


def _exact_grid_for_comparison(
    baseline_dir: Path,
    extension_dir: Path,
) -> pd.DataFrame | None:
    """Resolve and validate the shared exact-age evaluation grid."""
    baseline_config = _run_config(baseline_dir)
    extension_config = _run_config(extension_dir)
    exact_dirs = [
        run_dir
        for run_dir, config in (
            (baseline_dir, baseline_config),
            (extension_dir, extension_config),
        )
        if _age_model(config) == "single_year"
    ]
    if not exact_dirs:
        return None

    recorded_definitions = {
        str(config.get("recorded_definition", "confirmed_or_pending"))
        for config in (baseline_config, extension_config)
    }
    if len(recorded_definitions) != 1:
        raise ValueError(
            "Common-grid PPC requires the same recorded_definition in both runs."
        )
    if baseline_config.get("year_range") != extension_config.get("year_range"):
        raise ValueError("Common-grid PPC requires identical year ranges.")
    if not np.isclose(
        _false_positive_rate(baseline_config),
        _false_positive_rate(extension_config),
    ):
        raise ValueError("Common-grid PPC requires the same false-positive rate.")

    candidates = [_sorted_exact_cells(run_dir) for run_dir in exact_dirs]
    for candidate in candidates:
        missing = set(GRID_KEY_COLUMNS).difference(candidate.columns)
        if missing:
            raise ValueError(
                f"Exact-age cells are missing columns: {sorted(missing)!r}."
            )
    # Prefer a revision-split grid when one run supplies it: it is the finest
    # shared cell definition, and a fit made on pooled cells reconstructs on it
    # exactly, because both halves of a cell then share one sensitivity and
    # their predictive counts sum back to the pooled cell's.
    exact = next(
        (cells for cells in candidates if "revised" in cells),
        candidates[0],
    )

    for candidate in candidates:
        if candidate is exact:
            continue
        # Grid identity is the pooled age-year cell, so a revision-split run and
        # a pooled one over the same births count as the same grid.
        both_split = "revised" in exact and "revised" in candidate
        if not _grid_signature(exact, split=both_split).equals(
            _grid_signature(candidate, split=both_split)
        ):
            raise ValueError("Exact-age runs do not share the same evaluation grid.")

    for run_dir in (baseline_dir, extension_dir):
        run_cells = _run_cells(run_dir)
        if int(run_cells["N_cell"].sum()) != int(exact["N_cell"].sum()):
            raise ValueError("Compared runs do not use the same birth cohort.")
        if int(run_cells["R_cell"].sum()) != int(exact["R_cell"].sum()):
            raise ValueError("Compared runs do not use the same recorded-case cohort.")
    return exact


def _stack_posterior(da: Any, *tail_dims: str) -> np.ndarray:
    return np.asarray(
        da.stack(sample=("chain", "draw")).transpose("sample", *tail_dims).values,
        dtype=float,
    )


def _theta_on_exact_grid(
    idata: Any,
    config: dict[str, Any],
    exact_cells: pd.DataFrame,
) -> tuple[np.ndarray, str]:
    maternal_age = exact_cells["maternal_age"].to_numpy(dtype=int)
    if _age_model(config) == "band":
        theta_band = np.asarray(config["priors"]["theta_lb_age"], dtype=float)
        if len(theta_band) != len(AGE_LEVELS):
            raise ValueError("Band model theta_lb_age must have seven values.")
        return theta_band[_maternal_age_band_index(maternal_age)], "seven_band"

    age_table = (
        exact_cells[["age_idx", "maternal_age"]]
        .drop_duplicates()
        .sort_values("age_idx")
    )
    age_values = age_table["maternal_age"].to_numpy(dtype=int)
    if hasattr(idata, "constant_data") and "theta_lb_age" in idata.constant_data:
        theta_age = np.asarray(idata.constant_data["theta_lb_age"].values, dtype=float)
    else:
        theta_age = np.asarray(get_ds_lb_nt_probability_array(age_values), dtype=float)
    if len(theta_age) != len(age_values):
        raise ValueError("Exact-age theta does not match the exact-age grid.")
    return theta_age[exact_cells["age_idx"].to_numpy(dtype=int)], "single_year"


def _recording_s_on_exact_grid(
    idata: Any,
    config: dict[str, Any],
    exact_cells: pd.DataFrame,
) -> tuple[np.ndarray, str]:
    """Return per-cell recording sensitivity draws for a supplied grid.

    A revision fit's ``recording_s_year`` is the *revised* sensitivity held
    constant across years, so the unrevised cells have to be reconstructed from
    ``recording_s_unrevised`` instead; using the year series for both would
    overstate recorded counts through the 2004-2015 phase-in.
    """
    recording_model = _recording_model(config)
    if recording_model != "revision":
        recording_s_year = _stack_posterior(
            idata.posterior["recording_s_year"],
            "year",
        )
        return (
            recording_s_year[:, exact_cells["year_idx"].to_numpy(dtype=int)],
            recording_model,
        )

    if "revised" not in exact_cells:
        raise ValueError(
            "A revision-split fit cannot be reconstructed on a grid without a "
            "'revised' column; recording_s_year carries only the revised "
            "sensitivity."
        )
    missing = [
        name
        for name in ("recording_s", "recording_s_unrevised")
        if name not in idata.posterior
    ]
    if missing:
        raise ValueError(
            f"recording_model='revision' posterior is missing {missing!r}."
        )
    revised_cell = exact_cells["revised"].to_numpy(dtype=int)
    if not np.all(np.isin(revised_cell, (0, 1))):
        raise ValueError("revised must be 0/1 in a revision-split grid.")
    s_revised = _stack_posterior(idata.posterior["recording_s"])
    s_unrevised = _stack_posterior(idata.posterior["recording_s_unrevised"])
    s_cell = np.where(
        revised_cell[None, :] == 1,
        s_revised[:, None],
        s_unrevised[:, None],
    )
    return s_cell, "revision_split"


def _common_grid_prediction(
    run_dir: Path,
    exact_cells: pd.DataFrame,
    *,
    random_seed: int = COMMON_GRID_RANDOM_SEED,
    interval_prob: float = DEFAULT_ETI_PROB,
) -> _CommonGridPrediction:
    """Reconstruct one fitted model's PPC on a supplied exact-age grid.

    Draws are taken per supplied cell, then pooled over the revision dimension,
    so the returned table and draws carry one entry per (year_idx, age_idx).
    """
    config = _run_config(run_dir)
    idata = az.from_netcdf(_require_file(run_dir / "idata.nc"))
    theta_cell, theta_resolution = _theta_on_exact_grid(idata, config, exact_cells)
    year_idx = exact_cells["year_idx"].to_numpy(dtype=int)
    age_idx = exact_cells["age_idx"].to_numpy(dtype=int)
    n_cell = exact_cells["N_cell"].to_numpy(dtype=int)
    rho_variable = next(
        (name for name in ("rho_year_age", "rho_age_year") if name in idata.posterior),
        None,
    )
    if rho_variable is None:
        rho_year = _stack_posterior(idata.posterior["rho_year"], "year")
        survival = 1.0 - rho_year[:, year_idx]
    else:
        rho_year_age = _stack_posterior(
            idata.posterior[rho_variable],
            "year",
            "age",
        )
        survival = 1.0 - rho_year_age[:, year_idx, age_idx]
    s_cell, recording_resolution = _recording_s_on_exact_grid(
        idata,
        config,
        exact_cells,
    )
    if s_cell.shape[0] != survival.shape[0]:
        raise ValueError("rho and recording sensitivity have incompatible draw counts.")
    p_ds_lb = theta_cell[None, :] * survival
    false_positive_rate = _false_positive_rate(config)
    p_recorded = p_ds_lb * s_cell + (1.0 - p_ds_lb) * false_positive_rate
    rng = np.random.default_rng(random_seed)
    predictive = rng.binomial(n_cell[None, :], np.clip(p_recorded, 0.0, 1.0))

    # Summarise on the pooled cell: the standardised residual belongs to the
    # age-year cell that the two certificate versions jointly describe, and
    # summing the draws first keeps the pooled interval and sd exact.
    pooled_cells = _pool_cells_over_revision(exact_cells)
    revision_pooled = len(pooled_cells) != len(exact_cells)
    if revision_pooled:
        predictive = predictive @ _revision_pooling_matrix(exact_cells, pooled_cells)

    tail = (1.0 - interval_prob) / 2.0
    mean = predictive.mean(axis=0)
    lo, hi = np.quantile(predictive, [tail, 1.0 - tail], axis=0)
    sd = predictive.std(axis=0, ddof=1) if len(predictive) > 1 else np.zeros(len(mean))
    observed = pooled_cells["R_cell"].to_numpy(dtype=int)
    residual = observed - mean
    year_start = int(config["year_range"][0])
    labels = (
        pooled_cells["maternal_age_label"].astype(str)
        if "maternal_age_label" in pooled_cells
        else pooled_cells["maternal_age"].astype(str)
    )
    table = pooled_cells[
        ["year_idx", "age_idx", "maternal_age", "N_cell", "R_cell"]
    ].copy()
    table.insert(1, "year", year_start + table["year_idx"])
    table.insert(4, "age", labels.to_numpy())
    table = table.rename(columns={"N_cell": "births", "R_cell": "observed"})
    table["predicted_mean"] = mean
    table["predicted_lo"] = lo
    table["predicted_hi"] = hi
    table["posterior_predictive_sd"] = sd
    table["residual_observed_minus_predicted"] = residual
    table["standardized_residual"] = np.divide(
        residual,
        sd,
        out=np.full_like(residual, np.nan, dtype=float),
        where=sd > 0.0,
    )
    table["observed_in_interval"] = (lo <= observed) & (observed <= hi)
    table["model_id"] = str(config.get("model_id", run_dir.name))
    table["theta_resolution"] = theta_resolution
    table["recording_resolution"] = recording_resolution
    table["interval_prob"] = interval_prob
    return _CommonGridPrediction(
        table=table,
        draws=predictive,
        revision_pooled=revision_pooled,
    )


def headline_comparison_table(
    baseline_dir: Path,
    extension_dir: Path,
    *,
    metrics: tuple[str, ...] = HEADLINE_METRICS,
) -> pd.DataFrame:
    """Return a direct headline comparison for two fitted runs."""
    baseline_id = _run_label(baseline_dir)
    extension_id = _run_label(extension_dir)
    baseline = _headlines(baseline_dir)
    extension = _headlines(extension_dir)
    rows = []
    for metric in metrics:
        if metric not in baseline.index or metric not in extension.index:
            continue
        base_row = baseline.loc[metric]
        ext_row = extension.loc[metric]
        rows.append(
            {
                "metric": metric,
                "baseline_model_id": baseline_id,
                "extension_model_id": extension_id,
                "baseline_mean": base_row["mean"],
                "baseline_lo": base_row["lo"],
                "baseline_hi": base_row["hi"],
                "extension_mean": ext_row["mean"],
                "extension_lo": ext_row["lo"],
                "extension_hi": ext_row["hi"],
                "extension_minus_baseline_mean": (ext_row["mean"] - base_row["mean"]),
                "baseline_notes": base_row.get("notes", ""),
                "extension_notes": ext_row.get("notes", ""),
            }
        )
    return pd.DataFrame(rows)


def recording_s_year_comparison_table(
    baseline_dir: Path,
    extension_dir: Path,
) -> pd.DataFrame:
    """Return a by-year comparison of recording sensitivity summaries."""
    baseline_id = _run_label(baseline_dir)
    extension_id = _run_label(extension_dir)
    baseline = _recording_by_year(baseline_dir)
    extension = _recording_by_year(extension_dir)

    keep = ["year", "posterior_mean", "posterior_lo", "posterior_hi"]
    merged = baseline[keep].merge(
        extension[keep],
        on="year",
        how="inner",
        suffixes=("_baseline", "_extension"),
    )
    merged.insert(1, "baseline_model_id", baseline_id)
    merged.insert(2, "extension_model_id", extension_id)
    merged["extension_minus_baseline_mean"] = (
        merged["posterior_mean_extension"] - merged["posterior_mean_baseline"]
    )
    return merged


def age_reduction_comparison_table(
    baseline_dir: Path,
    extension_dir: Path,
) -> pd.DataFrame:
    """Compare an age-specific extension with its baseline reduction.

    If the baseline has no age-specific table, its common ``rho_year`` is
    repeated across age cells. This is the direct DSP001-to-DSP003 comparison.
    """
    baseline_id = _run_label(baseline_dir)
    extension_id = _run_label(extension_dir)
    extension = _age_reduction_by_year(extension_dir)
    if extension is None:
        raise ValueError(
            f"{extension_dir} has no core_reduction_by_age_year.csv artifact."
        )

    required_extension = {
        "year",
        "age_idx",
        "age",
        "rho_year_age_mean",
        "rho_year_age_lo",
        "rho_year_age_hi",
        "rho_year_marginal_mean",
        "rho_year_marginal_lo",
        "rho_year_marginal_hi",
    }
    missing_extension = required_extension.difference(extension.columns)
    if missing_extension:
        raise ValueError(
            "Age-specific extension table is missing columns: "
            f"{sorted(missing_extension)!r}."
        )

    baseline_age = _age_reduction_by_year(baseline_dir)
    if baseline_age is None:
        baseline_year = _reduction_by_year(baseline_dir)
        required_baseline = {"year", "rho_year_mean", "rho_year_lo", "rho_year_hi"}
        missing_baseline = required_baseline.difference(baseline_year.columns)
        if missing_baseline:
            raise ValueError(
                "Baseline reduction table is missing columns: "
                f"{sorted(missing_baseline)!r}."
            )
        baseline = extension[["year", "age_idx", "age"]].merge(
            baseline_year[["year", "rho_year_mean", "rho_year_lo", "rho_year_hi"]],
            on="year",
            how="left",
            validate="many_to_one",
        )
        baseline = baseline.rename(
            columns={
                "rho_year_mean": "baseline_mean",
                "rho_year_lo": "baseline_lo",
                "rho_year_hi": "baseline_hi",
            }
        )
        baseline["baseline_quantity"] = "common reduction across age cells"
    else:
        required_baseline_age = {
            "year",
            "age_idx",
            "age",
            "rho_year_age_mean",
            "rho_year_age_lo",
            "rho_year_age_hi",
        }
        missing_baseline_age = required_baseline_age.difference(baseline_age.columns)
        if missing_baseline_age:
            raise ValueError(
                "Age-specific baseline table is missing columns: "
                f"{sorted(missing_baseline_age)!r}."
            )
        baseline = baseline_age[
            [
                "year",
                "age_idx",
                "age",
                "rho_year_age_mean",
                "rho_year_age_lo",
                "rho_year_age_hi",
            ]
        ].rename(
            columns={
                "rho_year_age_mean": "baseline_mean",
                "rho_year_age_lo": "baseline_lo",
                "rho_year_age_hi": "baseline_hi",
            }
        )
        baseline["baseline_quantity"] = "age-specific reduction"

    extension_columns = [
        "year",
        "age_idx",
        "age",
        "rho_year_age_mean",
        "rho_year_age_lo",
        "rho_year_age_hi",
        "rho_year_marginal_mean",
        "rho_year_marginal_lo",
        "rho_year_marginal_hi",
    ]
    for optional in (
        "maternal_age",
        "sparse_boundary_age",
        "maternal_age_endpoint_capped",
        "natural_expected_ds",
        "natural_ds_weight_share",
        "rho_year_marginal_max_abs_draw_difference",
    ):
        if optional in extension.columns:
            extension_columns.append(optional)
    merged = baseline.merge(
        extension[extension_columns],
        on=["year", "age_idx", "age"],
        how="inner",
        validate="one_to_one",
    ).rename(
        columns={
            "rho_year_age_mean": "extension_mean",
            "rho_year_age_lo": "extension_lo",
            "rho_year_age_hi": "extension_hi",
            "rho_year_marginal_mean": "extension_marginal_mean",
            "rho_year_marginal_lo": "extension_marginal_lo",
            "rho_year_marginal_hi": "extension_marginal_hi",
        }
    )
    merged.insert(3, "baseline_model_id", baseline_id)
    merged.insert(4, "extension_model_id", extension_id)
    merged["extension_quantity"] = "age-specific reduction"
    merged["extension_minus_baseline_mean"] = (
        merged["extension_mean"] - merged["baseline_mean"]
    )
    return merged.sort_values(["year", "age_idx"], ignore_index=True)


def common_grid_ppc_comparison_table(
    baseline: _CommonGridPrediction,
    extension: _CommonGridPrediction,
) -> pd.DataFrame:
    """Compare two reconstructions on the same exact age-by-year grid."""
    keys = [
        "year_idx",
        "year",
        "age_idx",
        "maternal_age",
        "age",
        "births",
        "observed",
    ]
    metrics = [
        "predicted_mean",
        "predicted_lo",
        "predicted_hi",
        "posterior_predictive_sd",
        "residual_observed_minus_predicted",
        "standardized_residual",
        "observed_in_interval",
        "model_id",
        "theta_resolution",
        "recording_resolution",
    ]
    merged = baseline.table[keys + metrics].merge(
        extension.table[keys + metrics],
        on=keys,
        how="inner",
        suffixes=("_baseline", "_extension"),
        validate="one_to_one",
    )
    if len(merged) != len(baseline.table) or len(merged) != len(extension.table):
        raise ValueError("Common-grid prediction tables do not cover the same cells.")
    merged["extension_minus_baseline_predicted_mean"] = (
        merged["predicted_mean_extension"] - merged["predicted_mean_baseline"]
    )
    merged["extension_minus_baseline_absolute_residual"] = (
        merged["residual_observed_minus_predicted_extension"].abs()
        - merged["residual_observed_minus_predicted_baseline"].abs()
    )
    merged["extension_minus_baseline_absolute_standardized_residual"] = (
        merged["standardized_residual_extension"].abs()
        - merged["standardized_residual_baseline"].abs()
    )
    merged["comparison_grid"] = "shared exact maternal-age by year cells"
    merged["revision_pooling"] = (
        "revised and unrevised cells summed"
        if baseline.revision_pooled
        else "not applicable; cells are not revision-split"
    )
    merged["evidence_scope"] = (
        "reconstructed in-sample posterior-predictive check; not held-out evidence"
    )
    return merged


def common_grid_ppc_summary_table(comparison: pd.DataFrame) -> pd.DataFrame:
    """Return compact, scale-aware summaries of common-grid PPC calibration."""
    rows = []
    for role in ("baseline", "extension"):
        residual = comparison[f"residual_observed_minus_predicted_{role}"].to_numpy(
            dtype=float
        )
        standardized = comparison[f"standardized_residual_{role}"].to_numpy(dtype=float)
        finite_standardized = standardized[np.isfinite(standardized)]
        coverage = comparison[f"observed_in_interval_{role}"].astype(bool)
        rows.append(
            {
                "comparison_role": role,
                "model_id": comparison[f"model_id_{role}"].iloc[0],
                "theta_resolution": comparison[f"theta_resolution_{role}"].iloc[0],
                "recording_resolution": (
                    comparison[f"recording_resolution_{role}"].iloc[0]
                ),
                "n_exact_age_year_cells": len(comparison),
                "coverage_count": int(coverage.sum()),
                "coverage_fraction": float(coverage.mean()),
                "mean_absolute_standardized_residual": float(
                    np.mean(np.abs(finite_standardized))
                )
                if finite_standardized.size
                else np.nan,
                "root_mean_squared_residual": float(np.sqrt(np.mean(residual**2))),
                "total_absolute_residual": float(np.abs(residual).sum()),
                "evidence_scope": (
                    "reconstructed in-sample posterior-predictive check; "
                    "not held-out evidence"
                ),
            }
        )
    return pd.DataFrame(rows)


def _band_prediction_table(
    prediction: _CommonGridPrediction,
    exact_cells: pd.DataFrame,
    *,
    interval_prob: float = DEFAULT_ETI_PROB,
) -> pd.DataFrame:
    """Reaggregate a common-grid reconstruction to the seven reporting bands.

    ``exact_cells`` must be revision-pooled, so its rows line up with the
    pooled draws that ``_common_grid_prediction`` returns.
    """
    band_idx = _maternal_age_band_index(exact_cells["maternal_age"].to_numpy(dtype=int))
    if prediction.draws.shape[1] != len(exact_cells):
        raise ValueError("Band reaggregation cells do not match the prediction draws.")
    tail = (1.0 - interval_prob) / 2.0
    rows = []
    for idx, label in enumerate(AGE_LEVELS):
        mask = band_idx == idx
        if not np.any(mask):
            continue
        draws = prediction.draws[:, mask].sum(axis=1)
        observed = int(exact_cells.loc[mask, "R_cell"].sum())
        mean = float(draws.mean())
        lo, hi = np.quantile(draws, [tail, 1.0 - tail])
        sd = float(draws.std(ddof=1)) if len(draws) > 1 else 0.0
        residual = observed - mean
        rows.append(
            {
                "age_band_idx": idx,
                "age_band": label,
                "births": int(exact_cells.loc[mask, "N_cell"].sum()),
                "observed": observed,
                "predicted_mean": mean,
                "predicted_lo": float(lo),
                "predicted_hi": float(hi),
                "posterior_predictive_sd": sd,
                "residual_observed_minus_predicted": residual,
                "standardized_residual": residual / sd if sd > 0.0 else np.nan,
                "observed_in_interval": bool(lo <= observed <= hi),
                "model_id": prediction.table["model_id"].iloc[0],
                "theta_resolution": prediction.table["theta_resolution"].iloc[0],
                "recording_resolution": (
                    prediction.table["recording_resolution"].iloc[0]
                ),
                "interval_prob": interval_prob,
            }
        )
    return pd.DataFrame(rows)


def common_age_band_ppc_comparison_table(
    baseline: _CommonGridPrediction,
    extension: _CommonGridPrediction,
    exact_cells: pd.DataFrame,
) -> pd.DataFrame:
    """Compare reconstructed PPCs after a shared seven-band reaggregation."""
    pooled_cells = _pool_cells_over_revision(exact_cells)
    baseline_band = _band_prediction_table(baseline, pooled_cells)
    extension_band = _band_prediction_table(extension, pooled_cells)
    keys = ["age_band_idx", "age_band", "births", "observed"]
    metrics = [
        "predicted_mean",
        "predicted_lo",
        "predicted_hi",
        "posterior_predictive_sd",
        "residual_observed_minus_predicted",
        "standardized_residual",
        "observed_in_interval",
        "model_id",
        "theta_resolution",
        "recording_resolution",
    ]
    merged = baseline_band[keys + metrics].merge(
        extension_band[keys + metrics],
        on=keys,
        how="inner",
        suffixes=("_baseline", "_extension"),
        validate="one_to_one",
    )
    merged["extension_minus_baseline_absolute_residual"] = (
        merged["residual_observed_minus_predicted_extension"].abs()
        - merged["residual_observed_minus_predicted_baseline"].abs()
    )
    merged["comparison_grid"] = (
        "shared exact-age cells reaggregated to seven maternal-age bands"
    )
    return merged


def _recording_comparison_plot(df: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(df))
    labels = (
        str(df["baseline_model_id"].iloc[0]),
        str(df["extension_model_id"].iloc[0]),
    )
    for label, suffix, marker in (
        (labels[0], "baseline", "o"),
        (labels[1], "extension", "s"),
    ):
        mean = df[f"posterior_mean_{suffix}"].to_numpy(dtype=float)
        lo = df[f"posterior_lo_{suffix}"].to_numpy(dtype=float)
        hi = df[f"posterior_hi_{suffix}"].to_numpy(dtype=float)
        ax.errorbar(
            x,
            mean,
            yerr=np.vstack((np.maximum(mean - lo, 0.0), np.maximum(hi - mean, 0.0))),
            fmt=f"{marker}-",
            capsize=4,
            label=label,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(df["year"].astype(str), rotation=35, ha="right")
    ax.set_ylabel("certificate recording sensitivity")
    ax.set_title("Certificate recording sensitivity by year")
    ax.legend()
    fig.tight_layout()
    return fig


def _age_reduction_comparison_plot(df: pd.DataFrame):
    import matplotlib.pyplot as plt

    years = list(df["year"].drop_duplicates())
    ages = list(df.sort_values("age_idx")["age"].drop_duplicates())
    matrix = (
        df.pivot(
            index="age",
            columns="year",
            values="extension_minus_baseline_mean",
        )
        .reindex(index=ages, columns=years)
        .to_numpy(dtype=float)
    )
    limit = max(float(np.nanmax(np.abs(matrix))), 1e-6)
    fig, ax = plt.subplots(figsize=(10, 5.8), layout="constrained")
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
    colourbar.set_label("extension minus baseline posterior mean")
    age_ticks = np.unique(
        np.linspace(0, max(len(ages) - 1, 0), min(len(ages), 12), dtype=int)
    )
    ax.set_yticks(age_ticks)
    ax.set_yticklabels([str(ages[idx]) for idx in age_ticks])
    ax.set_xticks(np.arange(len(years)))
    ax.set_xticklabels([str(year) for year in years], rotation=35, ha="right")
    ax.set_ylabel("maternal age")
    ax.set_xlabel("year")
    ax.set_title("Age-specific reduction relative to the baseline")
    return fig


def _common_grid_ppc_comparison_plot(df: pd.DataFrame):
    import matplotlib.pyplot as plt

    years = list(df["year"].drop_duplicates())
    ages = list(df.sort_values("age_idx")["age"].drop_duplicates())
    matrix = (
        df.pivot(
            index="age",
            columns="year",
            values="extension_minus_baseline_absolute_standardized_residual",
        )
        .reindex(index=ages, columns=years)
        .to_numpy(dtype=float)
    )
    limit = max(float(np.nanmax(np.abs(matrix))), 1e-6)
    fig, ax = plt.subplots(figsize=(10, 5.8), layout="constrained")
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
        "extension minus baseline absolute standardized residual; "
        "negative favours extension"
    )
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
    ax.set_title("Common-grid posterior-predictive residual comparison")
    return fig


def _common_age_band_ppc_comparison_plot(df: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    x = np.arange(len(df))
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.plot(
        x,
        df["residual_observed_minus_predicted_baseline"],
        marker="o",
        label=str(df["model_id_baseline"].iloc[0]),
    )
    ax.plot(
        x,
        df["residual_observed_minus_predicted_extension"],
        marker="s",
        label=str(df["model_id_extension"].iloc[0]),
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df["age_band"], rotation=35, ha="right")
    ax.set_ylabel("observed minus posterior-predictive mean")
    ax.set_title("Common-grid PPC reaggregated to maternal-age bands")
    ax.legend()
    return fig


def compare_core_model_outputs(
    baseline_dir: Path,
    extension_dir: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Write direct comparison tables and plots for two fitted core models."""
    baseline_dir = Path(baseline_dir)
    extension_dir = Path(extension_dir)
    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    headline = headline_comparison_table(baseline_dir, extension_dir)
    recording = recording_s_year_comparison_table(baseline_dir, extension_dir)

    headline_path = tables_dir / "core_model_headline_comparison.csv"
    recording_path = tables_dir / "core_model_recording_s_year_comparison.csv"
    headline.to_csv(headline_path, index=False)
    recording.to_csv(recording_path, index=False)

    fig = _recording_comparison_plot(recording)
    png_path = plots_dir / "core_model_recording_s_year_comparison.png"
    svg_path = plots_dir / "core_model_recording_s_year_comparison.svg"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)

    age_reduction = _age_reduction_by_year(extension_dir)
    age_paths: dict[str, Path] = {}
    if age_reduction is not None:
        age_comparison = age_reduction_comparison_table(
            baseline_dir,
            extension_dir,
        )
        age_table_path = tables_dir / "core_model_age_reduction_comparison.csv"
        age_comparison.to_csv(age_table_path, index=False)
        age_fig = _age_reduction_comparison_plot(age_comparison)
        age_png_path = plots_dir / "core_model_age_reduction_comparison.png"
        age_svg_path = plots_dir / "core_model_age_reduction_comparison.svg"
        age_fig.savefig(age_png_path, dpi=150, bbox_inches="tight")
        age_fig.savefig(age_svg_path, bbox_inches="tight")
        plt.close(age_fig)
        age_paths = {
            "age_reduction": age_table_path,
            "age_reduction_plot_png": age_png_path,
            "age_reduction_plot_svg": age_svg_path,
        }

    exact_grid = _exact_grid_for_comparison(baseline_dir, extension_dir)
    common_grid_paths: dict[str, Path] = {}
    if exact_grid is not None:
        baseline_prediction = _common_grid_prediction(baseline_dir, exact_grid)
        extension_prediction = _common_grid_prediction(extension_dir, exact_grid)
        common_grid = common_grid_ppc_comparison_table(
            baseline_prediction,
            extension_prediction,
        )
        common_bands = common_age_band_ppc_comparison_table(
            baseline_prediction,
            extension_prediction,
            exact_grid,
        )
        common_grid_summary = common_grid_ppc_summary_table(common_grid)
        common_grid_table_path = tables_dir / "core_model_common_grid_ppc.csv"
        common_grid_summary_path = tables_dir / "core_model_common_grid_ppc_summary.csv"
        common_band_table_path = tables_dir / "core_model_common_age_band_ppc.csv"
        common_grid.to_csv(common_grid_table_path, index=False)
        common_grid_summary.to_csv(common_grid_summary_path, index=False)
        common_bands.to_csv(common_band_table_path, index=False)

        common_grid_fig = _common_grid_ppc_comparison_plot(common_grid)
        common_grid_png_path = plots_dir / "core_model_common_grid_ppc.png"
        common_grid_svg_path = plots_dir / "core_model_common_grid_ppc.svg"
        common_grid_fig.savefig(common_grid_png_path, dpi=150, bbox_inches="tight")
        common_grid_fig.savefig(common_grid_svg_path, bbox_inches="tight")
        plt.close(common_grid_fig)

        common_band_fig = _common_age_band_ppc_comparison_plot(common_bands)
        common_band_png_path = plots_dir / "core_model_common_age_band_ppc.png"
        common_band_svg_path = plots_dir / "core_model_common_age_band_ppc.svg"
        common_band_fig.savefig(common_band_png_path, dpi=150, bbox_inches="tight")
        common_band_fig.savefig(common_band_svg_path, bbox_inches="tight")
        plt.close(common_band_fig)
        common_grid_paths = {
            "common_grid_ppc": common_grid_table_path,
            "common_grid_ppc_summary": common_grid_summary_path,
            "common_grid_ppc_plot_png": common_grid_png_path,
            "common_grid_ppc_plot_svg": common_grid_svg_path,
            "common_age_band_ppc": common_band_table_path,
            "common_age_band_ppc_plot_png": common_band_png_path,
            "common_age_band_ppc_plot_svg": common_band_svg_path,
        }

    config = {
        "baseline_dir": str(baseline_dir),
        "extension_dir": str(extension_dir),
        "baseline_model_id": _run_label(baseline_dir),
        "extension_model_id": _run_label(extension_dir),
        "age_reduction_comparison": age_reduction is not None,
        "common_grid_ppc_comparison": exact_grid is not None,
        "common_grid_revision_pooled": (
            exact_grid is not None and "revised" in exact_grid
        ),
        "raw_loo_or_waic_compared": False,
        "information_criterion_note": (
            "Raw pointwise LOO/WAIC is not compared across different cell "
            "aggregations. Common-grid posterior-predictive checks are used instead."
        ),
        "common_grid_evidence_scope": (
            "Reconstructed in-sample posterior-predictive checks; not held-out "
            "predictive evidence."
        ),
    }
    config_path = output_dir / "comparison_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    paths = {
        "headline": headline_path,
        "recording": recording_path,
        "recording_plot_png": png_path,
        "recording_plot_svg": svg_path,
        "config": config_path,
    }
    paths.update(age_paths)
    paths.update(common_grid_paths)
    return paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare two core reduction-recording model fit outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("baseline_dir", type=Path)
    p.add_argument("extension_dir", type=Path)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory (default: output/selection_core_reduction/"
            "comparisons/<baseline>-vs-<extension>/<timestamp>)."
        ),
    )
    ns = p.parse_args(argv)
    if ns.output_dir is None:
        baseline_id = _run_label(ns.baseline_dir)
        extension_id = _run_label(ns.extension_dir)
        ns.output_dir = (
            Path("output")
            / CORE_REDUCTION_MODEL_ID
            / "comparisons"
            / f"{baseline_id}-vs-{extension_id}"
            / datetime.now().strftime("%Y%m%d-%H%M%S")
        )
    return ns


def main(argv: list[str] | None = None) -> int:
    ns = parse_args(argv)
    setup.init_script()

    cli_output.banner(
        "compare_core_reduction_models",
        f"{_run_label(ns.baseline_dir)} vs {_run_label(ns.extension_dir)}",
    )
    paths = compare_core_model_outputs(
        ns.baseline_dir,
        ns.extension_dir,
        ns.output_dir,
    )
    cli_output.success(f"comparison tables and plots -> {ns.output_dir}")
    cli_output.print_kv("Outputs", paths.items())
    return 0


if __name__ == "__main__":
    sys.exit(main())
