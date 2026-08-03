"""Audit DSP004 race composition against centred surveillance prevalence.

This command is deliberately read-only. It reconstructs the fitted DSP004
cohort by exact maternal age, race, and year; verifies that collapsing race
reproduces the saved fit cells; and combines the already-sampled national
``eta_year`` draws with maternal-race exposures over each five-year window. It
does not fit race effects. The 2016-labelled 2014-2018 window is explicitly
unsupported because the saved fit starts in 2016; only the 2018-labelled
2016-2020 window has complete model support.

Only finite ``surveillance_prev_per10k`` observations are eligible. The
regression-filled ``est_true_*`` columns are retained for provenance checks but
are never used as substitutes for missing surveillance observations. The
source prevalence is confirmed as a pooled count ratio, so the model analogue
pools five-year true counts and births before calculating prevalence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as importlib_metadata
import json
import platform
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import arviz as az
import dse_research_utils.environment.setup as setup
import dse_research_utils.plot.styles as plot_styles
import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2

from dspopulations_us_birth_certificates.intervals import (
    DEFAULT_ETI_PROB,
    interval_label,
)
from dspopulations_us_birth_certificates.selection.data import (
    RACE_MAP,
    RACE_UNKNOWN_IDX,
    case_from_map,
)
from dspopulations_us_birth_certificates.selection.priors import RACE_LEVELS

DEFAULT_DB_PATH = Path("data/us_births.db")
REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "notes/20260803-dsp004-race-surveillance-audit.md"
DEFAULT_SURVEILLANCE_CSV = Path(
    "data/us-births-degraaf-prevalence-recording-2000-2024.csv"
)
DEFAULT_OUTPUT_ROOT = Path(
    "output/selection_core_reduction/audits/DSP004-race-surveillance"
)
DEFAULT_SURVEILLANCE_YEARS = (2016, 2018)
CENTERED_WINDOW_HALF_WIDTH = 2
SOURCE_AGGREGATION_OPERATOR = "pooled_count_ratio"
MODEL_AGGREGATION_CONSTRUCTION = "pooled_birth_weighted_prevalence"
EXPECTED_FIT_YEAR_RANGE = (2016, 2024)
EXPECTED_FALSE_POSITIVE_RATE = 0.000078
EXPECTED_REDUCTION_ERROR_CORRELATION = 0.0
EXPECTED_REDUCTION_CALIBRATION_SHIFT_LOGIT = 0.0
NAMED_RACE_INDICES = (0, 1, 2, 3, 4)
STABLE_RACE_INDICES = (0, 1, 3, 4)
AIAN_RACE_IDX = 2
WHITE_RACE_IDX = 0
SOURCE_RACE_CODE_LABELS = {
    1: "nhw",
    2: "nhb",
    3: "ai/an",
    4: "as/pi",
    5: "his",
}
HEALTH_SUMMARY_VARIABLES = (
    "rho_year",
    "eta_year",
    "recording_s",
    "recording_s_year",
    "true_count_year",
    "true_count_total",
)

TV_MATERIAL = 0.05
WRMS_MATERIAL = float(np.log(1.10))
CONTRAST_RELATIVE_MATERIAL = 0.15
DENOMINATOR_GROUP_RELATIVE_MATERIAL = 0.05
DENOMINATOR_NAMED_TOTAL_RELATIVE_MATERIAL = 0.01
UNCERTAINTY_REFERENCE_PROB = 0.89

# These are assumption scenarios, not source-provided standard errors.
UNCERTAINTY_SCENARIOS: dict[str, dict[str, float]] = {
    "narrow": {"shared_cv": 0.05, "stable_race_cv": 0.05, "aian_cv": 0.25},
    "moderate": {"shared_cv": 0.10, "stable_race_cv": 0.10, "aian_cv": 0.35},
    "wide": {"shared_cv": 0.20, "stable_race_cv": 0.15, "aian_cv": 0.50},
}

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


@dataclass(frozen=True)
class FitAuditInput:
    """Validated saved-fit inputs needed by the read-only audit."""

    run_dir: Path
    config: dict[str, Any]
    run_config: dict[str, Any]
    cells: pd.DataFrame
    idata: Any
    year_range: tuple[int, int]
    interval_prob: float
    fit_health: dict[str, Any]


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(_require_file(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with _require_file(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_provenance() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            completed = subprocess.run(
                ["git", "-C", str(REPO_ROOT), *args],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError("git provenance could not be resolved") from exc
        return completed.stdout.strip()

    status = run("status", "--porcelain=v1", "--untracked-files=all")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "worktree_dirty": bool(status),
        "worktree_status": status.splitlines(),
    }


def _package_versions() -> dict[str, str]:
    distributions = (
        "arviz",
        "dse-research-utils",
        "dspopulations-us-birth-certificates",
        "duckdb",
        "numpy",
        "pandas",
        "scipy",
    )
    versions: dict[str, str] = {}
    for distribution in distributions:
        try:
            versions[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"required distribution metadata is unavailable: {distribution}"
            ) from exc
    return versions


def _frame_digest(frame: pd.DataFrame, columns: list[str]) -> str:
    ordered = frame.loc[:, columns].sort_values(columns, ignore_index=True)
    values = pd.util.hash_pandas_object(ordered, index=False).to_numpy(dtype="uint64")
    return hashlib.sha256(values.tobytes()).hexdigest()


def _parse_years(value: str) -> tuple[int, ...]:
    try:
        years = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "years must be comma-separated integers"
        ) from exc
    if not years or len(set(years)) != len(years):
        raise argparse.ArgumentTypeError("years must be non-empty and unique")
    return years


def _stack_posterior(data: Any, *tail_dims: str) -> np.ndarray:
    return np.asarray(
        data.stack(sample=("chain", "draw")).transpose("sample", *tail_dims).values,
        dtype=float,
    )


def _validate_interval_prob(value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("interval_prob must be finite and lie in (0, 1)")
    return value


def _validate_fit_config(config: dict[str, Any]) -> tuple[int, int]:
    required = {
        "model_id": "DSP004",
        "age_model": "single_year",
        "theta_model": "morris_double_logistic_by_age_code",
        "recording_model": "constant",
        "reduction_model": "year",
        "recorded_definition": "confirmed_or_pending",
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(
                f"race-surveillance audit requires config.{key}={expected!r}; "
                f"found {config.get(key)!r}"
            )
    value = config.get("year_range")
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError("config.year_range must contain [start, end]")
    start, end = int(value[0]), int(value[1])
    if start > end:
        raise ValueError("config.year_range is reversed")
    if (start, end) != EXPECTED_FIT_YEAR_RANGE:
        raise ValueError(
            "race-surveillance audit is frozen to the 2016-2024 DSP004 "
            f"reference; found year_range={(start, end)!r}"
        )
    expected_endpoint = {
        "12": "10-12; Morris evaluated at age 12",
        "50": "50+; Morris evaluated at age 50",
    }
    endpoint = config.get("age_endpoint_convention")
    if endpoint != expected_endpoint:
        raise ValueError(
            "config.age_endpoint_convention does not match the frozen exact-age "
            f"contract; expected {expected_endpoint!r}, found {endpoint!r}"
        )
    priors = config.get("priors")
    if not isinstance(priors, dict):
        raise ValueError("config.priors must be an object")
    if priors.get("theta_lb_age_used") is not False:
        raise ValueError(
            "config.priors.theta_lb_age_used must be false; the audit uses the "
            "exact-age theta_lb_age stored in idata.constant_data"
        )
    frozen_priors = {
        "false_positive_rate": EXPECTED_FALSE_POSITIVE_RATE,
        "reduction_error_correlation": EXPECTED_REDUCTION_ERROR_CORRELATION,
        "reduction_calibration_shift_logit": (
            EXPECTED_REDUCTION_CALIBRATION_SHIFT_LOGIT
        ),
    }
    for key, expected in frozen_priors.items():
        try:
            observed = float(priors[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"config.priors.{key} must be numeric") from exc
        if not np.isfinite(observed) or not np.isclose(
            observed, expected, rtol=0.0, atol=1e-15
        ):
            raise ValueError(
                "race-surveillance audit requires the frozen reference "
                f"config.priors.{key}={expected!r}; found {observed!r}"
            )
    return start, end


def _fit_health(run_dir: Path, idata: Any) -> dict[str, Any]:
    summary_path = _require_file(run_dir / "summary.csv")
    summary = pd.read_csv(summary_path, index_col=0)
    required = {"r_hat", "ess_bulk", "ess_tail"}
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(
            f"{summary_path} is missing unrounded health columns: {sorted(missing)}"
        )
    cached_values = summary.loc[:, sorted(required)].apply(
        pd.to_numeric, errors="raise"
    )
    if not np.isfinite(cached_values.to_numpy()).all():
        raise ValueError("summary health columns must be finite")
    missing_posterior = set(HEALTH_SUMMARY_VARIABLES).difference(
        idata.posterior.data_vars
    )
    if missing_posterior:
        raise ValueError(
            f"idata posterior is missing health variables: {sorted(missing_posterior)}"
        )
    recomputed = az.summary(
        idata,
        var_names=list(HEALTH_SUMMARY_VARIABLES),
        ci_prob=DEFAULT_ETI_PROB,
        ci_kind="hdi",
        round_to="none",
    )
    if set(summary.index.astype(str)) != set(recomputed.index.astype(str)):
        raise ValueError(
            "cached summary rows do not match the frozen DSP004 health summary"
        )
    recomputed = recomputed.loc[summary.index]
    compare_columns = ("mean", "ess_bulk", "ess_tail", "r_hat")
    if any(column not in summary for column in compare_columns):
        raise ValueError(
            "cached summary must contain mean, ess_bulk, ess_tail, and r_hat"
        )
    for column in compare_columns:
        observed = pd.to_numeric(summary[column], errors="raise").to_numpy(dtype=float)
        expected = pd.to_numeric(recomputed[column], errors="raise").to_numpy(
            dtype=float
        )
        if not np.allclose(observed, expected, rtol=1e-12, atol=1e-12):
            raise ValueError(f"cached summary column {column!r} does not match idata")
    if not hasattr(idata, "sample_stats") or "diverging" not in idata.sample_stats:
        raise ValueError("idata.sample_stats.diverging is required")
    divergences = int(np.asarray(idata.sample_stats["diverging"].values).sum())
    max_rhat = float(recomputed["r_hat"].max())
    min_ess_bulk = float(recomputed["ess_bulk"].min())
    min_ess_tail = float(recomputed["ess_tail"].min())
    healthy = (
        divergences == 0
        and max_rhat < 1.01
        and min_ess_bulk >= 400.0
        and min_ess_tail >= 400.0
    )
    return {
        "divergences": divergences,
        "max_unrounded_rhat": max_rhat,
        "min_bulk_ess": min_ess_bulk,
        "min_tail_ess": min_ess_tail,
        "cached_summary_matches_idata": True,
        "health_gate_passed": healthy,
        "health_gate_definition": (
            "zero divergences; max unrounded Rhat < 1.01; min bulk/tail ESS >= 400"
        ),
    }


def load_fit_input(
    run_dir: Path | str,
    *,
    interval_prob: float = DEFAULT_ETI_PROB,
) -> FitAuditInput:
    """Load and validate the exact saved DSP004 fit contract."""
    run_dir = Path(run_dir)
    config_path = _require_file(run_dir / "config.json")
    run_config_path = _require_file(run_dir / "run_config.json")
    cells_path = _require_file(run_dir / "cells.parquet")
    idata_path = _require_file(run_dir / "idata.nc")
    config = _read_json(config_path)
    run_config = _read_json(run_config_path)
    if run_config.get("name") != "reporting":
        raise ValueError(
            "race-surveillance audit requires a reporting-profile fit; "
            f"found run_config.name={run_config.get('name')!r}"
        )
    year_range = _validate_fit_config(config)
    interval_prob = _validate_interval_prob(interval_prob)

    cells = pd.read_parquet(cells_path).sort_values(
        ["year_idx", "age_idx"], ignore_index=True
    )
    required_cells = {
        "year_idx",
        "age_idx",
        "maternal_age",
        "N_cell",
        "R_cell",
    }
    missing_cells = required_cells.difference(cells.columns)
    if missing_cells:
        raise ValueError(f"saved cells are missing: {sorted(missing_cells)}")
    if cells.duplicated(["year_idx", "age_idx"]).any():
        raise ValueError("saved DSP004 cells must be unique by year_idx and age_idx")

    idata = az.from_netcdf(idata_path)
    if not hasattr(idata, "posterior"):
        raise ValueError("idata posterior group is required")
    required_posterior = {
        "eta_year",
        "rho_year",
        "recording_s",
        "p_ds_lb",
        "true_count_year",
        "true_count_total",
    }
    missing_posterior = required_posterior.difference(idata.posterior.data_vars)
    if missing_posterior:
        raise ValueError(
            f"idata posterior is missing variables: {sorted(missing_posterior)}"
        )
    if not hasattr(idata, "constant_data") or "theta_lb_age" not in idata.constant_data:
        raise ValueError(
            "idata.constant_data.theta_lb_age is required for exact-age audit"
        )
    n_year = year_range[1] - year_range[0] + 1
    if idata.posterior["eta_year"].sizes.get("year") != n_year:
        raise ValueError("eta_year does not have one value per configured year")
    if idata.posterior["p_ds_lb"].sizes.get("cell") != len(cells):
        raise ValueError("p_ds_lb cell dimension does not match cells.parquet")

    health = _fit_health(run_dir, idata)
    if not health["health_gate_passed"]:
        raise ValueError(f"saved fit failed the scientific health gate: {health}")
    return FitAuditInput(
        run_dir=run_dir,
        config=config,
        run_config=run_config,
        cells=cells,
        idata=idata,
        year_range=year_range,
        interval_prob=interval_prob,
        fit_health=health,
    )


def load_race_age_year_cells(
    db_path: Path | str,
    *,
    year_range: tuple[int, int],
    table: str = "us_births",
) -> pd.DataFrame:
    """Reconstruct the DSP004 cohort by year, exact age, and seven-race index."""
    if not _IDENTIFIER.fullmatch(table):
        raise ValueError(f"invalid DuckDB table identifier: {table!r}")
    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    start, end = year_range
    race_case = case_from_map("mracehisp_c", RACE_MAP, default=RACE_UNKNOWN_IDX)
    age_case = (
        "CASE WHEN CAST(mage_c AS INTEGER) <= 12 THEN 12 "
        "WHEN CAST(mage_c AS INTEGER) >= 50 THEN 50 "
        "ELSE CAST(mage_c AS INTEGER) END"
    )
    con = duckdb.connect(str(path), read_only=True)
    try:
        frame = con.execute(
            f"""
            SELECT
                CAST(year AS INTEGER) AS year,
                CAST(year AS INTEGER) - ? AS year_idx,
                {age_case} AS maternal_age,
                {race_case} AS race_idx,
                COUNT(*) AS N_cell,
                SUM(CAST(down_ind AS INTEGER)) AS R_cell
            FROM {table}
            WHERE year BETWEEN ? AND ?
              AND mage_c IS NOT NULL
              AND down_ind IS NOT NULL
            GROUP BY 1, 2, 3, 4
            ORDER BY 1, 3, 4
            """,
            [start, start, end],
        ).df()
    finally:
        con.close()
    if frame.empty:
        raise ValueError(f"no DSP004-eligible rows found for {start}-{end}")
    for column in ("year", "year_idx", "maternal_age", "race_idx", "N_cell", "R_cell"):
        frame[column] = frame[column].astype("int64")
    if not frame["race_idx"].between(0, len(RACE_LEVELS) - 1).all():
        raise ValueError("reconstructed race_idx is outside the seven-race vocabulary")
    frame["race"] = frame["race_idx"].map(dict(enumerate(RACE_LEVELS)))
    return frame


def reconcile_saved_cells(
    fit: FitAuditInput,
    race_cells: pd.DataFrame,
) -> pd.DataFrame:
    """Require race collapse to reproduce every saved age-year N/R cell."""
    collapsed = (
        race_cells.groupby(["year_idx", "maternal_age"], as_index=False, observed=True)[
            ["N_cell", "R_cell"]
        ]
        .sum()
        .sort_values(["year_idx", "maternal_age"], ignore_index=True)
    )
    saved = (
        fit.cells[["year_idx", "maternal_age", "N_cell", "R_cell"]]
        .sort_values(["year_idx", "maternal_age"], ignore_index=True)
        .copy()
    )
    merged = saved.merge(
        collapsed,
        on=["year_idx", "maternal_age"],
        how="outer",
        suffixes=("_saved", "_race"),
        indicator=True,
        validate="1:1",
    )
    for metric in ("N_cell", "R_cell"):
        merged[f"{metric}_difference"] = (
            merged[f"{metric}_race"] - merged[f"{metric}_saved"]
        )
    exact = (
        (merged["_merge"] == "both").all()
        and (merged["N_cell_difference"] == 0).all()
        and (merged["R_cell_difference"] == 0).all()
    )
    if not exact:
        raise ValueError(
            "race-stratified DuckDB cohort does not reproduce saved DSP004 cells"
        )
    return merged.drop(columns="_merge")


def load_surveillance(
    path: Path | str,
    *,
    years: tuple[int, ...] = DEFAULT_SURVEILLANCE_YEARS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the corrected source and select raw, finite surveillance rows only."""
    path = Path(path)
    source = pd.read_csv(_require_file(path))
    required = {
        "year",
        "race",
        "mracehisp_c",
        "recorded_bc",
        "births_bc",
        "bc_prev_per10k",
        "recording_frac_g",
        "est_true_count",
        "est_true_prev_per10k",
        "surveillance_prev_per10k",
    }
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"surveillance source is missing: {sorted(missing)}")
    numeric = sorted(required.difference({"race"}))
    for column in numeric:
        source[column] = pd.to_numeric(source[column], errors="raise")
    nonnullable_numeric = set(numeric).difference({"surveillance_prev_per10k"})
    if not np.isfinite(source[list(nonnullable_numeric)].to_numpy(dtype=float)).all():
        raise ValueError(
            "source numeric fields other than raw surveillance prevalence must "
            "be finite"
        )
    for column in ("year", "mracehisp_c", "recorded_bc", "births_bc"):
        values = source[column].to_numpy(dtype=float)
        if not np.equal(values, np.floor(values)).all():
            raise ValueError(f"source {column} values must be integers")
        source[column] = values.astype("int64")
    if source.duplicated(["year", "mracehisp_c"]).any():
        raise ValueError(
            "surveillance source keys must be unique by year and race code"
        )
    if set(source["mracehisp_c"].unique()) != set(SOURCE_RACE_CODE_LABELS):
        raise ValueError("surveillance source must contain exactly race codes 1-5")
    source["race"] = source["race"].astype(str).str.strip().str.lower()
    observed_pairs = set(
        source[["mracehisp_c", "race"]].itertuples(index=False, name=None)
    )
    expected_pairs = set(SOURCE_RACE_CODE_LABELS.items())
    if observed_pairs != expected_pairs:
        raise ValueError(
            "surveillance source race labels do not match the frozen code mapping; "
            f"expected {sorted(expected_pairs)!r}, found {sorted(observed_pairs)!r}"
        )
    if (source["births_bc"] <= 0).any() or (source["recorded_bc"] < 0).any():
        raise ValueError(
            "source birth counts must be positive and recorded counts non-negative"
        )
    if not source["recording_frac_g"].between(0.0, 1.0, inclusive="neither").all():
        raise ValueError("source recording fractions must lie in (0, 1)")

    expected_bc_prev = 1e4 * source["recorded_bc"] / source["births_bc"]
    expected_true_count = source["recorded_bc"] / source["recording_frac_g"]
    expected_true_prev = 1e4 * expected_true_count / source["births_bc"]
    for observed, expected, label in (
        (source["bc_prev_per10k"], expected_bc_prev, "bc_prev_per10k"),
        (source["est_true_count"], expected_true_count, "est_true_count"),
        (
            source["est_true_prev_per10k"],
            expected_true_prev,
            "est_true_prev_per10k",
        ),
    ):
        if not np.allclose(observed, expected, rtol=1e-9, atol=1e-9):
            raise ValueError(f"source algebra failed for {label}")

    requested = source[source["year"].isin(years)].copy()
    if len(requested) != len(years) * len(NAMED_RACE_INDICES):
        raise ValueError("requested years must each contain exactly five source rows")
    if requested["surveillance_prev_per10k"].isna().any():
        raise ValueError(
            "requested primary years must have raw surveillance_prev_per10k; "
            "filled est_true values are not an allowed fallback"
        )
    finite = np.isfinite(requested["surveillance_prev_per10k"].to_numpy(dtype=float))
    if not finite.all() or (requested["surveillance_prev_per10k"] <= 0.0).any():
        raise ValueError("raw surveillance prevalence must be finite and positive")
    requested["race_idx"] = requested["mracehisp_c"].map(RACE_MAP)
    if requested["race_idx"].isna().any():
        raise ValueError("raw surveillance race code does not map to a named race")
    requested["race_idx"] = requested["race_idx"].astype(int)
    requested["race_model_label"] = requested["race_idx"].map(
        dict(enumerate(RACE_LEVELS))
    )
    pooled_rows: list[dict[str, Any]] = []
    for label_year, race_code in requested[["year", "mracehisp_c"]].itertuples(
        index=False, name=None
    ):
        window_start = int(label_year) - CENTERED_WINDOW_HALF_WIDTH
        window_end = int(label_year) + CENTERED_WINDOW_HALF_WIDTH
        window = source[
            (source["mracehisp_c"] == int(race_code))
            & source["year"].between(window_start, window_end)
        ]
        expected_years = list(range(window_start, window_end + 1))
        if sorted(window["year"].astype(int).tolist()) != expected_years:
            raise ValueError(
                "source birth denominators must cover every centred window year"
            )
        births = int(window["births_bc"].sum())
        recorded = int(window["recorded_bc"].sum())
        pooled_rows.append(
            {
                "year": int(label_year),
                "mracehisp_c": int(race_code),
                "source_window_years": _year_list(expected_years),
                "source_window_births_bc": births,
                "source_window_recorded_bc": recorded,
                "source_window_recorded_prev_per10k": 1e4 * recorded / births,
            }
        )
    requested = requested.merge(
        pd.DataFrame(pooled_rows),
        on=["year", "mracehisp_c"],
        how="left",
        validate="1:1",
    )
    requested["source_native_implied_true_count"] = (
        requested["source_window_births_bc"]
        * requested["surveillance_prev_per10k"]
        / 1e4
    )
    requested["source_value_status"] = (
        "direct_raw_surveillance_centred_five_year_pooled_count_ratio_maternal_race"
    )
    requested["source_aggregation_operator"] = SOURCE_AGGREGATION_OPERATOR
    requested["source_aggregation_operator_status"] = "confirmed"
    requested["filled_estimate_used"] = False
    requested = requested.sort_values(["year", "race_idx"], ignore_index=True)

    inventory = (
        source.assign(
            raw_surveillance_available=source["surveillance_prev_per10k"].notna(),
            in_requested_primary_year=source["year"].isin(years),
        )
        .groupby("year", as_index=False, observed=True)
        .agg(
            source_rows=("mracehisp_c", "size"),
            raw_surveillance_rows=("raw_surveillance_available", "sum"),
            requested_primary_year=("in_requested_primary_year", "max"),
        )
    )
    inventory["filled_values_eligible_for_primary"] = False
    return requested, inventory


def _theta_by_age(fit: FitAuditInput) -> pd.Series:
    theta_da = fit.idata.constant_data["theta_lb_age"]
    if "age" not in theta_da.dims or theta_da.ndim != 1:
        raise ValueError("constant_data.theta_lb_age must be one-dimensional over age")
    age_values = np.asarray(theta_da.coords["age"].values, dtype=int)
    theta = np.asarray(theta_da.values, dtype=float)
    if len(age_values) != len(theta) or len(set(age_values.tolist())) != len(
        age_values
    ):
        raise ValueError("theta age coordinate must be unique and match theta values")
    if not np.all(np.isfinite(theta)) or not np.all((theta > 0.0) & (theta < 1.0)):
        raise ValueError("theta values must be finite probabilities")
    saved_ages = set(fit.cells["maternal_age"].astype(int))
    if saved_ages != set(age_values.tolist()):
        raise ValueError("theta age coordinate does not match represented saved ages")
    return pd.Series(theta, index=age_values, name="theta_age")


def reconstruct_internal_accounting(
    fit: FitAuditInput,
    race_cells: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Reconstruct draw-wise seven-race true counts and verify fit identities."""
    theta_by_age = _theta_by_age(fit)
    cells = race_cells.copy()
    cells["theta_age"] = cells["maternal_age"].map(theta_by_age)
    if cells["theta_age"].isna().any():
        raise ValueError("a reconstructed maternal age has no fitted theta value")
    cells["natural_expected_ds"] = cells["N_cell"] * cells["theta_age"]
    grouped = (
        cells.groupby(["year", "year_idx", "race_idx", "race"], as_index=False)[
            ["N_cell", "R_cell", "natural_expected_ds"]
        ]
        .sum()
        .sort_values(["year", "race_idx"], ignore_index=True)
    )

    eta = _stack_posterior(fit.idata.posterior["eta_year"], "year")
    rho = _stack_posterior(fit.idata.posterior["rho_year"], "year")
    if eta.shape != rho.shape or not np.allclose(eta, 1.0 - rho, rtol=0.0, atol=1e-12):
        raise ValueError("posterior eta_year must equal 1 - rho_year draw by draw")
    if not np.all((eta > 0.0) & (eta < 1.0)):
        raise ValueError("posterior eta_year must lie in (0, 1)")

    saved_theta = fit.cells["maternal_age"].astype(int).map(theta_by_age).to_numpy()
    saved_year_idx = fit.cells["year_idx"].to_numpy(dtype=int)
    expected_p_ds = saved_theta[None, :] * eta[:, saved_year_idx]
    stored_p_ds = _stack_posterior(fit.idata.posterior["p_ds_lb"], "cell")
    p_ds_max_error = float(np.max(np.abs(expected_p_ds - stored_p_ds)))
    if p_ds_max_error > 1e-12:
        raise ValueError(f"stored p_ds_lb identity failed; max error={p_ds_max_error}")

    natural = grouped["natural_expected_ds"].to_numpy(dtype=float)
    year_idx = grouped["year_idx"].to_numpy(dtype=int)
    true_draws = eta[:, year_idx] * natural[None, :]
    grouped_n = grouped["N_cell"].to_numpy(dtype=float)
    prevalence_draws = 1e4 * true_draws / grouped_n[None, :]

    stored_year = _stack_posterior(fit.idata.posterior["true_count_year"], "year")
    reconstructed_year = np.zeros_like(stored_year)
    for y in range(stored_year.shape[1]):
        reconstructed_year[:, y] = true_draws[:, year_idx == y].sum(axis=1)
    year_identity_error = float(np.max(np.abs(reconstructed_year - stored_year)))
    if year_identity_error > 1e-8:
        raise ValueError(
            f"race accounting does not reproduce true_count_year; error={year_identity_error}"
        )
    stored_total = np.asarray(
        fit.idata.posterior["true_count_total"].stack(sample=("chain", "draw")).values,
        dtype=float,
    )
    reconstructed_total = true_draws.sum(axis=1)
    total_identity_error = float(np.max(np.abs(reconstructed_total - stored_total)))
    if total_identity_error > 1e-8:
        raise ValueError(
            f"race accounting does not reproduce true_count_total; error={total_identity_error}"
        )

    tail = (1.0 - fit.interval_prob) / 2.0
    for prefix, draws in (
        ("model_true_count", true_draws),
        ("model_prevalence_per10k", prevalence_draws),
    ):
        grouped[f"{prefix}_mean"] = draws.mean(axis=0)
        grouped[f"{prefix}_lo"] = np.quantile(draws, tail, axis=0)
        grouped[f"{prefix}_hi"] = np.quantile(draws, 1.0 - tail, axis=0)

    grouped["named_surveillance_supported"] = grouped["race_idx"].isin(
        NAMED_RACE_INDICES
    )
    grouped["external_target_status"] = np.where(
        grouped["named_surveillance_supported"],
        "eligible_in_direct_source_years_only",
        "unanchored_no_surveillance_category",
    )
    grouped["model_named_case_share"] = np.nan
    for year, indices in grouped.groupby("year", observed=True).groups.items():
        del year
        idx = np.asarray(list(indices), dtype=int)
        named = idx[
            grouped.loc[idx, "named_surveillance_supported"].to_numpy(dtype=bool)
        ]
        denominator = grouped.loc[named, "natural_expected_ds"].sum()
        grouped.loc[named, "model_named_case_share"] = (
            grouped.loc[named, "natural_expected_ds"] / denominator
        )

    identities = {
        "p_ds_lb_max_absolute_error": p_ds_max_error,
        "true_count_year_max_absolute_error": year_identity_error,
        "true_count_total_max_absolute_error": total_identity_error,
    }
    return grouped, identities


def _year_list(values: list[int] | tuple[int, ...] | np.ndarray) -> str:
    """Return a stable, CSV-friendly representation of integer years."""
    return ",".join(str(int(value)) for value in values)


def centered_window_support_table(
    internal: pd.DataFrame,
    surveillance: pd.DataFrame,
) -> pd.DataFrame:
    """Audit model support for each centred five-year source point and race."""
    if internal.duplicated(["year", "race_idx"]).any():
        raise ValueError("internal accounting must be unique by year and race")
    rows: list[dict[str, Any]] = []
    for source in surveillance.sort_values(["year", "race_idx"]).itertuples(
        index=False
    ):
        label_year = int(source.year)
        race_idx = int(source.race_idx)
        expected = list(
            range(
                label_year - CENTERED_WINDOW_HALF_WIDTH,
                label_year + CENTERED_WINDOW_HALF_WIDTH + 1,
            )
        )
        available = sorted(
            int(value)
            for value in internal.loc[internal["race_idx"] == race_idx, "year"].unique()
            if int(value) in expected
        )
        missing = sorted(set(expected).difference(available))
        complete = not missing
        if complete:
            status = "complete_five_year_window"
        elif available and min(available) > min(expected):
            status = "partial_left_truncated"
        else:
            status = "partial_or_internally_missing"
        rows.append(
            {
                "label_year": label_year,
                "race_idx": race_idx,
                "race_model_label": RACE_LEVELS[race_idx],
                "window_start_year": expected[0],
                "window_end_year": expected[-1],
                "window_width_years": len(expected),
                "expected_fit_years": _year_list(expected),
                "supported_fit_years": _year_list(available),
                "missing_fit_years": _year_list(missing),
                "supported_year_count": len(available),
                "coverage_fraction": len(available) / len(expected),
                "window_complete": complete,
                "window_support_status": status,
                "source_aligned_evaluable": complete,
                "calibration_eligible": False,
                "source_window_alignment": "centred",
                "source_label_role": "centre_year",
                "source_race_basis": "maternal",
                "source_aggregation_operator": SOURCE_AGGREGATION_OPERATOR,
                "source_aggregation_operator_status": "confirmed",
            }
        )
    return pd.DataFrame(rows).sort_values(["label_year", "race_idx"], ignore_index=True)


def pooled_window_denominator_reconciliation(
    internal: pd.DataFrame,
    surveillance: pd.DataFrame,
) -> pd.DataFrame:
    """Compare source and model birth denominators over each centred window."""
    rows: list[dict[str, Any]] = []
    for source in surveillance.sort_values(["year", "race_idx"]).itertuples(
        index=False
    ):
        label_year = int(source.year)
        race_idx = int(source.race_idx)
        expected_years = list(
            range(
                label_year - CENTERED_WINDOW_HALF_WIDTH,
                label_year + CENTERED_WINDOW_HALF_WIDTH + 1,
            )
        )
        model = internal[
            (internal["race_idx"] == race_idx) & internal["year"].isin(expected_years)
        ]
        supported_years = sorted(model["year"].astype(int).unique().tolist())
        complete = supported_years == expected_years
        model_births = float(model["N_cell"].sum()) if complete else np.nan
        model_recorded = float(model["R_cell"].sum()) if complete else np.nan
        source_births = float(source.source_window_births_bc)
        source_recorded = float(source.source_window_recorded_bc)
        birth_difference = source_births - model_births if complete else np.nan
        relative_difference = birth_difference / model_births if complete else np.nan
        rows.append(
            {
                "label_year": label_year,
                "window_start_year": expected_years[0],
                "window_end_year": expected_years[-1],
                "race_idx": race_idx,
                "model_race_label": RACE_LEVELS[race_idx],
                "mracehisp_c": int(source.mracehisp_c),
                "source_window_births_bc": source_births,
                "source_window_recorded_bc": source_recorded,
                "source_native_implied_true_count": float(
                    source.source_native_implied_true_count
                ),
                "model_window_births": model_births,
                "model_window_recorded": model_recorded,
                "birth_difference_source_minus_model": birth_difference,
                "birth_relative_difference_source_vs_model": relative_difference,
                "recorded_difference_source_minus_model": (
                    source_recorded - model_recorded if complete else np.nan
                ),
                "window_denominator_evaluable": complete,
                "group_denominator_material": bool(
                    complete
                    and abs(relative_difference) >= DENOMINATOR_GROUP_RELATIVE_MATERIAL
                ),
            }
        )
    denominator = pd.DataFrame(rows)
    denominator["source_named_births"] = np.nan
    denominator["model_named_births"] = np.nan
    denominator["named_birth_relative_difference_source_vs_model"] = np.nan
    denominator["named_total_denominator_material"] = False
    for _label_year, indices in denominator.groupby(
        "label_year", observed=True
    ).groups.items():
        idx = list(indices)
        frame = denominator.loc[idx]
        if not frame["window_denominator_evaluable"].all():
            continue
        source_named = float(frame["source_window_births_bc"].sum())
        model_named = float(frame["model_window_births"].sum())
        relative = (source_named - model_named) / model_named
        denominator.loc[idx, "source_named_births"] = source_named
        denominator.loc[idx, "model_named_births"] = model_named
        denominator.loc[idx, "named_birth_relative_difference_source_vs_model"] = (
            relative
        )
        denominator.loc[idx, "named_total_denominator_material"] = bool(
            abs(relative) >= DENOMINATOR_NAMED_TOTAL_RELATIVE_MATERIAL
        )
    denominator["diagnostic_role"] = (
        "pooled_window_denominator_mapping_not_calibration_ready"
    )
    denominator["calibration_eligible"] = False
    return denominator.sort_values(["label_year", "race_idx"], ignore_index=True)


def _pooled_window_model_draws(
    fit: FitAuditInput,
    internal: pd.DataFrame,
    *,
    label_year: int,
) -> dict[str, np.ndarray]:
    """Construct the draw-wise pooled-count model analogue for one window."""
    years = np.arange(
        label_year - CENTERED_WINDOW_HALF_WIDTH,
        label_year + CENTERED_WINDOW_HALF_WIDTH + 1,
        dtype=int,
    )
    frame = internal[internal["year"].isin(years)].copy()
    named = frame[frame["race_idx"].isin(NAMED_RACE_INDICES)]
    expected_named = len(years) * len(NAMED_RACE_INDICES)
    if len(named) != expected_named or named.duplicated(["year", "race_idx"]).any():
        raise ValueError(
            f"centred window {label_year} lacks complete named race-year support"
        )

    n_race = len(RACE_LEVELS)
    births = np.zeros((n_race, len(years)), dtype=float)
    recorded = np.zeros_like(births)
    natural = np.zeros_like(births)
    for row in frame.itertuples(index=False):
        race_idx = int(row.race_idx)
        year_position = int(row.year) - int(years[0])
        births[race_idx, year_position] += float(row.N_cell)
        recorded[race_idx, year_position] += float(row.R_cell)
        natural[race_idx, year_position] += float(row.natural_expected_ds)
    if (births[list(NAMED_RACE_INDICES)] <= 0.0).any():
        raise ValueError("complete named race-year windows require positive births")

    eta = _stack_posterior(fit.idata.posterior["eta_year"], "year")
    fit_positions = years - fit.year_range[0]
    if (fit_positions < 0).any() or (fit_positions >= eta.shape[1]).any():
        raise ValueError("requested centred window is outside the saved eta_year draws")
    true_annual = natural[None, :, :] * eta[:, None, fit_positions]
    window_births = births.sum(axis=1)
    window_recorded = recorded.sum(axis=1)
    count = true_annual.sum(axis=2)
    prevalence = np.divide(
        1e4 * count,
        window_births[None, :],
        out=np.zeros_like(count),
        where=window_births[None, :] > 0.0,
    )

    named_count = count[:, list(NAMED_RACE_INDICES)]
    named_share = named_count / named_count.sum(axis=1, keepdims=True)
    return {
        "years": years,
        "births": window_births,
        "recorded": window_recorded,
        "prevalence": prevalence,
        "count": count,
        "named_share": named_share,
    }


def build_centered_window_comparison(
    fit: FitAuditInput,
    internal: pd.DataFrame,
    surveillance: pd.DataFrame,
    support: pd.DataFrame,
) -> pd.DataFrame:
    """Compare the confirmed pooled source ratio with its model analogue."""
    rows: list[dict[str, Any]] = []
    tail = (1.0 - fit.interval_prob) / 2.0
    false_positive_rate = float(fit.config["priors"]["false_positive_rate"])
    for label_year, source_window in surveillance.groupby("year", observed=True):
        label_year = int(label_year)
        source_window = source_window.sort_values("race_idx")
        support_window = support[support["label_year"] == label_year]
        window_complete = bool(
            len(support_window) == len(NAMED_RACE_INDICES)
            and support_window["window_complete"].all()
        )
        window_start = label_year - CENTERED_WINDOW_HALF_WIDTH
        window_end = label_year + CENTERED_WINDOW_HALF_WIDTH
        bundle = (
            _pooled_window_model_draws(
                fit,
                internal,
                label_year=label_year,
            )
            if window_complete
            else None
        )
        source_prevalence = source_window["surveillance_prev_per10k"].to_numpy(
            dtype=float
        )
        source_native_count = source_window[
            "source_native_implied_true_count"
        ].to_numpy(dtype=float)
        source_native_share = source_native_count / source_native_count.sum()
        if bundle is not None:
            model_prevalence = bundle["prevalence"]
            model_count = bundle["count"]
            model_share = bundle["named_share"]
            named_model_births = bundle["births"][list(NAMED_RACE_INDICES)]
            source_standardised_count = source_prevalence * named_model_births / 1e4
            source_standardised_share = (
                source_standardised_count / source_standardised_count.sum()
            )
            white_model_log_rate = np.log(model_prevalence[:, WHITE_RACE_IDX])
            white_source_log_rate = np.log(source_prevalence[WHITE_RACE_IDX])
            model_all_mean = float(model_count.sum(axis=1).mean())
            model_unsupported_mean = float(model_count[:, [5, 6]].sum(axis=1).mean())
        for source in source_window.itertuples(index=False):
            race_idx = int(source.race_idx)
            source_true_count = float(source.source_native_implied_true_count)
            source_recording_numerator = (
                float(source.source_window_recorded_bc)
                - (float(source.source_window_births_bc) - source_true_count)
                * false_positive_rate
            )
            row: dict[str, Any] = {
                "label_year": label_year,
                "window_start_year": window_start,
                "window_end_year": window_end,
                "window_width_years": 5,
                "model_aggregation_construction": (MODEL_AGGREGATION_CONSTRUCTION),
                "source_aggregation_operator": SOURCE_AGGREGATION_OPERATOR,
                "source_aggregation_operator_status": "confirmed",
                "race_idx": race_idx,
                "race_model_label": RACE_LEVELS[race_idx],
                "source_race_basis": "maternal",
                "surveillance_prev_per10k": float(source.surveillance_prev_per10k),
                "source_window_births_bc": float(source.source_window_births_bc),
                "source_window_recorded_bc": float(source.source_window_recorded_bc),
                "source_native_implied_true_count": source_true_count,
                "source_native_named_case_share": float(source_native_share[race_idx]),
                "source_native_implied_recording_s_approx": (
                    source_recording_numerator / source_true_count
                ),
                "source_native_implied_recording_s_valid_for_calibration": False,
                "window_complete": window_complete,
                "source_aligned_evaluable": window_complete,
                "calibration_eligible": False,
                "filled_estimate_used": False,
                "model_window_births": np.nan,
                "model_window_recorded": np.nan,
                "model_prevalence_per10k_mean": np.nan,
                "model_prevalence_per10k_lo": np.nan,
                "model_prevalence_per10k_hi": np.nan,
                "model_true_count_mean": np.nan,
                "model_true_count_lo": np.nan,
                "model_true_count_hi": np.nan,
                "model_named_case_share_mean": np.nan,
                "model_named_case_share_lo": np.nan,
                "model_named_case_share_hi": np.nan,
                "model_all_race_true_count_mean": np.nan,
                "model_unsupported_true_count_mean": np.nan,
                "source_standardised_count_on_model_window": np.nan,
                "source_standardised_named_case_share": np.nan,
                "standardised_share_difference_percentage_points": np.nan,
                "native_share_difference_percentage_points": np.nan,
                "surveillance_to_model_prevalence_ratio": np.nan,
                "source_standardised_to_model_case_share_ratio": np.nan,
                "source_native_to_model_case_share_ratio": np.nan,
                "source_minus_model_relative_log_rate_vs_white_mean": np.nan,
                "source_minus_model_relative_log_rate_vs_white_lo": np.nan,
                "source_minus_model_relative_log_rate_vs_white_hi": np.nan,
                "source_vs_model_relative_rate_ratio_to_white_mean": np.nan,
                "model_window_implied_recording_s_approx": np.nan,
                "model_window_implied_recording_s_valid_for_calibration": False,
            }
            if bundle is None:
                row["comparison_status"] = (
                    "not_estimable_missing_2014_2015_eta_draws"
                    if label_year == 2016
                    else "not_estimable_incomplete_named_race_window"
                )
                row["composition_interpretation"] = (
                    "source_native_pooled_count_only_model_not_estimable"
                )
            else:
                prevalence_draw = model_prevalence[:, race_idx]
                count_draw = model_count[:, race_idx]
                share_draw = model_share[:, race_idx]
                relative_residual = (
                    np.log(float(source.surveillance_prev_per10k))
                    - white_source_log_rate
                    - (np.log(prevalence_draw) - white_model_log_rate)
                )
                model_share_mean = float(share_draw.mean())
                standardised_share = float(source_standardised_share[race_idx])
                native_share = float(source_native_share[race_idx])
                standardised_true_count = float(source_standardised_count[race_idx])
                model_recording_numerator = (
                    float(bundle["recorded"][race_idx])
                    - (float(bundle["births"][race_idx]) - standardised_true_count)
                    * false_positive_rate
                )
                row.update(
                    {
                        "model_window_births": float(bundle["births"][race_idx]),
                        "model_window_recorded": float(bundle["recorded"][race_idx]),
                        "model_prevalence_per10k_mean": float(prevalence_draw.mean()),
                        "model_prevalence_per10k_lo": float(
                            np.quantile(prevalence_draw, tail)
                        ),
                        "model_prevalence_per10k_hi": float(
                            np.quantile(prevalence_draw, 1.0 - tail)
                        ),
                        "model_true_count_mean": float(count_draw.mean()),
                        "model_true_count_lo": float(np.quantile(count_draw, tail)),
                        "model_true_count_hi": float(
                            np.quantile(count_draw, 1.0 - tail)
                        ),
                        "model_named_case_share_mean": model_share_mean,
                        "model_named_case_share_lo": float(
                            np.quantile(share_draw, tail)
                        ),
                        "model_named_case_share_hi": float(
                            np.quantile(share_draw, 1.0 - tail)
                        ),
                        "model_all_race_true_count_mean": model_all_mean,
                        "model_unsupported_true_count_mean": (model_unsupported_mean),
                        "source_standardised_count_on_model_window": (
                            standardised_true_count
                        ),
                        "source_standardised_named_case_share": (standardised_share),
                        "standardised_share_difference_percentage_points": (
                            100.0 * (standardised_share - model_share_mean)
                        ),
                        "native_share_difference_percentage_points": (
                            100.0 * (native_share - model_share_mean)
                        ),
                        "surveillance_to_model_prevalence_ratio": float(
                            source.surveillance_prev_per10k / prevalence_draw.mean()
                        ),
                        "source_standardised_to_model_case_share_ratio": (
                            standardised_share / model_share_mean
                        ),
                        "source_native_to_model_case_share_ratio": (
                            native_share / model_share_mean
                        ),
                        "source_minus_model_relative_log_rate_vs_white_mean": (
                            float(relative_residual.mean())
                        ),
                        "source_minus_model_relative_log_rate_vs_white_lo": (
                            float(np.quantile(relative_residual, tail))
                        ),
                        "source_minus_model_relative_log_rate_vs_white_hi": (
                            float(np.quantile(relative_residual, 1.0 - tail))
                        ),
                        "source_vs_model_relative_rate_ratio_to_white_mean": (
                            float(np.exp(relative_residual.mean()))
                        ),
                        "model_window_implied_recording_s_approx": (
                            model_recording_numerator / standardised_true_count
                        ),
                        "comparison_status": (
                            "source_aligned_pooled_count_ratio_descriptive_only"
                        ),
                        "composition_interpretation": (
                            "pooled_case_allocation_reported_on_native_and_model_standardised_denominators"
                        ),
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["label_year", "race_idx"], ignore_index=True)


def centered_composition_summary_table(comparison: pd.DataFrame) -> pd.DataFrame:
    """Summarise the complete pooled window on native and standardised births."""
    eligible = comparison[comparison["source_aligned_evaluable"]].copy()
    rows: list[dict[str, Any]] = []
    for label_year, observed in eligible.groupby("label_year", observed=True):
        observed = observed.sort_values("race_idx")
        model_share = observed["model_named_case_share_mean"].to_numpy(dtype=float)
        standardised_share = observed["source_standardised_named_case_share"].to_numpy(
            dtype=float
        )
        native_share = observed["source_native_named_case_share"].to_numpy(dtype=float)
        standardised_log_ratio = np.log(standardised_share / model_share)
        native_log_ratio = np.log(native_share / model_share)
        standardised_tv = 0.5 * float(np.abs(standardised_share - model_share).sum())
        native_tv = 0.5 * float(np.abs(native_share - model_share).sum())
        standardised_wrms = float(
            np.sqrt(np.sum(standardised_share * np.square(standardised_log_ratio)))
        )
        native_wrms = float(np.sqrt(np.sum(native_share * np.square(native_log_ratio))))
        model_named = float(observed["model_true_count_mean"].sum())
        source_standardised_named = float(
            observed["source_standardised_count_on_model_window"].sum()
        )
        source_native_named = float(observed["source_native_implied_true_count"].sum())
        model_all = float(observed["model_all_race_true_count_mean"].iloc[0])
        model_unsupported = float(observed["model_unsupported_true_count_mean"].iloc[0])
        standardised_coherence = (
            source_standardised_named + model_unsupported
        ) / model_all - 1.0
        native_coherence = (source_native_named + model_unsupported) / model_all - 1.0
        rows.append(
            {
                "label_year": int(label_year),
                "window_start_year": int(observed["window_start_year"].iloc[0]),
                "window_end_year": int(observed["window_end_year"].iloc[0]),
                "source_aggregation_operator": (SOURCE_AGGREGATION_OPERATOR),
                "model_aggregation_construction": (MODEL_AGGREGATION_CONSTRUCTION),
                "named_group_total_variation_distance": (standardised_tv),
                "named_group_wrms_log_share_discrepancy": (standardised_wrms),
                "native_named_group_total_variation_distance": native_tv,
                "native_named_group_wrms_log_share_discrepancy": (native_wrms),
                "tv_material_at_0_05": standardised_tv >= TV_MATERIAL,
                "wrms_material_at_log_1_10": (standardised_wrms >= WRMS_MATERIAL),
                "native_tv_material_at_0_05": native_tv >= TV_MATERIAL,
                "native_wrms_material_at_log_1_10": (native_wrms >= WRMS_MATERIAL),
                "model_named_true_count_mean": model_named,
                "source_standardised_named_count_on_model_window": (
                    source_standardised_named
                ),
                "source_native_named_implied_true_count": (source_native_named),
                "model_unsupported_true_count_mean": model_unsupported,
                "model_all_race_true_count_mean": model_all,
                "standardised_absolute_coherence_relative_difference": (
                    standardised_coherence
                ),
                "standardised_absolute_coherence_within_10_percent": (
                    abs(standardised_coherence) <= 0.10
                ),
                "native_absolute_coherence_relative_difference": (native_coherence),
                "native_absolute_coherence_within_10_percent": (
                    abs(native_coherence) <= 0.10
                ),
                "composition_interpretation": observed[
                    "composition_interpretation"
                ].iloc[0],
                "comparison_status": ("single_complete_window_pooled_descriptive_only"),
                "calibration_eligible": False,
            }
        )
    return pd.DataFrame(rows).sort_values(["label_year"], ignore_index=True)


def centered_uncertainty_table(
    fit: FitAuditInput,
    internal: pd.DataFrame,
    surveillance: pd.DataFrame,
    support: pd.DataFrame,
) -> pd.DataFrame:
    """Combine model covariance with conditional source-uncertainty scenarios."""
    rows: list[dict[str, Any]] = []
    contrast_indices = [idx for idx in STABLE_RACE_INDICES if idx != WHITE_RACE_IDX]
    for label_year, source_window in surveillance.groupby("year", observed=True):
        label_year = int(label_year)
        support_window = support[support["label_year"] == label_year]
        if (
            len(support_window) != len(NAMED_RACE_INDICES)
            or not support_window["window_complete"].all()
        ):
            continue
        source_by_race = source_window.set_index("race_idx")["surveillance_prev_per10k"]
        bundle = _pooled_window_model_draws(
            fit,
            internal,
            label_year=label_year,
        )
        prevalence = bundle["prevalence"]
        model_log_contrasts = np.column_stack(
            [
                np.log(prevalence[:, idx]) - np.log(prevalence[:, WHITE_RACE_IDX])
                for idx in contrast_indices
            ]
        )
        source_log_contrasts = np.asarray(
            [
                np.log(float(source_by_race.loc[idx]))
                - np.log(float(source_by_race.loc[WHITE_RACE_IDX]))
                for idx in contrast_indices
            ]
        )
        difference = source_log_contrasts - model_log_contrasts.mean(axis=0)
        model_covariance = np.atleast_2d(
            np.cov(model_log_contrasts, rowvar=False, ddof=1)
        )
        for scenario, values in UNCERTAINTY_SCENARIOS.items():
            sigma = float(np.sqrt(np.log1p(values["stable_race_cv"] ** 2)))
            source_covariance = (
                np.diag(np.full(len(contrast_indices), sigma**2)) + sigma**2
            )
            covariance = source_covariance + model_covariance
            distance_sq = float(difference @ np.linalg.solve(covariance, difference))
            degrees_freedom = len(contrast_indices)
            threshold = float(
                chi2.ppf(
                    UNCERTAINTY_REFERENCE_PROB,
                    df=degrees_freedom,
                )
            )
            rows.append(
                {
                    "label_year": label_year,
                    "source_aggregation_operator": (SOURCE_AGGREGATION_OPERATOR),
                    "model_aggregation_construction": (MODEL_AGGREGATION_CONSTRUCTION),
                    "scenario": scenario,
                    "shared_cv": values["shared_cv"],
                    "stable_race_specific_cv": values["stable_race_cv"],
                    "aian_race_specific_cv_reported_not_in_primary_distance": (
                        values["aian_cv"]
                    ),
                    "common_source_component_cancels_from_log_rate_contrasts": (True),
                    "model_posterior_covariance_included": True,
                    "stable_nonwhite_log_rate_dimensions": (degrees_freedom),
                    "mahalanobis_distance_squared": distance_sq,
                    "conditional_chi_squared_reference_prob": (
                        UNCERTAINTY_REFERENCE_PROB
                    ),
                    "conditional_chi_squared_reference_threshold": (threshold),
                    "exceeds_conditional_reference": (distance_sq > threshold),
                    "interpretation": (
                        "assumption-scenario discrepancy scale, not a "
                        "p-value or source-provided uncertainty interval"
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["label_year", "scenario"], ignore_index=True)


def centered_audit_decision(
    comparison: pd.DataFrame,
    composition: pd.DataFrame,
    uncertainty: pd.DataFrame,
    support: pd.DataFrame,
    denominator: pd.DataFrame,
) -> dict[str, Any]:
    """Evaluate the single complete pooled window while keeping calibration closed."""
    complete_by_label = support.groupby("label_year", observed=True)[
        "window_complete"
    ].all()
    complete_labels = [
        int(label) for label, complete in complete_by_label.items() if bool(complete)
    ]
    partial_labels = [
        int(label)
        for label, complete in complete_by_label.items()
        if not bool(complete)
    ]
    evaluable = comparison[comparison["source_aligned_evaluable"]].copy()
    condition_size = bool(
        len(composition) == len(complete_labels) == 1
        and (
            composition["tv_material_at_0_05"]
            | composition["wrms_material_at_log_1_10"]
        ).all()
    )
    robust_groups: list[str] = []
    contrast_indices = [idx for idx in STABLE_RACE_INDICES if idx != WHITE_RACE_IDX]
    for race_idx in contrast_indices:
        frame = evaluable[evaluable["race_idx"] == race_idx]
        if len(frame) != 1:
            continue
        ratio = float(
            frame["source_vs_model_relative_rate_ratio_to_white_mean"].iloc[0]
        )
        if (
            ratio >= 1.0 + CONTRAST_RELATIVE_MATERIAL
            or ratio <= 1.0 - CONTRAST_RELATIVE_MATERIAL
        ):
            robust_groups.append(RACE_LEVELS[race_idx])
    condition_groups = len(robust_groups) >= 2
    moderate = uncertainty[uncertainty["scenario"] == "moderate"]
    condition_uncertainty = bool(
        len(moderate) == len(complete_labels) == 1
        and moderate["exceeds_conditional_reference"].all()
    )
    pooled_descriptive_signal = (
        len(complete_labels) == 1
        and condition_size
        and condition_groups
        and condition_uncertainty
    )
    denominator_material = bool(
        denominator["group_denominator_material"].any()
        or denominator["named_total_denominator_material"].any()
    )
    blockers = [
        "only one centred five-year source window is fully covered by DSP004",
        "the 2016 centred window requires missing 2014-2015 eta draws",
        "Hispanic-origin precedence and multi-race bridging are unresolved",
        "surveillance sampling covariance is not source-provided",
        "the two centred source windows overlap in 2016-2018 and are not independent",
        "dependence on the national reduction evidence is unresolved",
        "the mirrored age-on-recording model-adequacy gate is outstanding",
    ]
    if denominator_material:
        blockers.append(
            "source and model pooled-window race denominators differ materially"
        )
    return {
        "surveillance_label_years": sorted(
            int(value) for value in support["label_year"].unique()
        ),
        "complete_centred_window_label_years": complete_labels,
        "partial_centred_window_label_years": partial_labels,
        "source_window_alignment_confirmed": "centred",
        "source_race_basis_confirmed": "maternal",
        "source_aggregation_operator_confirmed": (SOURCE_AGGREGATION_OPERATOR),
        "pooled_source_counts_available": True,
        "pooled_source_counts_calibration_eligible": False,
        "descriptive_signal_for_time_invariant_race_layer": None,
        "original_annual_label_protocol_decision": (
            "not_evaluable_under_confirmed_centred_window_definition"
        ),
        "temporal_replication_evaluable": False,
        "cross_window_transport_evaluable": False,
        "single_complete_window_pooled_descriptive_signal": (pooled_descriptive_signal),
        "pooled_descriptive_signal_components": {
            "material_composition_discrepancy": condition_size,
            "robust_non_aian_relative_rate_groups": robust_groups,
            "at_least_two_robust_non_aian_groups": condition_groups,
            "moderate_uncertainty_distance_exceeded": (condition_uncertainty),
        },
        "source_denominator_mapping_material": denominator_material,
        "calibration_eligible": False,
        "calibration_blockers": blockers,
        "time_invariant_race_layer_fit_authorized": False,
        "race_by_year_layer_fit_authorized": False,
        "absolute_race_scale_fit_authorized": False,
        "decision_status": (
            "single_complete_window_pooled_signal_calibration_blocked"
            if pooled_descriptive_signal
            else "centred_window_partially_unsupported_inconclusive"
        ),
        "recommended_next_action": (
            "resolve the category crosswalk and source covariance; extend the "
            "reference fit to 2014 only if the 2016 window is required; no "
            "race calibration or race-layer fit is authorized"
        ),
    }


def _save_figure(fig: Any, output_dir: Path, stem: str) -> tuple[Path, Path]:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    png = plots_dir / f"{stem}.png"
    svg = plots_dir / f"{stem}.svg"
    fig.savefig(png, dpi=plot_styles.DPI_FILE, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return png, svg


def plot_centered_prevalence_comparison(
    comparison: pd.DataFrame,
    support: pd.DataFrame,
    output_dir: Path,
    *,
    interval_prob: float,
) -> tuple[Path, Path]:
    """Plot incomplete 2016 support and the aligned pooled 2018 comparison."""
    fig, axes = plt.subplots(1, 2, figsize=plot_styles.FIGSIZE_XL, sharey=True)
    partial = sorted(
        int(label)
        for label, frame in support.groupby("label_year", observed=True)
        if not frame["window_complete"].all()
    )
    partial_label = partial[0] if partial else 2016
    axes[0].axis("off")
    axes[0].text(
        0.5,
        0.60,
        f"{partial_label} centred window\n"
        f"({partial_label - 2}–{partial_label + 2})\n\n"
        "Not estimable under DSP004:\n"
        "2014–2015 reduction draws absent",
        ha="center",
        va="center",
        transform=axes[0].transAxes,
    )
    frame = comparison[comparison["source_aligned_evaluable"]].sort_values("race_idx")
    x = np.arange(len(frame))
    mean = frame["model_prevalence_per10k_mean"].to_numpy(dtype=float)
    lo = frame["model_prevalence_per10k_lo"].to_numpy(dtype=float)
    hi = frame["model_prevalence_per10k_hi"].to_numpy(dtype=float)
    axes[1].errorbar(
        x - 0.08,
        mean,
        yerr=np.vstack((mean - lo, hi - mean)),
        fmt="o",
        capsize=3,
        color=plot_styles.COLOUR_BLUE,
        label=f"DSP004 pooled analogue ({interval_label(interval_prob)} ETI)",
    )
    axes[1].scatter(
        x + 0.08,
        frame["surveillance_prev_per10k"],
        marker="s",
        color=plot_styles.COLOUR_ORANGE,
        label="Raw pooled source ratio",
    )
    label = int(frame["label_year"].iloc[0])
    axes[1].set_title(f"{label}: {label - 2}–{label + 2}\nConfirmed pooled count ratio")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(frame["race_model_label"], rotation=35, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_ylabel("True DS livebirth prevalence per 10,000")
    handles, labels = axes[1].get_legend_handles_labels()
    axes[0].legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.10),
        frameon=False,
    )
    fig.suptitle("DSP004 centred-window race-prevalence audit")
    fig.text(
        0.5,
        -0.04,
        "The pooled operator is confirmed; category crosswalk and source "
        "covariance remain unresolved.",
        ha="center",
        fontsize="small",
    )
    return _save_figure(fig, output_dir, "core_race_surveillance_prevalence")


def plot_centered_composition_comparison(
    comparison: pd.DataFrame,
    support: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Plot pooled model and source composition on two birth denominators."""
    fig, axes = plt.subplots(1, 2, figsize=plot_styles.FIGSIZE_XL, sharey=True)
    partial = sorted(
        int(label)
        for label, frame in support.groupby("label_year", observed=True)
        if not frame["window_complete"].all()
    )
    partial_label = partial[0] if partial else 2016
    axes[0].axis("off")
    axes[0].text(
        0.5,
        0.60,
        f"{partial_label} centred window\n"
        f"({partial_label - 2}–{partial_label + 2})\n\n"
        "No composition comparison:\n"
        "window coverage is 3/5",
        ha="center",
        va="center",
        transform=axes[0].transAxes,
    )
    frame = comparison[comparison["source_aligned_evaluable"]].sort_values("race_idx")
    width = 0.25
    x = np.arange(len(frame))
    axes[1].bar(
        x - width,
        frame["model_named_case_share_mean"],
        width,
        color=plot_styles.COLOUR_BLUE,
        label="DSP004 pooled analogue",
    )
    axes[1].bar(
        x,
        frame["source_standardised_named_case_share"],
        width,
        color=plot_styles.COLOUR_ORANGE,
        label="Source rates on model births",
    )
    axes[1].bar(
        x + width,
        frame["source_native_named_case_share"],
        width,
        color=plot_styles.COLOUR_GREEN,
        label="Source pooled births",
    )
    label = int(frame["label_year"].iloc[0])
    axes[1].set_title(
        f"{label}: {label - 2}–{label + 2}\nConfirmed pooled case allocation"
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(frame["race_model_label"], rotation=35, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_ylabel("Share among five named groups")
    handles, labels = axes[1].get_legend_handles_labels()
    axes[0].legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.08),
        frameon=False,
    )
    fig.suptitle("DSP004 centred-window named-group composition audit")
    fig.text(
        0.5,
        -0.04,
        "Native source counts use summed five-year race-specific births; "
        "crosswalk differences remain a calibration blocker.",
        ha="center",
        fontsize="small",
    )
    return _save_figure(fig, output_dir, "core_race_surveillance_composition")


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def _write_table(frame: pd.DataFrame, output_dir: Path, stem: str) -> Path:
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    path = tables_dir / f"{stem}.csv"
    frame.to_csv(path, index=False)
    return path


def write_audit_outputs(
    output_dir: Path | str,
    *,
    fit: FitAuditInput,
    db_path: Path,
    table: str,
    surveillance_path: Path,
    invocation: str,
    race_cells: pd.DataFrame,
    cohort_reconciliation: pd.DataFrame,
    internal: pd.DataFrame,
    source_inventory: pd.DataFrame,
    support: pd.DataFrame,
    comparison: pd.DataFrame,
    denominator: pd.DataFrame,
    composition: pd.DataFrame,
    uncertainty: pd.DataFrame,
    decision: dict[str, Any],
    identities: dict[str, float],
) -> dict[str, Path]:
    """Write machine-readable audit tables, provenance, decision, and plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "cohort_reconciliation": cohort_reconciliation,
        "race_internal_accounting": internal,
        "source_inventory": source_inventory,
        "centered_window_support": support,
        "centered_window_comparison": comparison,
        "source_pooled_window_denominator_mapping": denominator,
        "centered_composition_summary": composition,
        "centered_uncertainty_scenarios": uncertainty,
    }
    paths: dict[str, Path] = {}
    for name, frame in tables.items():
        paths[name] = _write_table(frame, output_dir, name)

    fit_dir = fit.run_dir.resolve()
    config_path = fit_dir / "config.json"
    cells_path = fit_dir / "cells.parquet"
    idata_path = fit_dir / "idata.nc"
    summary_path = fit_dir / "summary.csv"
    run_config_path = fit_dir / "run_config.json"
    input_hashes = {
        "audit_script_sha256": _sha256(Path(__file__).resolve()),
        "protocol_note_sha256": _sha256(PROTOCOL_PATH),
        "fit_config_sha256": _sha256(config_path),
        "fit_cells_sha256": _sha256(cells_path),
        "fit_idata_sha256": _sha256(idata_path),
        "fit_summary_sha256": _sha256(summary_path),
        "surveillance_source_sha256": _sha256(surveillance_path),
        "race_aggregate_digest": _frame_digest(
            race_cells,
            ["year", "maternal_age", "race_idx", "N_cell", "R_cell"],
        ),
    }
    if run_config_path.is_file():
        input_hashes["fit_run_config_sha256"] = _sha256(run_config_path)

    metadata = {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "audit_id": "DSP004-race-surveillance-centred-no-refit",
        "protocol_status": (
            "corrected_source_alignment_protocol_not_blind_preregistration"
        ),
        "protocol_path": str(PROTOCOL_PATH),
        "no_refit": True,
        "model_id": fit.config["model_id"],
        "fit_profile": fit.run_config["name"],
        "fit_dir": str(fit_dir),
        "duckdb_path": str(db_path.resolve()),
        "duckdb_access": "read_only",
        "duckdb_table": table,
        "surveillance_path": str(surveillance_path.resolve()),
        "surveillance_primary_field": "surveillance_prev_per10k",
        "filled_estimate_fields_eligible_for_primary": False,
        "surveillance_label_years": sorted(
            int(value) for value in comparison["label_year"].unique()
        ),
        "year_range": list(fit.year_range),
        "interval_prob": fit.interval_prob,
        "analytic_filters": ["mage_c IS NOT NULL", "down_ind IS NOT NULL"],
        "recorded_definition": fit.config["recorded_definition"],
        "maternal_age_endpoint_convention": fit.config.get("age_endpoint_convention"),
        "race_map": {str(key): value for key, value in RACE_MAP.items()},
        "race_levels": list(RACE_LEVELS),
        "named_surveillance_race_indices": list(NAMED_RACE_INDICES),
        "unsupported_external_race_indices": [5, 6],
        "source_definition": {
            "window_width_years": 5,
            "window_alignment": "centred",
            "label_role": "centre_year",
            "race_basis": "maternal",
            "confirmed_by": "project lead",
            "confirmed_date": "2026-08-03",
            "aggregation_operator": {
                "status": "confirmed",
                "name": SOURCE_AGGREGATION_OPERATOR,
                "definition": (
                    "five-year true-case numerator divided by five-year "
                    "birth denominator"
                ),
            },
            "birth_denominator_components": (
                "sum of annual race-specific births_bc over the centred window"
            ),
            "window_overlap": {
                "label_years": [2016, 2018],
                "shared_years": [2016, 2017, 2018],
                "fraction_of_each_window": 0.6,
                "jaccard_fraction": 3 / 7,
                "independent_validation": False,
            },
        },
        "source_window_alignment_status": "confirmed_centred",
        "source_race_basis_status": "confirmed_maternal",
        "source_aggregation_operator_status": "confirmed_pooled_count_ratio",
        "source_hispanic_precedence_status": "unresolved",
        "source_multi_race_crosswalk_status": "unresolved",
        "source_sampling_covariance_status": "not_source_provided",
        "source_race_crosswalk_status": "unresolved",
        "complete_centred_window_label_years": decision[
            "complete_centred_window_label_years"
        ],
        "partial_centred_window_label_years": decision[
            "partial_centred_window_label_years"
        ],
        "comparison_status": "single_complete_window_pooled_descriptive_only",
        "invocation": invocation,
        "runtime": {
            "python": platform.python_version(),
            "packages": _package_versions(),
        },
        "git": _git_provenance(),
        "fit_health": fit.fit_health,
        "posterior_identity_checks": identities,
        "input_hashes": input_hashes,
        "thresholds": {
            "total_variation_material": TV_MATERIAL,
            "wrms_log_share_material": WRMS_MATERIAL,
            "relative_rate_ratio_upper": 1.0 + CONTRAST_RELATIVE_MATERIAL,
            "relative_rate_ratio_lower": 1.0 - CONTRAST_RELATIVE_MATERIAL,
            "group_denominator_relative_material": (
                DENOMINATOR_GROUP_RELATIVE_MATERIAL
            ),
            "named_total_denominator_relative_material": (
                DENOMINATOR_NAMED_TOTAL_RELATIVE_MATERIAL
            ),
            "uncertainty_reference_prob": UNCERTAINTY_REFERENCE_PROB,
        },
        "uncertainty_scenarios": UNCERTAINTY_SCENARIOS,
        "decision": decision,
        "interpretation_limits": [
            "aggregate population accounting only",
            "surveillance points are centred five-year estimates, not annual observations",
            "2016 is unsupported because DSP004 has no 2014-2015 eta draws",
            "source pooled counts remain descriptive until category mapping and covariance are resolved",
            "surveillance sampling covariance is not source-provided",
            "race contrasts are national associations and not separable mechanisms",
            "Unknown and NH Multi-race have no external surveillance target",
        ],
    }
    sections = {name: _json_records(frame) for name, frame in tables.items()}
    payload = {"metadata": metadata, "sections": sections}
    audit_json = output_dir / "audit.json"
    audit_json.write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    paths["audit_json"] = audit_json

    prevalence_png, prevalence_svg = plot_centered_prevalence_comparison(
        comparison,
        support,
        output_dir,
        interval_prob=fit.interval_prob,
    )
    composition_png, composition_svg = plot_centered_composition_comparison(
        comparison, support, output_dir
    )
    paths.update(
        {
            "prevalence_plot_png": prevalence_png,
            "prevalence_plot_svg": prevalence_svg,
            "composition_plot_png": composition_png,
            "composition_plot_svg": composition_svg,
        }
    )

    config_path_out = output_dir / "race_surveillance_audit_config.json"
    paths["audit_config"] = config_path_out
    config_payload = {"metadata": metadata, "artefact_paths": {}}
    config_payload["artefact_paths"] = {
        key: str(path.relative_to(output_dir)) for key, path in paths.items()
    }
    config_path_out.write_text(
        json.dumps(config_payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    return paths


def run_audit(
    *,
    fit_dir: Path | str,
    db_path: Path | str,
    surveillance_path: Path | str,
    years: tuple[int, ...] = DEFAULT_SURVEILLANCE_YEARS,
    interval_prob: float = DEFAULT_ETI_PROB,
    output_dir: Path | str,
    table: str = "us_births",
    invocation: str = "programmatic run_audit call; see structured parameters",
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Run the complete read-only race-surveillance audit."""
    if len(years) != 2 or set(years) != set(DEFAULT_SURVEILLANCE_YEARS):
        raise ValueError(
            "the frozen audit requires exactly the direct-surveillance years "
            f"{DEFAULT_SURVEILLANCE_YEARS!r}; found {years!r}"
        )
    fit = load_fit_input(fit_dir, interval_prob=interval_prob)
    if any(year < fit.year_range[0] or year > fit.year_range[1] for year in years):
        raise ValueError("surveillance years must lie inside the fitted year range")
    db_path = Path(db_path)
    surveillance_path = Path(surveillance_path)
    race_cells = load_race_age_year_cells(
        db_path, year_range=fit.year_range, table=table
    )
    cohort = reconcile_saved_cells(fit, race_cells)
    internal, identities = reconstruct_internal_accounting(fit, race_cells)
    surveillance, inventory = load_surveillance(surveillance_path, years=years)
    support = centered_window_support_table(internal, surveillance)
    comparison = build_centered_window_comparison(fit, internal, surveillance, support)
    denominator = pooled_window_denominator_reconciliation(internal, surveillance)
    composition = centered_composition_summary_table(comparison)
    uncertainty = centered_uncertainty_table(fit, internal, surveillance, support)
    decision = centered_audit_decision(
        comparison, composition, uncertainty, support, denominator
    )
    paths = write_audit_outputs(
        output_dir,
        fit=fit,
        db_path=db_path,
        table=table,
        surveillance_path=surveillance_path,
        invocation=invocation,
        race_cells=race_cells,
        cohort_reconciliation=cohort,
        internal=internal,
        source_inventory=inventory,
        support=support,
        comparison=comparison,
        denominator=denominator,
        composition=composition,
        uncertainty=uncertainty,
        decision=decision,
        identities=identities,
    )
    return paths, decision


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--duckdb-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--surveillance-csv", type=Path, default=DEFAULT_SURVEILLANCE_CSV
    )
    parser.add_argument(
        "--years",
        type=_parse_years,
        default=DEFAULT_SURVEILLANCE_YEARS,
        help="Exactly two comma-separated direct-surveillance years (default: 2016,2018).",
    )
    parser.add_argument("--interval-prob", type=float, default=DEFAULT_ETI_PROB)
    parser.add_argument("--table", default="us_births")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    setup.init_script()
    raw_args = list(sys.argv[1:] if argv is None else argv)
    ns = parse_args(raw_args)
    output_dir = ns.output_dir or (
        DEFAULT_OUTPUT_ROOT / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    paths, decision = run_audit(
        fit_dir=ns.fit_dir,
        db_path=ns.duckdb_path,
        surveillance_path=ns.surveillance_csv,
        years=ns.years,
        interval_prob=ns.interval_prob,
        output_dir=output_dir,
        table=ns.table,
        invocation=shlex.join(
            [sys.executable, str(Path(__file__).resolve()), *raw_args]
        ),
    )
    print("DSP004 no-refit race-surveillance audit")
    print(f"output: {Path(output_dir).resolve()}")
    print(f"decision: {decision['decision_status']}")
    print(f"calibration eligible: {decision['calibration_eligible']}")
    print(f"artefacts: {len(paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
