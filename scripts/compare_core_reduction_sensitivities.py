"""Compare a saved multi-run core-model assumption-sensitivity grid.

The comparison is deliberately table-driven.  Runs of the same core model use
the same native age-by-year cells, so their checked-in reporting contracts are
more direct than reconstructing another posterior-predictive sample from each
NetCDF file.  The NetCDF is opened only to count sampler divergences.

The two intended sensitivity axes are the fixed false-positive probability
``f`` and the surveillance-derived reduction-prior widths.  Incomplete
factorial grids are allowed and are recorded explicitly in the output config.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import dse_research_utils.environment.setup as setup
import dse_research_utils.plot.styles as plot_styles
import numpy as np
import pandas as pd
import xarray as xr

from dspopulations_us_birth_certificates import cli_output
from dspopulations_us_birth_certificates.selection import diagnostics
from dspopulations_us_birth_certificates.selection.core_reduction import (
    CORE_REDUCTION_MODEL_ID,
)

ALLOWED_VARYING_PRIORS = frozenset({"false_positive_rate", "reduction_sigma"})
REQUIRED_TABLES = {
    "headlines": "core_headlines.csv",
    "accounting_by_year": "core_accounting_by_year.csv",
    "reduction_by_year": "core_reduction_prior_posterior.csv",
    "recording_by_year": "core_recording_s_by_year.csv",
    "ppc_age_band": "core_ppc_by_age_band.csv",
    "ppc_age_year": "core_ppc_by_age_year.csv",
}
CONTRAST_METRICS = (
    "true_ds_mean",
    "recording_s_mean",
    "aggregate_reduction_mean",
    "age_year_coverage_fraction",
    "age_year_mean_absolute_standardized_residual",
    "age_band_coverage_fraction",
)
ENVELOPE_METRICS = {
    "true_ds_livebirths": ("true_ds_mean", "true_ds_lo", "true_ds_hi"),
    "recording_s": ("recording_s_mean", "recording_s_lo", "recording_s_hi"),
    "aggregate_reduction": ("aggregate_reduction_mean", None, None),
    "expected_false_positive_flags": (
        "expected_false_positive_flags_mean",
        "expected_false_positive_flags_lo",
        "expected_false_positive_flags_hi",
    ),
    "age_year_coverage_fraction": ("age_year_coverage_fraction", None, None),
    "age_year_mean_absolute_standardized_residual": (
        "age_year_mean_absolute_standardized_residual",
        None,
        None,
    ),
}


@dataclass(frozen=True)
class CoreSensitivityRun:
    """One completed fit and its native reporting artefacts."""

    run_dir: Path
    scenario_id: str
    is_reference: bool
    config: dict[str, Any]
    run_config: dict[str, Any]
    cells: pd.DataFrame
    tables: dict[str, pd.DataFrame]
    summary: pd.DataFrame
    false_positive_rate: float
    observed_reduction_sigma: float
    extrapolated_reduction_sigma: float
    interval_prob: float
    max_rhat: float
    min_ess: float
    convergence_ok: bool
    divergences: int | None

    @property
    def reduction_width_id(self) -> str:
        """Human-readable identifier for one prior-width regime."""
        return (
            f"{self.observed_reduction_sigma:.8g}/"
            f"{self.extrapolated_reduction_sigma:.8g}"
        )

    @property
    def factor_key(self) -> tuple[float, float, float]:
        """Stable numeric key for duplicate-scenario checks."""
        return (
            self.false_positive_rate,
            self.observed_reduction_sigma,
            self.extrapolated_reduction_sigma,
        )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return value


def _read_csv(path: Path, *, index_col: int | None = None) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, index_col=index_col)


def _require_columns(
    frame: pd.DataFrame,
    columns: set[str],
    *,
    description: str,
) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(
            f"{description} is missing required columns: {sorted(missing)!r}."
        )


def _reduction_prior_widths(config: dict[str, Any]) -> tuple[float, float]:
    """Extract the pre/post-extrapolation logit-scale prior widths."""
    try:
        start, end = (int(value) for value in config["year_range"])
        priors = config["priors"]
        extrapolated_start = int(priors["extrapolated_reduction_start"])
        sigma = np.asarray(priors["reduction_sigma"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "config must contain year_range and the reduction-prior provenance."
        ) from exc
    years = np.arange(start, end + 1)
    if len(sigma) != len(years):
        raise ValueError(
            "priors.reduction_sigma must have one value per modelled year."
        )
    if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError("priors.reduction_sigma values must be finite and positive.")
    observed = sigma[years < extrapolated_start]
    extrapolated = sigma[years >= extrapolated_start]
    if not len(observed) or not len(extrapolated):
        raise ValueError(
            "The sensitivity comparison requires both observed and extrapolated "
            "reduction-prior years."
        )
    if not np.allclose(observed, observed[0], rtol=0.0, atol=1e-12):
        raise ValueError("Observed-year reduction prior widths are not constant.")
    if not np.allclose(extrapolated, extrapolated[0], rtol=0.0, atol=1e-12):
        raise ValueError("Extrapolated-year reduction prior widths are not constant.")
    return float(observed[0]), float(extrapolated[0])


def _report_interval_probability(tables: dict[str, pd.DataFrame]) -> float:
    values: list[float] = []
    for name in (
        "accounting_by_year",
        "recording_by_year",
        "ppc_age_band",
        "ppc_age_year",
    ):
        frame = tables[name]
        if "interval_prob" not in frame:
            raise ValueError(f"{name} has no interval_prob provenance column.")
        unique = frame["interval_prob"].dropna().to_numpy(dtype=float)
        if unique.size:
            values.extend(unique.tolist())
    if not values:
        raise ValueError("Reporting tables contain no interval probability.")
    if not np.allclose(values, values[0], rtol=0.0, atol=1e-12):
        raise ValueError("Reporting tables use inconsistent interval probabilities.")
    return float(values[0])


def _count_divergences(path: Path) -> int | None:
    """Read only the sampler divergence array from a saved NetCDF."""
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        with xr.open_dataset(path, group="sample_stats") as sample_stats:
            if "diverging" not in sample_stats:
                return None
            return int(np.asarray(sample_stats["diverging"].values).sum())
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read sample_stats from {path}.") from exc


def _scenario_id(
    false_positive_rate: float, observed: float, extrapolated: float
) -> str:
    return f"f={false_positive_rate:.8g}; rho_sigma={observed:.8g}/{extrapolated:.8g}"


def load_sensitivity_run(
    run_dir: Path,
    *,
    is_reference: bool = False,
) -> CoreSensitivityRun:
    """Load and validate the reporting contract for one saved fit."""
    run_dir = Path(run_dir)
    config = _read_json(run_dir / "config.json")
    run_config = _read_json(run_dir / "run_config.json")
    cells_path = run_dir / "cells.parquet"
    if not cells_path.is_file():
        raise FileNotFoundError(cells_path)
    cells = pd.read_parquet(cells_path)
    tables = {
        name: _read_csv(run_dir / "tables" / filename)
        for name, filename in REQUIRED_TABLES.items()
    }
    summary = _read_csv(run_dir / "summary.csv", index_col=0)

    _require_columns(
        tables["headlines"],
        {"metric", "mean", "lo", "hi"},
        description="core_headlines.csv",
    )
    _require_columns(
        tables["accounting_by_year"],
        {
            "year",
            "births",
            "recorded_ds",
            "natural_expected_ds",
            "true_count_year_mean",
            "true_count_year_lo",
            "true_count_year_hi",
        },
        description="core_accounting_by_year.csv",
    )
    _require_columns(
        tables["reduction_by_year"],
        {
            "year",
            "rho_prior_mean",
            "rho_prior_lo",
            "rho_prior_hi",
            "rho_prior_sigma_logit",
            "rho_year_mean",
            "rho_year_lo",
            "rho_year_hi",
        },
        description="core_reduction_prior_posterior.csv",
    )
    _require_columns(
        tables["recording_by_year"],
        {"year", "posterior_mean", "posterior_lo", "posterior_hi"},
        description="core_recording_s_by_year.csv",
    )
    _require_columns(
        tables["ppc_age_band"],
        {
            "age_band_idx",
            "label",
            "observed",
            "predicted_mean",
            "predicted_lo",
            "predicted_hi",
            "observed_in_interval",
        },
        description="core_ppc_by_age_band.csv",
    )
    _require_columns(
        tables["ppc_age_year"],
        {
            "year_idx",
            "year",
            "age_idx",
            "age",
            "births",
            "observed",
            "predicted_mean",
            "predicted_lo",
            "predicted_hi",
            "posterior_predictive_sd",
            "standardized_residual",
            "observed_in_interval",
        },
        description="core_ppc_by_age_year.csv",
    )

    try:
        false_positive_rate = float(config["priors"]["false_positive_rate"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("config has no valid priors.false_positive_rate.") from exc
    if not np.isfinite(false_positive_rate) or not 0.0 <= false_positive_rate < 1.0:
        raise ValueError("priors.false_positive_rate must be finite and lie in [0, 1).")
    observed_sigma, extrapolated_sigma = _reduction_prior_widths(config)
    interval_prob = _report_interval_probability(tables)
    health = diagnostics.convergence_health(summary)
    divergences = _count_divergences(run_dir / "idata.nc")

    return CoreSensitivityRun(
        run_dir=run_dir,
        scenario_id=_scenario_id(
            false_positive_rate,
            observed_sigma,
            extrapolated_sigma,
        ),
        is_reference=is_reference,
        config=config,
        run_config=run_config,
        cells=cells,
        tables=tables,
        summary=summary,
        false_positive_rate=false_positive_rate,
        observed_reduction_sigma=observed_sigma,
        extrapolated_reduction_sigma=extrapolated_sigma,
        interval_prob=interval_prob,
        max_rhat=float(health["max_rhat"]),
        min_ess=float(health["min_ess"]),
        convergence_ok=bool(health["all_ok"]),
        divergences=divergences,
    )


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _invariant_priors(config: dict[str, Any]) -> dict[str, Any]:
    priors = config.get("priors")
    if not isinstance(priors, dict):
        raise ValueError("config.priors must be a JSON object.")
    return {
        key: value for key, value in priors.items() if key not in ALLOWED_VARYING_PRIORS
    }


def _sorted_cells(run: CoreSensitivityRun) -> pd.DataFrame:
    required = ["year_idx", "age_idx", "N_cell", "R_cell"]
    if run.config.get("age_model") == "single_year":
        required.insert(2, "maternal_age")
    _require_columns(
        run.cells, set(required), description=f"{run.run_dir}/cells.parquet"
    )
    return (
        run.cells[required].sort_values(["year_idx", "age_idx"]).reset_index(drop=True)
    )


def _assert_same_native_observations(
    reference: CoreSensitivityRun,
    candidate: CoreSensitivityRun,
) -> None:
    ref_age_year = reference.tables["ppc_age_year"].sort_values(["year_idx", "age_idx"])
    candidate_age_year = candidate.tables["ppc_age_year"].sort_values(
        ["year_idx", "age_idx"]
    )
    age_year_keys = [
        "year_idx",
        "year",
        "age_idx",
        "age",
        "births",
        "observed",
    ]
    if (
        not ref_age_year[age_year_keys]
        .reset_index(drop=True)
        .equals(candidate_age_year[age_year_keys].reset_index(drop=True))
    ):
        raise ValueError("Runs do not share the same native age-by-year PPC grid.")

    ref_band = reference.tables["ppc_age_band"].sort_values("age_band_idx")
    candidate_band = candidate.tables["ppc_age_band"].sort_values("age_band_idx")
    band_keys = ["age_band_idx", "label", "observed"]
    if (
        not ref_band[band_keys]
        .reset_index(drop=True)
        .equals(candidate_band[band_keys].reset_index(drop=True))
    ):
        raise ValueError("Runs do not share the same native age-band observations.")


def validate_sensitivity_runs(runs: list[CoreSensitivityRun]) -> None:
    """Require matched models and allow only the two intended prior axes."""
    if len(runs) < 2:
        raise ValueError("At least two fitted runs are required for comparison.")
    if sum(run.is_reference for run in runs) != 1:
        raise ValueError("Exactly one run must be marked as the reference.")
    paths = [run.run_dir.resolve() for run in runs]
    if len(set(paths)) != len(paths):
        raise ValueError("The same fit directory was supplied more than once.")

    factor_keys = [run.factor_key for run in runs]
    if len(set(factor_keys)) != len(factor_keys):
        raise ValueError(
            "Each false-positive/rho-prior factor combination must appear at most once."
        )

    reference = next(run for run in runs if run.is_reference)
    invariant_config_keys = (
        "model_id",
        "model_slug",
        "family_id",
        "recording_model",
        "reduction_model",
        "age_model",
        "recorded_definition",
        "theta_model",
        "age_endpoint_convention",
        "year_range",
    )
    reference_structure = {
        key: reference.config.get(key) for key in invariant_config_keys
    }
    reference_priors = _canonical(_invariant_priors(reference.config))
    reference_cells = _sorted_cells(reference)
    for run in runs:
        structure = {key: run.config.get(key) for key in invariant_config_keys}
        if _canonical(structure) != _canonical(reference_structure):
            raise ValueError(
                f"{run.run_dir} does not use the same core model structure as "
                f"{reference.run_dir}."
            )
        if _canonical(_invariant_priors(run.config)) != reference_priors:
            raise ValueError(
                f"{run.run_dir} changes a prior other than false_positive_rate "
                "or reduction_sigma."
            )
        if not _sorted_cells(run).equals(reference_cells):
            raise ValueError(f"{run.run_dir} does not use the same model cells.")
        if not np.isclose(run.interval_prob, reference.interval_prob):
            raise ValueError("Runs use different reporting interval probabilities.")
        _assert_same_native_observations(reference, run)


def _headline(run: CoreSensitivityRun, metric: str) -> pd.Series:
    table = run.tables["headlines"].set_index("metric")
    if metric not in table.index:
        raise ValueError(f"{run.run_dir} has no {metric!r} headline metric.")
    row = table.loc[metric]
    if isinstance(row, pd.DataFrame):
        raise ValueError(f"{run.run_dir} has duplicate {metric!r} headline rows.")
    return row


def scenario_table(runs: list[CoreSensitivityRun]) -> pd.DataFrame:
    """Return run provenance, factor levels, and sampler-health diagnostics."""
    rows = []
    for order, run in enumerate(runs):
        cfg = run.run_config
        no_divergences = run.divergences == 0 if run.divergences is not None else False
        rows.append(
            {
                "scenario_order": order,
                "scenario_id": run.scenario_id,
                "is_reference": run.is_reference,
                "run_dir": str(run.run_dir),
                "model_id": run.config.get("model_id"),
                "false_positive_rate": run.false_positive_rate,
                "false_positive_rate_per_100k": run.false_positive_rate * 100_000.0,
                "observed_reduction_sigma_logit": run.observed_reduction_sigma,
                "extrapolated_reduction_sigma_logit": (
                    run.extrapolated_reduction_sigma
                ),
                "reduction_width_id": run.reduction_width_id,
                "interval_prob": run.interval_prob,
                "profile": cfg.get("name"),
                "draws": cfg.get("draws"),
                "tune": cfg.get("tune"),
                "chains": cfg.get("chains"),
                "target_accept": cfg.get("target_accept"),
                "nuts_sampler": cfg.get("nuts_sampler"),
                "random_seed": cfg.get("random_seed"),
                "posterior_predictive": cfg.get("posterior_predictive"),
                "max_rhat": run.max_rhat,
                "min_ess": run.min_ess,
                "convergence_ok": run.convergence_ok,
                "divergences": run.divergences,
                "no_divergences": no_divergences,
                "fit_healthy": run.convergence_ok and no_divergences,
            }
        )
    return pd.DataFrame(rows)


def _residual_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "residual_observed_minus_predicted" not in out:
        out["residual_observed_minus_predicted"] = (
            out["observed"] - out["predicted_mean"]
        )
    if "relative_residual" not in out:
        out["relative_residual"] = np.divide(
            out["residual_observed_minus_predicted"],
            out["predicted_mean"],
            out=np.full(len(out), np.nan, dtype=float),
            where=out["predicted_mean"].to_numpy(dtype=float) != 0.0,
        )
    return out


def sensitivity_summary_table(runs: list[CoreSensitivityRun]) -> pd.DataFrame:
    """Return one headline and PPC-calibration row per scenario."""
    rows = []
    for run in runs:
        livebirths = _headline(run, "livebirths")
        recorded = _headline(run, "recorded_ds")
        natural = _headline(run, "natural_expected_ds")
        true_ds = _headline(run, "true_ds_livebirths")
        reduction = _headline(run, "aggregate_reduction")
        recording = _headline(run, "recording_s")
        n_births = float(livebirths["mean"])
        recorded_count = float(recorded["mean"])
        true_mean = float(true_ds["mean"])
        true_lo = float(true_ds["lo"])
        true_hi = float(true_ds["hi"])
        f = run.false_positive_rate
        expected_fp_mean = f * (n_births - true_mean)
        expected_fp_lo = f * (n_births - true_hi)
        expected_fp_hi = f * (n_births - true_lo)

        age_year = _residual_columns(run.tables["ppc_age_year"])
        residual = age_year["residual_observed_minus_predicted"].to_numpy(dtype=float)
        standardized = age_year["standardized_residual"].to_numpy(dtype=float)
        finite_standardized = standardized[np.isfinite(standardized)]
        age_band = _residual_columns(run.tables["ppc_age_band"])
        band_residual = age_band["residual_observed_minus_predicted"].to_numpy(
            dtype=float
        )
        rows.append(
            {
                "scenario_id": run.scenario_id,
                "is_reference": run.is_reference,
                "false_positive_rate": f,
                "false_positive_rate_per_100k": f * 100_000.0,
                "observed_reduction_sigma_logit": run.observed_reduction_sigma,
                "extrapolated_reduction_sigma_logit": (
                    run.extrapolated_reduction_sigma
                ),
                "reduction_width_id": run.reduction_width_id,
                "livebirths": n_births,
                "recorded_ds": recorded_count,
                "natural_expected_ds": float(natural["mean"]),
                "true_ds_mean": true_mean,
                "true_ds_lo": true_lo,
                "true_ds_hi": true_hi,
                "true_ds_interval_width": true_hi - true_lo,
                "aggregate_reduction_mean": float(reduction["mean"]),
                "recording_s_mean": float(recording["mean"]),
                "recording_s_lo": float(recording["lo"]),
                "recording_s_hi": float(recording["hi"]),
                "recording_s_interval_width": (
                    float(recording["hi"]) - float(recording["lo"])
                ),
                "expected_false_positive_flags_mean": expected_fp_mean,
                "expected_false_positive_flags_lo": expected_fp_lo,
                "expected_false_positive_flags_hi": expected_fp_hi,
                "expected_false_positive_share_recorded_mean": (
                    expected_fp_mean / recorded_count
                    if recorded_count > 0.0
                    else np.nan
                ),
                "age_year_cells": len(age_year),
                "age_year_coverage_count": int(
                    age_year["observed_in_interval"].astype(bool).sum()
                ),
                "age_year_coverage_fraction": float(
                    age_year["observed_in_interval"].astype(bool).mean()
                ),
                "age_year_mean_absolute_standardized_residual": (
                    float(np.mean(np.abs(finite_standardized)))
                    if finite_standardized.size
                    else np.nan
                ),
                "age_year_root_mean_squared_residual": float(
                    np.sqrt(np.mean(residual**2))
                ),
                "age_year_total_absolute_residual": float(np.abs(residual).sum()),
                "age_band_cells": len(age_band),
                "age_band_coverage_count": int(
                    age_band["observed_in_interval"].astype(bool).sum()
                ),
                "age_band_coverage_fraction": float(
                    age_band["observed_in_interval"].astype(bool).mean()
                ),
                "age_band_root_mean_squared_residual": float(
                    np.sqrt(np.mean(band_residual**2))
                ),
                "age_band_total_absolute_residual": float(np.abs(band_residual).sum()),
                "ppc_evidence_scope": (
                    "native in-sample posterior-predictive check; not held-out evidence"
                ),
            }
        )
    out = pd.DataFrame(rows)
    reference = out.loc[out["is_reference"]].iloc[0]
    delta_columns = (
        "true_ds_mean",
        "aggregate_reduction_mean",
        "recording_s_mean",
        "age_year_coverage_fraction",
        "age_year_mean_absolute_standardized_residual",
        "age_band_coverage_fraction",
    )
    for column in delta_columns:
        out[f"{column}_difference_from_reference"] = out[column] - reference[column]
    out["true_ds_mean_percent_difference_from_reference"] = (
        100.0
        * out["true_ds_mean_difference_from_reference"]
        / reference["true_ds_mean"]
    )
    return out


def _scenario_columns(run: CoreSensitivityRun) -> dict[str, Any]:
    return {
        "scenario_id": run.scenario_id,
        "is_reference": run.is_reference,
        "false_positive_rate": run.false_positive_rate,
        "false_positive_rate_per_100k": run.false_positive_rate * 100_000.0,
        "observed_reduction_sigma_logit": run.observed_reduction_sigma,
        "extrapolated_reduction_sigma_logit": run.extrapolated_reduction_sigma,
        "reduction_width_id": run.reduction_width_id,
    }


def sensitivity_by_year_table(runs: list[CoreSensitivityRun]) -> pd.DataFrame:
    """Return comparable prior, posterior, recording and count summaries by year."""
    frames = []
    for run in runs:
        accounting = run.tables["accounting_by_year"]
        reduction = run.tables["reduction_by_year"]
        recording = run.tables["recording_by_year"].rename(
            columns={
                "posterior_mean": "recording_s_year_mean",
                "posterior_lo": "recording_s_year_lo",
                "posterior_hi": "recording_s_year_hi",
            }
        )
        accounting_columns = [
            column
            for column in (
                "year",
                "births",
                "recorded_ds",
                "natural_expected_ds",
                "true_count_year_mean",
                "true_count_year_lo",
                "true_count_year_hi",
                "recorded_count_year_mu_mean",
                "recorded_count_year_mu_lo",
                "recorded_count_year_mu_hi",
            )
            if column in accounting
        ]
        reduction_columns = [
            column
            for column in (
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
            )
            if column in reduction
        ]
        recording_columns = [
            "year",
            "recording_s_year_mean",
            "recording_s_year_lo",
            "recording_s_year_hi",
        ]
        merged = (
            accounting[accounting_columns]
            .merge(
                reduction[reduction_columns],
                on="year",
                how="inner",
                validate="one_to_one",
            )
            .merge(
                recording[recording_columns],
                on="year",
                how="inner",
                validate="one_to_one",
            )
        )
        if len(merged) != len(accounting):
            raise ValueError(f"{run.run_dir} has inconsistent by-year tables.")
        for column, value in reversed(list(_scenario_columns(run).items())):
            merged.insert(0, column, value)
        frames.append(merged)
    return pd.concat(frames, ignore_index=True)


def sensitivity_age_band_table(runs: list[CoreSensitivityRun]) -> pd.DataFrame:
    """Return all native broad-age PPC summaries with scenario provenance."""
    frames = []
    for run in runs:
        frame = _residual_columns(run.tables["ppc_age_band"])
        for column, value in reversed(list(_scenario_columns(run).items())):
            frame.insert(0, column, value)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def sensitivity_age_year_table(runs: list[CoreSensitivityRun]) -> pd.DataFrame:
    """Return all native exact-age by year PPC summaries."""
    frames = []
    for run in runs:
        frame = _residual_columns(run.tables["ppc_age_year"])
        for column, value in reversed(list(_scenario_columns(run).items())):
            frame.insert(0, column, value)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _contrast_rows(
    group: pd.DataFrame,
    *,
    varied_factor: str,
    held_factor: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ordered = group.sort_values(
        [
            "false_positive_rate",
            "observed_reduction_sigma_logit",
            "extrapolated_reduction_sigma_logit",
        ]
    )
    for (_, from_row), (_, to_row) in itertools.combinations(ordered.iterrows(), 2):
        for metric in CONTRAST_METRICS:
            from_value = float(from_row[metric])
            to_value = float(to_row[metric])
            rows.append(
                {
                    "varied_factor": varied_factor,
                    "held_factor": held_factor,
                    "from_scenario_id": from_row["scenario_id"],
                    "to_scenario_id": to_row["scenario_id"],
                    "from_false_positive_rate": from_row["false_positive_rate"],
                    "to_false_positive_rate": to_row["false_positive_rate"],
                    "from_reduction_width_id": from_row["reduction_width_id"],
                    "to_reduction_width_id": to_row["reduction_width_id"],
                    "metric": metric,
                    "from_value": from_value,
                    "to_value": to_value,
                    "difference": to_value - from_value,
                    "percent_difference": (
                        100.0 * (to_value - from_value) / from_value
                        if from_value != 0.0
                        else np.nan
                    ),
                    "contrast_scope": (
                        "controlled contrast of posterior summaries; no paired "
                        "posterior interval"
                    ),
                }
            )
    return rows


def sensitivity_contrast_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Return all observed one-factor contrasts in a possibly incomplete grid."""
    rows: list[dict[str, Any]] = []
    for width, group in summary.groupby("reduction_width_id", sort=False):
        if group["false_positive_rate"].nunique() > 1:
            rows.extend(
                _contrast_rows(
                    group,
                    varied_factor="false_positive_rate",
                    held_factor=f"reduction_width_id={width}",
                )
            )
    for f_value, group in summary.groupby("false_positive_rate", sort=False):
        if group["reduction_width_id"].nunique() > 1:
            rows.extend(
                _contrast_rows(
                    group,
                    varied_factor="reduction_prior_width",
                    held_factor=f"false_positive_rate={f_value:.8g}",
                )
            )
    columns = [
        "varied_factor",
        "held_factor",
        "from_scenario_id",
        "to_scenario_id",
        "from_false_positive_rate",
        "to_false_positive_rate",
        "from_reduction_width_id",
        "to_reduction_width_id",
        "metric",
        "from_value",
        "to_value",
        "difference",
        "percent_difference",
        "contrast_scope",
    ]
    return pd.DataFrame(rows, columns=columns)


def sensitivity_envelope_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Return the range across assumptions without calling it a posterior interval."""
    rows = []
    for metric, (mean_column, lo_column, hi_column) in ENVELOPE_METRICS.items():
        minimum_idx = summary[mean_column].idxmin()
        maximum_idx = summary[mean_column].idxmax()
        rows.append(
            {
                "metric": metric,
                "minimum_mean": float(summary.loc[minimum_idx, mean_column]),
                "minimum_mean_scenario_id": summary.loc[minimum_idx, "scenario_id"],
                "maximum_mean": float(summary.loc[maximum_idx, mean_column]),
                "maximum_mean_scenario_id": summary.loc[maximum_idx, "scenario_id"],
                "scenario_mean_span": float(
                    summary.loc[maximum_idx, mean_column]
                    - summary.loc[minimum_idx, mean_column]
                ),
                "envelope_lo": (
                    float(summary[lo_column].min()) if lo_column is not None else np.nan
                ),
                "envelope_hi": (
                    float(summary[hi_column].max()) if hi_column is not None else np.nan
                ),
                "envelope_definition": (
                    "range across fitted assumption scenarios; not a posterior "
                    "credible interval"
                ),
            }
        )
    return pd.DataFrame(rows)


def _factorial_grid_metadata(runs: list[CoreSensitivityRun]) -> dict[str, Any]:
    false_positive_levels = sorted({run.false_positive_rate for run in runs})
    width_levels = sorted(
        {
            (run.observed_reduction_sigma, run.extrapolated_reduction_sigma)
            for run in runs
        }
    )
    observed = {run.factor_key for run in runs}
    expected = {
        (f_value, observed_sigma, extrapolated_sigma)
        for f_value, (observed_sigma, extrapolated_sigma) in itertools.product(
            false_positive_levels,
            width_levels,
        )
    }

    def serialise(key: tuple[float, float, float]) -> dict[str, float]:
        return {
            "false_positive_rate": key[0],
            "observed_reduction_sigma_logit": key[1],
            "extrapolated_reduction_sigma_logit": key[2],
        }

    missing = sorted(expected.difference(observed))
    return {
        "false_positive_rate_levels": false_positive_levels,
        "reduction_prior_width_levels": [
            {
                "observed_reduction_sigma_logit": observed_sigma,
                "extrapolated_reduction_sigma_logit": extrapolated_sigma,
            }
            for observed_sigma, extrapolated_sigma in width_levels
        ],
        "observed_factor_combinations": [serialise(key) for key in sorted(observed)],
        "missing_factor_combinations": [serialise(key) for key in missing],
        "factorial_grid_complete": not missing,
        "incomplete_factorial_grid_allowed": True,
    }


def _save_figure(fig: Any, output_dir: Path, stem: str) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt

    png = output_dir / "plots" / f"{stem}.png"
    svg = output_dir / "plots" / f"{stem}.svg"
    fig.savefig(png, dpi=plot_styles.DPI_FILE, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return png, svg


def _scenario_colours(widths: list[str]) -> dict[str, str]:
    palette = (
        plot_styles.COLOUR_BLUE,
        plot_styles.COLOUR_ORANGE,
        plot_styles.COLOUR_GREEN,
        plot_styles.COLOUR_PURPLE,
        plot_styles.COLOUR_RED,
    )
    return {width: palette[idx % len(palette)] for idx, width in enumerate(widths)}


def _headline_plot(summary: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=plot_styles.FIGSIZE_XL)
    widths = list(summary["reduction_width_id"].drop_duplicates())
    colours = _scenario_colours(widths)
    for width in widths:
        group = summary.loc[summary["reduction_width_id"] == width].sort_values(
            "false_positive_rate"
        )
        x = group["false_positive_rate_per_100k"].to_numpy(dtype=float)
        for ax, mean_column, lo_column, hi_column in (
            (axes[0], "true_ds_mean", "true_ds_lo", "true_ds_hi"),
            (axes[1], "recording_s_mean", "recording_s_lo", "recording_s_hi"),
        ):
            mean = group[mean_column].to_numpy(dtype=float)
            lo = group[lo_column].to_numpy(dtype=float)
            hi = group[hi_column].to_numpy(dtype=float)
            ax.errorbar(
                x,
                mean,
                yerr=np.vstack((mean - lo, hi - mean)),
                fmt="o-",
                capsize=3,
                color=colours[width],
                ecolor=colours[width],
                label=f"rho sigma {width}",
            )
    reference = summary.loc[summary["is_reference"]].iloc[0]
    axes[0].scatter(
        [reference["false_positive_rate_per_100k"]],
        [reference["true_ds_mean"]],
        marker="*",
        s=100,
        color=plot_styles.TEXT_COLOUR,
        zorder=4,
        label="reference",
    )
    axes[1].scatter(
        [reference["false_positive_rate_per_100k"]],
        [reference["recording_s_mean"]],
        marker="*",
        s=100,
        color=plot_styles.TEXT_COLOUR,
        zorder=4,
        label="reference",
    )
    axes[0].set_ylabel("estimated true DS livebirths")
    axes[0].set_title("True DS total across assumptions")
    axes[1].set_ylabel("certificate recording sensitivity")
    axes[1].set_title("Recording sensitivity across assumptions")
    for ax in axes:
        ax.set_xlabel("fixed false-positive probability per 100,000 non-DS births")
        ax.legend(fontsize="small")
    fig.tight_layout()
    return fig


def _ppc_plot(summary: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=plot_styles.FIGSIZE_XL)
    widths = list(summary["reduction_width_id"].drop_duplicates())
    colours = _scenario_colours(widths)
    for width in widths:
        group = summary.loc[summary["reduction_width_id"] == width].sort_values(
            "false_positive_rate"
        )
        x = group["false_positive_rate_per_100k"].to_numpy(dtype=float)
        axes[0].plot(
            x,
            100.0 * group["age_year_coverage_fraction"],
            "o-",
            color=colours[width],
            label=f"rho sigma {width}",
        )
        axes[1].plot(
            x,
            group["age_year_mean_absolute_standardized_residual"],
            "o-",
            color=colours[width],
            label=f"rho sigma {width}",
        )
    axes[0].set_ylabel("age-year cells covered (%)")
    axes[0].set_title("Native age-year PPC coverage")
    axes[1].set_ylabel("mean absolute standardised residual")
    axes[1].set_title("Native age-year PPC residuals")
    for ax in axes:
        ax.set_xlabel("fixed false-positive probability per 100,000 non-DS births")
        ax.legend(fontsize="small")
    fig.tight_layout()
    return fig


def _age_band_residual_plot(age_band: pd.DataFrame):
    import matplotlib.pyplot as plt

    scenarios = list(age_band["scenario_id"].drop_duplicates())
    bands = list(age_band.sort_values("age_band_idx")["label"].drop_duplicates())
    matrix = (
        age_band.pivot(
            index="scenario_id",
            columns="label",
            values="relative_residual",
        )
        .reindex(index=scenarios, columns=bands)
        .to_numpy(dtype=float)
    )
    coverage = (
        age_band.pivot(
            index="scenario_id",
            columns="label",
            values="observed_in_interval",
        )
        .reindex(index=scenarios, columns=bands)
        .to_numpy(dtype=bool)
    )
    finite = np.abs(matrix[np.isfinite(matrix)])
    limit = max(float(finite.max()) if finite.size else 0.0, 0.01)
    height = max(plot_styles.FIGSIZE_MD[1], 0.55 * len(scenarios) + 2.0)
    fig, ax = plt.subplots(
        figsize=(plot_styles.FIGSIZE_XL[0], height),
        layout="constrained",
    )
    image = ax.imshow(
        matrix,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
    )
    colourbar = fig.colorbar(image, ax=ax, pad=0.015)
    colourbar.set_label("(observed - predicted mean) / predicted mean")
    failures = np.argwhere(~coverage)
    if len(failures):
        ax.scatter(
            failures[:, 1],
            failures[:, 0],
            marker="x",
            color=plot_styles.TEXT_COLOUR,
            s=28,
            label="observed outside predictive interval",
        )
        ax.legend(loc="upper left", fontsize="small")
    ax.set_xticks(np.arange(len(bands)))
    ax.set_xticklabels(bands, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(scenarios)))
    ax.set_yticklabels(scenarios)
    ax.set_xlabel("maternal-age band")
    ax.set_ylabel("assumption scenario")
    ax.set_title("Native broad-age posterior-predictive residuals")
    return fig


def compare_core_reduction_sensitivities(
    reference_dir: Path,
    scenario_dirs: list[Path],
    output_dir: Path,
) -> dict[str, Path]:
    """Validate, compare and write a multi-run sensitivity analysis."""
    runs = [load_sensitivity_run(reference_dir, is_reference=True)] + [
        load_sensitivity_run(path) for path in scenario_dirs
    ]
    validate_sensitivity_runs(runs)

    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    scenarios = scenario_table(runs)
    summary = sensitivity_summary_table(runs)
    by_year = sensitivity_by_year_table(runs)
    age_band = sensitivity_age_band_table(runs)
    age_year = sensitivity_age_year_table(runs)
    contrasts = sensitivity_contrast_table(summary)
    envelope = sensitivity_envelope_table(summary)
    table_frames = {
        "scenarios": scenarios,
        "summary": summary,
        "by_year": by_year,
        "age_band": age_band,
        "age_year": age_year,
        "contrasts": contrasts,
        "envelope": envelope,
    }
    paths: dict[str, Path] = {}
    for name, frame in table_frames.items():
        path = tables_dir / f"core_sensitivity_{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path

    for name, figure in (
        ("headlines_plot", _headline_plot(summary)),
        ("ppc_plot", _ppc_plot(summary)),
        ("age_band_residuals_plot", _age_band_residual_plot(age_band)),
    ):
        stem = f"core_sensitivity_{name.removesuffix('_plot')}"
        png, svg = _save_figure(figure, output_dir, stem)
        paths[f"{name}_png"] = png
        paths[f"{name}_svg"] = svg

    grid = _factorial_grid_metadata(runs)
    config = {
        "reference_dir": str(Path(reference_dir)),
        "scenario_dirs": [str(Path(path)) for path in scenario_dirs],
        "model_id": runs[0].config.get("model_id"),
        "allowed_varying_priors": sorted(ALLOWED_VARYING_PRIORS),
        "sampling_budgets_may_differ": True,
        "all_fits_healthy": bool(scenarios["fit_healthy"].all()),
        "reporting_interval_prob": runs[0].interval_prob,
        "false_positive_rate_definition": (
            "fixed probability that a non-DS birth is recorded as DS"
        ),
        "reduction_prior_width_definition": (
            "logit-scale uncertainty around surveillance-derived yearly "
            "combined-reduction means"
        ),
        "axes_are_separate_assumptions": True,
        "raw_loo_or_waic_compared": False,
        "ppc_evidence_scope": (
            "native in-sample posterior-predictive checks; not held-out evidence"
        ),
        "posterior_summary_contrasts_are_not_paired": True,
        "scenario_envelope_is_not_a_posterior_interval": True,
        **grid,
    }
    config_path = output_dir / "sensitivity_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    paths["config"] = config_path
    return paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare saved core-model false-positive/prior-width fits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("reference_dir", type=Path)
    parser.add_argument("scenario_dirs", nargs="+", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory (default: output/selection_core_reduction/"
            "sensitivities/<model>/<timestamp>)."
        ),
    )
    ns = parser.parse_args(argv)
    if ns.output_dir is None:
        reference_config = _read_json(ns.reference_dir / "config.json")
        model_id = str(reference_config.get("model_id", "core"))
        ns.output_dir = (
            Path("output")
            / CORE_REDUCTION_MODEL_ID
            / "sensitivities"
            / model_id
            / datetime.now().strftime("%Y%m%d-%H%M%S")
        )
    return ns


def main(argv: list[str] | None = None) -> int:
    ns = parse_args(argv)
    setup.init_script()
    cli_output.banner(
        "compare_core_reduction_sensitivities",
        f"reference={ns.reference_dir}",
    )
    paths = compare_core_reduction_sensitivities(
        ns.reference_dir,
        ns.scenario_dirs,
        ns.output_dir,
    )
    cli_output.success(f"sensitivity tables and plots -> {ns.output_dir}")
    cli_output.print_kv("Outputs", paths.items())
    return 0


if __name__ == "__main__":
    sys.exit(main())
