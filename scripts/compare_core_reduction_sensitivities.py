"""Compare a saved multi-run core-model assumption-sensitivity grid.

The comparison is deliberately table-driven.  Runs of the same core model use
the same native age-by-year cells, so their checked-in reporting contracts are
more direct than reconstructing another posterior-predictive sample from each
NetCDF file.  The NetCDF is also used for sampler divergences and a small set
of explicitly draw-wise dependence and missed-case diagnostics.

The intended sensitivity axes are the fixed false-positive probability ``f``,
the surveillance-derived reduction-prior widths, the correlation between
yearly reduction-prior errors, and a fixed logit-scale calibration shift.
Incomplete factorial grids are allowed and are recorded explicitly in the
output config.
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

ALLOWED_VARYING_PRIORS = frozenset(
    {
        "false_positive_rate",
        "reduction_sigma",
        "reduction_error_correlation",
        "reduction_calibration_shift_logit",
    }
)
TOTAL_MEAN_MATERIAL_PERCENT = 5.0
TOTAL_INTERVAL_WIDTH_MATERIAL_PERCENT = 25.0
RECORDING_S_MATERIAL_ABSOLUTE = 0.05
MISSED_TRUE_CASES_MEAN_MATERIAL_PERCENT = 10.0
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
    "true_ds_interval_width",
    "recording_s_mean",
    "recording_s_interval_width",
    "aggregate_reduction_mean",
    "model_implied_expected_missed_true_cases_mean",
    "model_implied_expected_missed_true_cases_interval_width",
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
    "model_implied_expected_missed_true_cases": (
        "model_implied_expected_missed_true_cases_mean",
        "model_implied_expected_missed_true_cases_lo",
        "model_implied_expected_missed_true_cases_hi",
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
    reduction_error_correlation: float
    reduction_calibration_shift_logit: float
    interval_prob: float
    max_rhat: float
    min_ess: float
    convergence_ok: bool
    divergences: int | None
    standardized_trajectory_error_index_recording_s_correlation: float | None
    standardized_trajectory_error_index_recording_s_logit_correlation: float | None
    model_implied_expected_missed_true_cases_mean: float | None
    model_implied_expected_missed_true_cases_lo: float | None
    model_implied_expected_missed_true_cases_hi: float | None
    model_implied_expected_missed_true_cases_method: str | None
    model_implied_expected_missed_true_cases_unavailable_reason: str | None

    @property
    def reduction_width_id(self) -> str:
        """Human-readable identifier for one prior-width regime."""
        return (
            f"{self.observed_reduction_sigma:.8g}/"
            f"{self.extrapolated_reduction_sigma:.8g}"
        )

    @property
    def calibration_id(self) -> str:
        """Human-readable identifier for one coherent-calibration regime."""
        return (
            f"corr={self.reduction_error_correlation:.8g}; "
            f"shift={self.reduction_calibration_shift_logit:+.8g}"
        )

    @property
    def factor_key(self) -> tuple[float, float, float, float, float]:
        """Stable numeric key for duplicate-scenario checks."""
        return (
            self.false_positive_rate,
            self.observed_reduction_sigma,
            self.extrapolated_reduction_sigma,
            self.reduction_error_correlation,
            self.reduction_calibration_shift_logit,
        )


@dataclass(frozen=True)
class PosteriorSensitivityDiagnostics:
    """Draw-wise quantities derived from an available posterior group."""

    trajectory_index_recording_s_correlation: float | None = None
    trajectory_index_recording_s_logit_correlation: float | None = None
    missed_true_cases_mean: float | None = None
    missed_true_cases_lo: float | None = None
    missed_true_cases_hi: float | None = None
    missed_true_cases_method: str | None = None
    missed_true_cases_unavailable_reason: str | None = None


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


def _coherent_calibration_factors(config: dict[str, Any]) -> tuple[float, float]:
    """Extract coherent-calibration factors, defaulting legacy runs to zero."""
    priors = config.get("priors")
    if not isinstance(priors, dict):
        raise ValueError("config.priors must be a JSON object.")
    try:
        correlation = float(priors.get("reduction_error_correlation", 0.0))
        shift = float(priors.get("reduction_calibration_shift_logit", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Coherent reduction-calibration factors must be numeric."
        ) from exc
    if not np.isfinite(correlation) or not 0.0 <= correlation < 1.0:
        raise ValueError("priors.reduction_error_correlation must lie in [0, 1).")
    if not np.isfinite(shift):
        raise ValueError("priors.reduction_calibration_shift_logit must be finite.")
    return correlation, shift


def _reduction_prior_location_and_scale(
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return shifted yearly logit locations and marginal standard deviations."""
    try:
        priors = config["priors"]
        location = np.asarray(priors["reduction_logit"], dtype=float)
        scale = np.asarray(priors["reduction_sigma"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "config must contain valid yearly reduction_logit and reduction_sigma."
        ) from exc
    _, shift = _coherent_calibration_factors(config)
    if location.ndim != 1 or scale.ndim != 1 or len(location) != len(scale):
        raise ValueError(
            "priors.reduction_logit and reduction_sigma must be same-length vectors."
        )
    if not len(location) or not np.all(np.isfinite(location)):
        raise ValueError("priors.reduction_logit values must be finite.")
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("priors.reduction_sigma values must be finite and positive.")
    return location + shift, scale


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


def _posterior_scalar_samples(array: xr.DataArray, *, name: str) -> np.ndarray:
    """Return one scalar posterior quantity per chain-draw sample."""
    sample_dims = {"chain", "draw"}
    if not sample_dims.issubset(array.dims):
        raise ValueError(f"Posterior variable {name!r} has no chain/draw dimensions.")
    other_dims = [dim for dim in array.dims if dim not in sample_dims]
    if other_dims:
        raise ValueError(f"Posterior variable {name!r} is not scalar per draw.")
    return np.asarray(array.transpose("chain", "draw").values, dtype=float).reshape(-1)


def _posterior_year_samples(array: xr.DataArray, *, name: str) -> np.ndarray:
    """Return a sample-by-year matrix for one posterior trajectory."""
    sample_dims = {"chain", "draw"}
    if not sample_dims.issubset(array.dims):
        raise ValueError(f"Posterior variable {name!r} has no chain/draw dimensions.")
    other_dims = [dim for dim in array.dims if dim not in sample_dims]
    if len(other_dims) != 1:
        raise ValueError(
            f"Posterior variable {name!r} must have exactly one trajectory dimension."
        )
    values = np.asarray(
        array.transpose("chain", "draw", other_dims[0]).values,
        dtype=float,
    )
    return values.reshape(-1, values.shape[-1])


def _finite_draw_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Return a finite-draw Pearson correlation, or NaN when degenerate."""
    if left.shape != right.shape:
        raise ValueError("Posterior quantities have incompatible draw counts.")
    finite = np.isfinite(left) & np.isfinite(right)
    left_finite = left[finite]
    right_finite = right[finite]
    if (
        left_finite.size < 2
        or np.ptp(left_finite) == 0.0
        or np.ptp(right_finite) == 0.0
    ):
        return float("nan")
    return float(np.corrcoef(left_finite, right_finite)[0, 1])


def _mean_eti(draws: np.ndarray, *, interval_prob: float) -> tuple[float, float, float]:
    """Summarise finite posterior draws with a mean and equal-tailed interval."""
    finite = np.asarray(draws, dtype=float).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return float("nan"), float("nan"), float("nan")
    tail = (1.0 - interval_prob) / 2.0
    lo, hi = np.quantile(finite, [tail, 1.0 - tail])
    return float(np.mean(finite)), float(lo), float(hi)


def _posterior_sensitivity_diagnostics(
    path: Path,
    *,
    config: dict[str, Any],
    interval_prob: float,
) -> PosteriorSensitivityDiagnostics:
    """Derive trajectory-dependence and missed-case summaries when available."""
    try:
        posterior_context = xr.open_dataset(path, group="posterior")
    except OSError, ValueError:
        return PosteriorSensitivityDiagnostics(
            missed_true_cases_unavailable_reason="posterior group is unavailable"
        )

    with posterior_context as posterior:
        trajectory_index: np.ndarray | None = None
        if "rho_logit_year" in posterior:
            rho_logit = _posterior_year_samples(
                posterior["rho_logit_year"], name="rho_logit_year"
            )
            location, scale = _reduction_prior_location_and_scale(config)
            if rho_logit.shape[1] != len(location):
                raise ValueError(
                    "Posterior rho_logit_year and reduction prior have different "
                    "year counts."
                )
            trajectory_index = np.mean(
                (rho_logit - location[None, :]) / scale[None, :],
                axis=1,
            )

        recording_correlation: float | None = None
        if trajectory_index is not None and "recording_s" in posterior:
            recording_s = _posterior_scalar_samples(
                posterior["recording_s"], name="recording_s"
            )
            recording_correlation = _finite_draw_correlation(
                trajectory_index, recording_s
            )

        recording_logit_correlation: float | None = None
        if trajectory_index is not None and "recording_s_logit" in posterior:
            recording_s_logit = _posterior_scalar_samples(
                posterior["recording_s_logit"], name="recording_s_logit"
            )
            recording_logit_correlation = _finite_draw_correlation(
                trajectory_index, recording_s_logit
            )

        missed_draws: np.ndarray | None = None
        missed_method: str | None = None
        missed_unavailable_reason: str | None = None
        if "true_count_year" in posterior and "recording_s_year" in posterior:
            true_year = _posterior_year_samples(
                posterior["true_count_year"], name="true_count_year"
            )
            recording_year = _posterior_year_samples(
                posterior["recording_s_year"], name="recording_s_year"
            )
            if true_year.shape != recording_year.shape:
                raise ValueError(
                    "Posterior true_count_year and recording_s_year have "
                    "incompatible shapes."
                )
            missed_draws = np.sum(true_year * (1.0 - recording_year), axis=1)
            missed_method = "sum_y true_count_year * (1 - recording_s_year)"
        elif (
            config.get("recording_model") == "constant"
            and "true_count_total" in posterior
            and "recording_s" in posterior
        ):
            true_total = _posterior_scalar_samples(
                posterior["true_count_total"], name="true_count_total"
            )
            recording_s = _posterior_scalar_samples(
                posterior["recording_s"], name="recording_s"
            )
            if true_total.shape != recording_s.shape:
                raise ValueError(
                    "Posterior true_count_total and recording_s have incompatible "
                    "draw counts."
                )
            missed_draws = true_total * (1.0 - recording_s)
            missed_method = "true_count_total * (1 - recording_s)"
        elif config.get("recording_model") != "constant":
            missed_unavailable_reason = (
                f"recording_model={config.get('recording_model')!r} requires "
                "posterior true_count_year and recording_s_year arrays"
            )
        else:
            missed_unavailable_reason = (
                "posterior true_count_total and recording_s are unavailable"
            )

        missed_mean: float | None = None
        missed_lo: float | None = None
        missed_hi: float | None = None
        if missed_draws is not None:
            missed_mean, missed_lo, missed_hi = _mean_eti(
                missed_draws,
                interval_prob=interval_prob,
            )
            if not all(np.isfinite((missed_mean, missed_lo, missed_hi))):
                missed_unavailable_reason = (
                    "model-implied expected missed true cases summary is non-finite"
                )

    return PosteriorSensitivityDiagnostics(
        trajectory_index_recording_s_correlation=recording_correlation,
        trajectory_index_recording_s_logit_correlation=(recording_logit_correlation),
        missed_true_cases_mean=missed_mean,
        missed_true_cases_lo=missed_lo,
        missed_true_cases_hi=missed_hi,
        missed_true_cases_method=missed_method,
        missed_true_cases_unavailable_reason=missed_unavailable_reason,
    )


def _scenario_id(
    false_positive_rate: float,
    observed: float,
    extrapolated: float,
    reduction_error_correlation: float,
    reduction_calibration_shift_logit: float,
) -> str:
    return (
        f"f={false_positive_rate:.8g}; "
        f"rho_sigma={observed:.8g}/{extrapolated:.8g}; "
        f"rho_corr={reduction_error_correlation:.8g}; "
        f"rho_shift_logit={reduction_calibration_shift_logit:+.8g}"
    )


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
    reduction_error_correlation, reduction_calibration_shift_logit = (
        _coherent_calibration_factors(config)
    )
    interval_prob = _report_interval_probability(tables)
    health = diagnostics.convergence_health(summary)
    idata_path = run_dir / "idata.nc"
    divergences = _count_divergences(idata_path)
    posterior_diagnostics = _posterior_sensitivity_diagnostics(
        idata_path,
        config=config,
        interval_prob=interval_prob,
    )

    return CoreSensitivityRun(
        run_dir=run_dir,
        scenario_id=_scenario_id(
            false_positive_rate,
            observed_sigma,
            extrapolated_sigma,
            reduction_error_correlation,
            reduction_calibration_shift_logit,
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
        reduction_error_correlation=reduction_error_correlation,
        reduction_calibration_shift_logit=reduction_calibration_shift_logit,
        interval_prob=interval_prob,
        max_rhat=float(health["max_rhat"]),
        min_ess=float(health["min_ess"]),
        convergence_ok=bool(health["all_ok"]),
        divergences=divergences,
        standardized_trajectory_error_index_recording_s_correlation=(
            posterior_diagnostics.trajectory_index_recording_s_correlation
        ),
        standardized_trajectory_error_index_recording_s_logit_correlation=(
            posterior_diagnostics.trajectory_index_recording_s_logit_correlation
        ),
        model_implied_expected_missed_true_cases_mean=(
            posterior_diagnostics.missed_true_cases_mean
        ),
        model_implied_expected_missed_true_cases_lo=(
            posterior_diagnostics.missed_true_cases_lo
        ),
        model_implied_expected_missed_true_cases_hi=(
            posterior_diagnostics.missed_true_cases_hi
        ),
        model_implied_expected_missed_true_cases_method=(
            posterior_diagnostics.missed_true_cases_method
        ),
        model_implied_expected_missed_true_cases_unavailable_reason=(
            posterior_diagnostics.missed_true_cases_unavailable_reason
        ),
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
    """Require matched models and allow only the intended sensitivity axes."""
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
            "Each sensitivity-factor combination must appear at most once."
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
                f"{run.run_dir} changes a prior other than an allowed sensitivity "
                "factor."
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


def _fit_healthy(run: CoreSensitivityRun) -> bool:
    """Return the pre-specified sampler-health gate for one run."""
    return bool(run.convergence_ok and run.divergences == 0)


def _is_finite_optional(value: float | None) -> bool:
    """Return whether an optional scalar is present and finite."""
    return value is not None and bool(np.isfinite(value))


def _materiality_health_status(
    *,
    reference_healthy: bool,
    candidate_healthy: bool,
    candidate_is_reference: bool = False,
) -> tuple[bool, str | None]:
    """Fail materiality evaluation closed when either fitted run is unhealthy."""
    if reference_healthy and candidate_healthy:
        return True, None
    if candidate_is_reference:
        return False, "reference fit is unhealthy"
    if not reference_healthy and not candidate_healthy:
        return False, "reference and candidate fits are unhealthy"
    if not reference_healthy:
        return False, "reference fit is unhealthy"
    return False, "candidate fit is unhealthy"


def _metric_materiality_status(
    frame: pd.DataFrame,
    *,
    availability_column: str,
    metric_label: str,
) -> tuple[pd.Series, pd.Series]:
    """Return scenario-vs-reference evaluation status and reason for one metric."""
    reference = frame.loc[frame["is_reference"]].iloc[0]
    reference_available = bool(reference[availability_column])
    evaluated: list[bool] = []
    reasons: list[str | None] = []
    for row in frame.itertuples(index=False):
        health_evaluated = bool(row.materiality_evaluated_against_reference)
        candidate_available = bool(getattr(row, availability_column))
        if not health_evaluated:
            evaluated.append(False)
            reasons.append(row.materiality_non_evaluation_reason)
        elif reference_available and candidate_available:
            evaluated.append(True)
            reasons.append(None)
        elif bool(row.is_reference) or (
            not reference_available and candidate_available
        ):
            evaluated.append(False)
            reasons.append(f"reference {metric_label} is unavailable")
        elif reference_available and not candidate_available:
            evaluated.append(False)
            reasons.append(f"candidate {metric_label} is unavailable")
        else:
            evaluated.append(False)
            reasons.append(
                f"reference and candidate {metric_label} summaries are unavailable"
            )
    return (
        pd.Series(evaluated, index=frame.index, dtype=bool),
        pd.Series(reasons, index=frame.index, dtype=object),
    )


def _nullable_materiality_result(
    evaluated: pd.Series,
    condition: pd.Series,
) -> pd.Series:
    """Return a nullable Boolean materiality decision."""
    result = pd.Series(pd.NA, index=evaluated.index, dtype="boolean")
    result.loc[evaluated] = condition.loc[evaluated].astype(bool)
    return result


def _combine_decomposition_materiality(
    recording_result: pd.Series,
    missed_result: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Combine decomposition components without treating unknown as false."""
    components = pd.concat([recording_result, missed_result], axis=1)
    any_material = components.fillna(False).astype(bool).any(axis=1)
    all_evaluated = components.notna().all(axis=1)
    evaluated = any_material | all_evaluated
    result = pd.Series(pd.NA, index=components.index, dtype="boolean")
    result.loc[any_material] = True
    result.loc[~any_material & all_evaluated] = False
    return result, evaluated


def scenario_table(runs: list[CoreSensitivityRun]) -> pd.DataFrame:
    """Return run provenance, factor levels, and sampler-health diagnostics."""
    rows = []
    reference = next(run for run in runs if run.is_reference)
    reference_fit_healthy = _fit_healthy(reference)
    for order, run in enumerate(runs):
        cfg = run.run_config
        no_divergences = run.divergences == 0 if run.divergences is not None else False
        fit_healthy = _fit_healthy(run)
        materiality_evaluated, materiality_reason = _materiality_health_status(
            reference_healthy=reference_fit_healthy,
            candidate_healthy=fit_healthy,
            candidate_is_reference=run.is_reference,
        )
        recording_correlation_available = _is_finite_optional(
            run.standardized_trajectory_error_index_recording_s_correlation
        )
        recording_logit_correlation_available = _is_finite_optional(
            run.standardized_trajectory_error_index_recording_s_logit_correlation
        )
        missed_cases_available = all(
            _is_finite_optional(value)
            for value in (
                run.model_implied_expected_missed_true_cases_mean,
                run.model_implied_expected_missed_true_cases_lo,
                run.model_implied_expected_missed_true_cases_hi,
            )
        )
        missed_cases_reason = (
            None
            if missed_cases_available
            else run.model_implied_expected_missed_true_cases_unavailable_reason
            or "model-implied expected missed true cases summary is non-finite"
        )
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
                "reduction_error_correlation": run.reduction_error_correlation,
                "reduction_calibration_shift_logit": (
                    run.reduction_calibration_shift_logit
                ),
                "reduction_calibration_shift_odds_multiplier": float(
                    np.exp(run.reduction_calibration_shift_logit)
                ),
                "calibration_id": run.calibration_id,
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
                "fit_healthy": fit_healthy,
                "reference_fit_healthy": reference_fit_healthy,
                "materiality_evaluated_against_reference": materiality_evaluated,
                "materiality_non_evaluation_reason": materiality_reason,
                "standardized_trajectory_error_index_available": (
                    recording_correlation_available
                    or recording_logit_correlation_available
                ),
                "standardized_trajectory_error_index_recording_s_correlation_available": (
                    recording_correlation_available
                ),
                "standardized_trajectory_error_index_recording_s_logit_correlation_available": (
                    recording_logit_correlation_available
                ),
                "standardized_trajectory_error_index_recording_s_correlation": (
                    run.standardized_trajectory_error_index_recording_s_correlation
                ),
                "standardized_trajectory_error_index_recording_s_logit_correlation": (
                    run.standardized_trajectory_error_index_recording_s_logit_correlation
                ),
                "model_implied_expected_missed_true_cases_available": (
                    missed_cases_available
                ),
                "model_implied_expected_missed_true_cases_method": (
                    run.model_implied_expected_missed_true_cases_method
                ),
                "model_implied_expected_missed_true_cases_unavailable_reason": (
                    missed_cases_reason
                ),
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
        fit_healthy = _fit_healthy(run)
        recording_correlation_available = _is_finite_optional(
            run.standardized_trajectory_error_index_recording_s_correlation
        )
        recording_logit_correlation_available = _is_finite_optional(
            run.standardized_trajectory_error_index_recording_s_logit_correlation
        )
        missed_cases_available = all(
            _is_finite_optional(value)
            for value in (
                run.model_implied_expected_missed_true_cases_mean,
                run.model_implied_expected_missed_true_cases_lo,
                run.model_implied_expected_missed_true_cases_hi,
            )
        )
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
        missed_mean = (
            float(run.model_implied_expected_missed_true_cases_mean)
            if run.model_implied_expected_missed_true_cases_mean is not None
            else np.nan
        )
        missed_lo = (
            float(run.model_implied_expected_missed_true_cases_lo)
            if run.model_implied_expected_missed_true_cases_lo is not None
            else np.nan
        )
        missed_hi = (
            float(run.model_implied_expected_missed_true_cases_hi)
            if run.model_implied_expected_missed_true_cases_hi is not None
            else np.nan
        )

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
                "fit_healthy": fit_healthy,
                "false_positive_rate": f,
                "false_positive_rate_per_100k": f * 100_000.0,
                "observed_reduction_sigma_logit": run.observed_reduction_sigma,
                "extrapolated_reduction_sigma_logit": (
                    run.extrapolated_reduction_sigma
                ),
                "reduction_width_id": run.reduction_width_id,
                "reduction_error_correlation": run.reduction_error_correlation,
                "reduction_calibration_shift_logit": (
                    run.reduction_calibration_shift_logit
                ),
                "reduction_calibration_shift_odds_multiplier": float(
                    np.exp(run.reduction_calibration_shift_logit)
                ),
                "calibration_id": run.calibration_id,
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
                "standardized_trajectory_error_index_recording_s_correlation": (
                    run.standardized_trajectory_error_index_recording_s_correlation
                ),
                "standardized_trajectory_error_index_recording_s_correlation_available": (
                    recording_correlation_available
                ),
                "standardized_trajectory_error_index_recording_s_logit_correlation": (
                    run.standardized_trajectory_error_index_recording_s_logit_correlation
                ),
                "standardized_trajectory_error_index_recording_s_logit_correlation_available": (
                    recording_logit_correlation_available
                ),
                "standardized_trajectory_error_index_available": (
                    recording_correlation_available
                    or recording_logit_correlation_available
                ),
                "model_implied_expected_missed_true_cases_mean": missed_mean,
                "model_implied_expected_missed_true_cases_lo": missed_lo,
                "model_implied_expected_missed_true_cases_hi": missed_hi,
                "model_implied_expected_missed_true_cases_interval_width": (
                    missed_hi - missed_lo
                ),
                "model_implied_expected_missed_true_cases_method": (
                    run.model_implied_expected_missed_true_cases_method
                ),
                "model_implied_expected_missed_true_cases_available": (
                    missed_cases_available
                ),
                "model_implied_expected_missed_true_cases_unavailable_reason": (
                    None
                    if missed_cases_available
                    else run.model_implied_expected_missed_true_cases_unavailable_reason
                    or "model-implied expected missed true cases summary is non-finite"
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
    reference_fit_healthy = bool(reference["fit_healthy"])
    materiality_status = [
        _materiality_health_status(
            reference_healthy=reference_fit_healthy,
            candidate_healthy=bool(row.fit_healthy),
            candidate_is_reference=bool(row.is_reference),
        )
        for row in out.itertuples(index=False)
    ]
    out["reference_fit_healthy"] = reference_fit_healthy
    out["materiality_evaluated_against_reference"] = [
        evaluated for evaluated, _ in materiality_status
    ]
    out["materiality_non_evaluation_reason"] = [
        reason for _, reason in materiality_status
    ]
    out["recording_s_mean_available"] = np.isfinite(out["recording_s_mean"])
    (
        out["recording_s_mean_materiality_evaluated_from_reference"],
        out["recording_s_mean_materiality_non_evaluation_reason"],
    ) = _metric_materiality_status(
        out,
        availability_column="recording_s_mean_available",
        metric_label="recording-sensitivity mean",
    )
    (
        out[
            "model_implied_expected_missed_true_cases_mean_materiality_evaluated_from_reference"
        ],
        out[
            "model_implied_expected_missed_true_cases_mean_materiality_non_evaluation_reason"
        ],
    ) = _metric_materiality_status(
        out,
        availability_column="model_implied_expected_missed_true_cases_available",
        metric_label="model-implied expected missed true cases",
    )
    delta_columns = (
        "true_ds_mean",
        "true_ds_interval_width",
        "aggregate_reduction_mean",
        "recording_s_mean",
        "recording_s_interval_width",
        "model_implied_expected_missed_true_cases_mean",
        "model_implied_expected_missed_true_cases_interval_width",
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
    out["true_ds_interval_width_percent_difference_from_reference"] = (
        100.0
        * out["true_ds_interval_width_difference_from_reference"]
        / reference["true_ds_interval_width"]
    )
    missed_reference = reference["model_implied_expected_missed_true_cases_mean"]
    out[
        "model_implied_expected_missed_true_cases_mean_percent_difference_from_reference"
    ] = (
        100.0
        * out["model_implied_expected_missed_true_cases_mean_difference_from_reference"]
        / missed_reference
    )
    out["true_ds_mean_material_change_from_reference"] = out[
        "materiality_evaluated_against_reference"
    ] & (
        out["true_ds_mean_percent_difference_from_reference"].abs()
        >= TOTAL_MEAN_MATERIAL_PERCENT
    )
    out["true_ds_interval_width_material_change_from_reference"] = out[
        "materiality_evaluated_against_reference"
    ] & (
        out["true_ds_interval_width_percent_difference_from_reference"]
        >= TOTAL_INTERVAL_WIDTH_MATERIAL_PERCENT
    )
    out["recording_s_mean_material_change_from_reference"] = (
        _nullable_materiality_result(
            out["recording_s_mean_materiality_evaluated_from_reference"],
            out["recording_s_mean_difference_from_reference"].abs()
            >= RECORDING_S_MATERIAL_ABSOLUTE,
        )
    )
    out[
        "model_implied_expected_missed_true_cases_mean_material_change_from_reference"
    ] = _nullable_materiality_result(
        out[
            "model_implied_expected_missed_true_cases_mean_materiality_evaluated_from_reference"
        ],
        out[
            "model_implied_expected_missed_true_cases_mean_percent_difference_from_reference"
        ].abs()
        >= MISSED_TRUE_CASES_MEAN_MATERIAL_PERCENT,
    )
    out["aggregate_material_change_from_reference"] = out[
        [
            "true_ds_mean_material_change_from_reference",
            "true_ds_interval_width_material_change_from_reference",
        ]
    ].any(axis=1)
    (
        out["decomposition_material_change_from_reference"],
        out["decomposition_materiality_evaluated_from_reference"],
    ) = _combine_decomposition_materiality(
        out["recording_s_mean_material_change_from_reference"],
        out[
            "model_implied_expected_missed_true_cases_mean_material_change_from_reference"
        ],
    )
    decomposition_reasons: list[str | None] = []
    for row in out.itertuples(index=False):
        if bool(row.decomposition_materiality_evaluated_from_reference):
            decomposition_reasons.append(None)
            continue
        component_reasons = [
            row.recording_s_mean_materiality_non_evaluation_reason,
            row.model_implied_expected_missed_true_cases_mean_materiality_non_evaluation_reason,
        ]
        unique_reasons = list(
            dict.fromkeys(reason for reason in component_reasons if reason)
        )
        decomposition_reasons.append("; ".join(unique_reasons))
    out["decomposition_materiality_non_evaluation_reason"] = decomposition_reasons
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
        "reduction_error_correlation": run.reduction_error_correlation,
        "reduction_calibration_shift_logit": (run.reduction_calibration_shift_logit),
        "reduction_calibration_shift_odds_multiplier": float(
            np.exp(run.reduction_calibration_shift_logit)
        ),
        "calibration_id": run.calibration_id,
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
                "rho_prior_centre",
                "rho_prior_location_logit",
                "rho_prior_lo",
                "rho_prior_hi",
                "rho_prior_sigma_logit",
                "rho_surveillance_anchor_mean",
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


def _contrast_materiality_non_evaluation_reason(
    *,
    materiality_rule: str | None,
    values_finite: bool,
    reference_contrast: bool,
    from_fit_healthy: bool,
    to_fit_healthy: bool,
) -> str | None:
    """Explain why a contrast's materiality rule was not evaluated."""
    if materiality_rule is None:
        return "metric has no pre-specified materiality rule"
    if not reference_contrast:
        return "not a scenario-versus-reference contrast"
    if not from_fit_healthy and not to_fit_healthy:
        return "reference and candidate fits are unhealthy"
    if not from_fit_healthy:
        return "reference fit is unhealthy"
    if not to_fit_healthy:
        return "candidate fit is unhealthy"
    if not values_finite:
        return "metric values are non-finite"
    return None


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
            "reduction_error_correlation",
            "reduction_calibration_shift_logit",
        ]
    )
    for (_, from_row), (_, to_row) in itertools.combinations(ordered.iterrows(), 2):
        if bool(to_row["is_reference"]) and not bool(from_row["is_reference"]):
            from_row, to_row = to_row, from_row
        reference_contrast = bool(from_row["is_reference"])
        from_fit_healthy = bool(from_row["fit_healthy"])
        to_fit_healthy = bool(to_row["fit_healthy"])
        both_fits_healthy = from_fit_healthy and to_fit_healthy
        pair_materiality_evaluated = reference_contrast and both_fits_healthy
        for metric in CONTRAST_METRICS:
            from_value = float(from_row[metric])
            to_value = float(to_row[metric])
            difference = to_value - from_value
            percent_difference = (
                100.0 * difference / from_value
                if np.isfinite(from_value) and from_value != 0.0
                else np.nan
            )
            materiality_rule: str | None = None
            materiality_threshold: float = np.nan
            material_change = False
            if metric == "true_ds_mean":
                materiality_rule = "absolute percent difference"
                materiality_threshold = TOTAL_MEAN_MATERIAL_PERCENT
                material_change = bool(
                    pair_materiality_evaluated
                    and np.isfinite(percent_difference)
                    and abs(percent_difference) >= materiality_threshold
                )
            elif metric == "true_ds_interval_width":
                materiality_rule = "percent increase"
                materiality_threshold = TOTAL_INTERVAL_WIDTH_MATERIAL_PERCENT
                material_change = bool(
                    pair_materiality_evaluated
                    and np.isfinite(percent_difference)
                    and percent_difference >= materiality_threshold
                )
            elif metric == "recording_s_mean":
                materiality_rule = "absolute difference"
                materiality_threshold = RECORDING_S_MATERIAL_ABSOLUTE
                material_change = bool(
                    pair_materiality_evaluated
                    and np.isfinite(difference)
                    and abs(difference) >= materiality_threshold
                )
            elif metric == "model_implied_expected_missed_true_cases_mean":
                materiality_rule = "absolute percent difference"
                materiality_threshold = MISSED_TRUE_CASES_MEAN_MATERIAL_PERCENT
                material_change = bool(
                    pair_materiality_evaluated
                    and np.isfinite(percent_difference)
                    and abs(percent_difference) >= materiality_threshold
                )
            values_finite = bool(np.isfinite(from_value) and np.isfinite(to_value))
            materiality_evaluated = bool(
                pair_materiality_evaluated
                and materiality_rule is not None
                and values_finite
            )
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
                    "from_reduction_error_correlation": from_row[
                        "reduction_error_correlation"
                    ],
                    "to_reduction_error_correlation": to_row[
                        "reduction_error_correlation"
                    ],
                    "from_reduction_calibration_shift_logit": from_row[
                        "reduction_calibration_shift_logit"
                    ],
                    "to_reduction_calibration_shift_logit": to_row[
                        "reduction_calibration_shift_logit"
                    ],
                    "reference_contrast": reference_contrast,
                    "from_fit_healthy": from_fit_healthy,
                    "to_fit_healthy": to_fit_healthy,
                    "both_fits_healthy": both_fits_healthy,
                    "metric": metric,
                    "from_value": from_value,
                    "to_value": to_value,
                    "difference": difference,
                    "percent_difference": percent_difference,
                    "materiality_rule": materiality_rule,
                    "materiality_threshold": materiality_threshold,
                    "materiality_evaluated": materiality_evaluated,
                    "materiality_non_evaluation_reason": (
                        _contrast_materiality_non_evaluation_reason(
                            materiality_rule=materiality_rule,
                            values_finite=values_finite,
                            reference_contrast=reference_contrast,
                            from_fit_healthy=from_fit_healthy,
                            to_fit_healthy=to_fit_healthy,
                        )
                    ),
                    "material_change": material_change,
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
    factors = (
        ("false_positive_rate", "false_positive_rate"),
        ("reduction_prior_width", "reduction_width_id"),
        ("reduction_error_correlation", "reduction_error_correlation"),
        (
            "reduction_calibration_shift_logit",
            "reduction_calibration_shift_logit",
        ),
    )
    for varied_name, varied_column in factors:
        held = [(name, column) for name, column in factors if name != varied_name]
        held_columns = [column for _, column in held]
        for held_values, group in summary.groupby(
            held_columns,
            sort=False,
            dropna=False,
        ):
            if group[varied_column].nunique() <= 1:
                continue
            if not isinstance(held_values, tuple):
                held_values = (held_values,)
            held_factor = "; ".join(
                f"{name}={value}"
                for (name, _), value in zip(held, held_values, strict=True)
            )
            rows.extend(
                _contrast_rows(
                    group,
                    varied_factor=varied_name,
                    held_factor=held_factor,
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
        "from_reduction_error_correlation",
        "to_reduction_error_correlation",
        "from_reduction_calibration_shift_logit",
        "to_reduction_calibration_shift_logit",
        "reference_contrast",
        "from_fit_healthy",
        "to_fit_healthy",
        "both_fits_healthy",
        "metric",
        "from_value",
        "to_value",
        "difference",
        "percent_difference",
        "materiality_rule",
        "materiality_threshold",
        "materiality_evaluated",
        "materiality_non_evaluation_reason",
        "material_change",
        "contrast_scope",
    ]
    return pd.DataFrame(rows, columns=columns)


def sensitivity_envelope_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Return the range across assumptions without calling it a posterior interval."""
    rows = []
    for metric, (mean_column, lo_column, hi_column) in ENVELOPE_METRICS.items():
        finite_mean = summary.loc[np.isfinite(summary[mean_column]), mean_column]
        if finite_mean.empty:
            minimum_idx = None
            maximum_idx = None
        else:
            minimum_idx = finite_mean.idxmin()
            maximum_idx = finite_mean.idxmax()
        rows.append(
            {
                "metric": metric,
                "minimum_mean": (
                    float(summary.loc[minimum_idx, mean_column])
                    if minimum_idx is not None
                    else np.nan
                ),
                "minimum_mean_scenario_id": (
                    summary.loc[minimum_idx, "scenario_id"]
                    if minimum_idx is not None
                    else None
                ),
                "maximum_mean": (
                    float(summary.loc[maximum_idx, mean_column])
                    if maximum_idx is not None
                    else np.nan
                ),
                "maximum_mean_scenario_id": (
                    summary.loc[maximum_idx, "scenario_id"]
                    if maximum_idx is not None
                    else None
                ),
                "scenario_mean_span": (
                    float(
                        summary.loc[maximum_idx, mean_column]
                        - summary.loc[minimum_idx, mean_column]
                    )
                    if minimum_idx is not None and maximum_idx is not None
                    else np.nan
                ),
                "envelope_lo": (
                    float(summary[lo_column].min())
                    if lo_column is not None and np.isfinite(summary[lo_column]).any()
                    else np.nan
                ),
                "envelope_hi": (
                    float(summary[hi_column].max())
                    if hi_column is not None and np.isfinite(summary[hi_column]).any()
                    else np.nan
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
    correlation_levels = sorted({run.reduction_error_correlation for run in runs})
    shift_levels = sorted({run.reduction_calibration_shift_logit for run in runs})
    observed = {run.factor_key for run in runs}
    expected = {
        (
            f_value,
            observed_sigma,
            extrapolated_sigma,
            correlation,
            shift,
        )
        for (
            f_value,
            (observed_sigma, extrapolated_sigma),
            correlation,
            shift,
        ) in itertools.product(
            false_positive_levels,
            width_levels,
            correlation_levels,
            shift_levels,
        )
    }

    def serialise(
        key: tuple[float, float, float, float, float],
    ) -> dict[str, float]:
        return {
            "false_positive_rate": key[0],
            "observed_reduction_sigma_logit": key[1],
            "extrapolated_reduction_sigma_logit": key[2],
            "reduction_error_correlation": key[3],
            "reduction_calibration_shift_logit": key[4],
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
        "reduction_error_correlation_levels": correlation_levels,
        "reduction_calibration_shift_logit_levels": shift_levels,
        "factor_names": [
            "false_positive_rate",
            "reduction_prior_width",
            "reduction_error_correlation",
            "reduction_calibration_shift_logit",
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


def _scenario_colours(scenario_ids: list[str]) -> dict[str, str]:
    palette = (
        plot_styles.COLOUR_BLUE,
        plot_styles.COLOUR_ORANGE,
        plot_styles.COLOUR_GREEN,
        plot_styles.COLOUR_PURPLE,
        plot_styles.COLOUR_RED,
    )
    return {
        scenario_id: palette[idx % len(palette)]
        for idx, scenario_id in enumerate(scenario_ids)
    }


def _categorical_scenario_labels(summary: pd.DataFrame) -> list[str]:
    """Return compact labels that expose every sensitivity factor."""
    return [
        (
            f"f {row.false_positive_rate_per_100k:.3g}/100k\n"
            f"sigma {row.reduction_width_id}\n"
            f"corr {row.reduction_error_correlation:.3g}; "
            f"shift {row.reduction_calibration_shift_logit:+.3g}"
        )
        for row in summary.itertuples(index=False)
    ]


def _calibration_axes_vary(summary: pd.DataFrame) -> bool:
    return (
        summary["reduction_error_correlation"].nunique() > 1
        or summary["reduction_calibration_shift_logit"].nunique() > 1
    )


def _only_calibration_axes_vary(summary: pd.DataFrame) -> bool:
    return (
        _calibration_axes_vary(summary)
        and summary["false_positive_rate"].nunique() == 1
        and summary["reduction_width_id"].nunique() == 1
    )


def _calibration_headline_plot(summary: pd.DataFrame):
    """Horizontal forest plot for a coherent-calibration-only sensitivity grid."""
    import matplotlib.pyplot as plt

    height = max(plot_styles.FIGSIZE_XL[1], 0.48 * len(summary) + 1.8)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(plot_styles.FIGSIZE_XL[0], height),
        sharey=True,
        layout="constrained",
    )
    scenario_ids = list(summary["scenario_id"])
    colours = _scenario_colours(scenario_ids)
    y = np.arange(len(summary))
    for idx, (_, row) in enumerate(summary.iterrows()):
        for ax, mean_column, lo_column, hi_column in (
            (axes[0], "true_ds_mean", "true_ds_lo", "true_ds_hi"),
            (axes[1], "recording_s_mean", "recording_s_lo", "recording_s_hi"),
        ):
            mean = float(row[mean_column])
            lo = float(row[lo_column])
            hi = float(row[hi_column])
            colour = colours[str(row["scenario_id"])]
            ax.errorbar(
                [mean],
                [idx],
                xerr=np.array([[mean - lo], [hi - mean]]),
                fmt="*" if bool(row["is_reference"]) else "o",
                markersize=10 if bool(row["is_reference"]) else 6,
                capsize=3,
                color=colour,
                ecolor=colour,
            )
    labels = list(summary["calibration_id"])
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("coherent calibration scenario (star = reference)")
    axes[0].set_xlabel("estimated true DS livebirths")
    axes[0].set_title("True DS total across assumptions")
    axes[1].set_xlabel("certificate recording sensitivity")
    axes[1].set_title("Recording sensitivity across assumptions")
    return fig


def _false_positive_width_headline_plot(summary: pd.DataFrame):
    """Preserve the original false-positive/prior-width line plot."""
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
    for ax, metric in (
        (axes[0], "true_ds_mean"),
        (axes[1], "recording_s_mean"),
    ):
        ax.scatter(
            [reference["false_positive_rate_per_100k"]],
            [reference[metric]],
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


def _categorical_headline_plot(summary: pd.DataFrame):
    """Fallback for grids in which calibration and legacy axes both vary."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=plot_styles.FIGSIZE_XL)
    scenario_ids = list(summary["scenario_id"])
    colours = _scenario_colours(scenario_ids)
    x = np.arange(len(summary))
    for idx, (_, row) in enumerate(summary.iterrows()):
        for ax, mean_column, lo_column, hi_column in (
            (axes[0], "true_ds_mean", "true_ds_lo", "true_ds_hi"),
            (axes[1], "recording_s_mean", "recording_s_lo", "recording_s_hi"),
        ):
            mean = float(row[mean_column])
            lo = float(row[lo_column])
            hi = float(row[hi_column])
            colour = colours[str(row["scenario_id"])]
            ax.errorbar(
                [idx],
                [mean],
                yerr=np.array([[mean - lo], [hi - mean]]),
                fmt="*" if bool(row["is_reference"]) else "o",
                markersize=10 if bool(row["is_reference"]) else 6,
                capsize=3,
                color=colour,
                ecolor=colour,
            )
    axes[0].set_ylabel("estimated true DS livebirths")
    axes[0].set_title("True DS total across assumptions")
    axes[1].set_ylabel("certificate recording sensitivity")
    axes[1].set_title("Recording sensitivity across assumptions")
    labels = _categorical_scenario_labels(summary)
    for ax in axes:
        ax.set_xticks(x, labels, rotation=35, ha="right")
        ax.set_xlabel("categorical assumption scenario (star = reference)")
    fig.tight_layout()
    return fig


def _headline_plot(summary: pd.DataFrame):
    if _only_calibration_axes_vary(summary):
        return _calibration_headline_plot(summary)
    if not _calibration_axes_vary(summary):
        return _false_positive_width_headline_plot(summary)
    return _categorical_headline_plot(summary)


def _calibration_ppc_plot(summary: pd.DataFrame):
    """Compact horizontal PPC plot for coherent-calibration-only grids."""
    import matplotlib.pyplot as plt

    height = max(plot_styles.FIGSIZE_XL[1], 0.48 * len(summary) + 1.8)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(plot_styles.FIGSIZE_XL[0], height),
        sharey=True,
        layout="constrained",
    )
    scenario_ids = list(summary["scenario_id"])
    colours = _scenario_colours(scenario_ids)
    y = np.arange(len(summary))
    for idx, (_, row) in enumerate(summary.iterrows()):
        marker = "*" if bool(row["is_reference"]) else "o"
        size = 100 if bool(row["is_reference"]) else 45
        colour = colours[str(row["scenario_id"])]
        axes[0].scatter(
            [100.0 * float(row["age_year_coverage_fraction"])],
            [idx],
            marker=marker,
            s=size,
            color=colour,
        )
        axes[1].scatter(
            [float(row["age_year_mean_absolute_standardized_residual"])],
            [idx],
            marker=marker,
            s=size,
            color=colour,
        )
    labels = list(summary["calibration_id"])
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("coherent calibration scenario (star = reference)")
    axes[0].set_xlabel("age-year cells covered (%)")
    axes[0].set_title("Native age-year PPC coverage")
    axes[1].set_xlabel("mean absolute standardised residual")
    axes[1].set_title("Native age-year PPC residuals")
    return fig


def _false_positive_width_ppc_plot(summary: pd.DataFrame):
    """Preserve the original false-positive/prior-width PPC line plot."""
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


def _categorical_ppc_plot(summary: pd.DataFrame):
    """Fallback PPC plot when calibration and legacy axes both vary."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=plot_styles.FIGSIZE_XL)
    scenario_ids = list(summary["scenario_id"])
    colours = _scenario_colours(scenario_ids)
    x = np.arange(len(summary))
    for idx, (_, row) in enumerate(summary.iterrows()):
        marker = "*" if bool(row["is_reference"]) else "o"
        size = 100 if bool(row["is_reference"]) else 45
        colour = colours[str(row["scenario_id"])]
        axes[0].scatter(
            [idx],
            [100.0 * float(row["age_year_coverage_fraction"])],
            marker=marker,
            s=size,
            color=colour,
        )
        axes[1].scatter(
            [idx],
            [float(row["age_year_mean_absolute_standardized_residual"])],
            marker=marker,
            s=size,
            color=colour,
        )
    axes[0].set_ylabel("age-year cells covered (%)")
    axes[0].set_title("Native age-year PPC coverage")
    axes[1].set_ylabel("mean absolute standardised residual")
    axes[1].set_title("Native age-year PPC residuals")
    labels = _categorical_scenario_labels(summary)
    for ax in axes:
        ax.set_xticks(x, labels, rotation=35, ha="right")
        ax.set_xlabel("categorical assumption scenario (star = reference)")
    fig.tight_layout()
    return fig


def _ppc_plot(summary: pd.DataFrame):
    if _only_calibration_axes_vary(summary):
        return _calibration_ppc_plot(summary)
    if not _calibration_axes_vary(summary):
        return _false_positive_width_ppc_plot(summary)
    return _categorical_ppc_plot(summary)


def _age_band_scenario_labels(age_band: pd.DataFrame) -> tuple[list[str], str]:
    """Choose compact heatmap labels without hiding any factor that varies."""
    metadata = age_band.drop_duplicates("scenario_id", keep="first")
    if _only_calibration_axes_vary(metadata):
        return list(metadata["calibration_id"]), "coherent calibration scenario"
    if not _calibration_axes_vary(metadata):
        return list(metadata["scenario_id"]), "assumption scenario"
    return _categorical_scenario_labels(metadata), "mixed assumption scenario"


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
    scenario_labels, scenario_axis_label = _age_band_scenario_labels(age_band)
    ax.set_yticks(np.arange(len(scenarios)))
    ax.set_yticklabels(scenario_labels)
    ax.set_xlabel("maternal-age band")
    ax.set_ylabel(scenario_axis_label)
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
        "reduction_error_correlation_definition": (
            "equicorrelation between yearly logit-scale reduction-prior errors; "
            "marginal yearly prior variances are preserved"
        ),
        "reduction_calibration_shift_logit_definition": (
            "fixed logit-scale shift applied to the complete surveillance-derived "
            "reduction trajectory"
        ),
        "standardized_trajectory_error_index_definition": (
            "within each posterior draw, mean across years of rho_logit_year minus "
            "its shifted surveillance-prior location, divided by that year's "
            "marginal prior standard deviation; a derived index, not a separately "
            "estimated calibration parameter"
        ),
        "standardized_trajectory_error_index_dependence_definition": (
            "Pearson correlation across posterior draws with recording_s and, "
            "when saved, recording_s_logit"
        ),
        "model_implied_expected_missed_true_cases_definition": (
            "draw-wise expected true DS livebirths not recorded by the certificate "
            "recording component; uses sum_y true_count_year * "
            "(1 - recording_s_year) when available; the scalar "
            "true_count_total * (1 - recording_s) fallback is valid only for "
            "constant-recording models and is unavailable otherwise"
        ),
        "model_implied_expected_missed_true_cases_interval": (
            "equal-tailed posterior interval at reporting_interval_prob"
        ),
        "materiality_thresholds": {
            "true_ds_mean_absolute_percent_difference": (TOTAL_MEAN_MATERIAL_PERCENT),
            "true_ds_interval_width_percent_increase": (
                TOTAL_INTERVAL_WIDTH_MATERIAL_PERCENT
            ),
            "recording_s_mean_absolute_difference": RECORDING_S_MATERIAL_ABSOLUTE,
            "model_implied_expected_missed_true_cases_mean_absolute_percent_difference": (
                MISSED_TRUE_CASES_MEAN_MATERIAL_PERCENT
            ),
        },
        "aggregate_materiality_drives_joint_corner_decisions": True,
        "decomposition_materiality_is_reported_separately": True,
        "materiality_requires_healthy_reference_and_candidate_fits": True,
        "materiality_proximity_within_two_combined_mcse_evaluated": False,
        "materiality_proximity_note": (
            "saved reporting contracts do not expose compatible Monte Carlo "
            "standard errors for every decision metric"
        ),
        "axes_are_separate_assumptions": True,
        "plot_layout_rule": (
            "horizontal calibration_id forest/point plots when only coherent-"
            "calibration factors vary; original false-positive/prior-width line "
            "plots when calibration factors are invariant; categorical fallback "
            "for mixed grids"
        ),
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
        description="Compare saved core-model assumption-sensitivity fits.",
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
