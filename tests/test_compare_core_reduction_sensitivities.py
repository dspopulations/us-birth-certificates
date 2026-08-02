"""Tests for ``scripts/compare_core_reduction_sensitivities.py``."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
import xarray as xr

matplotlib.use("Agg")

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import compare_core_reduction_sensitivities as sensitivity  # noqa: E402


def _write_run(
    run_dir: Path,
    *,
    false_positive_rate: float,
    observed_sigma: float,
    extrapolated_sigma: float,
    true_total: float,
    draws: int = 100,
    predictive_shift: float = 0.0,
    rhat: float = 1.001,
    divergences: int = 0,
) -> None:
    tables = run_dir / "tables"
    tables.mkdir(parents=True)
    config = {
        "model_id": "DSP004",
        "model_slug": "constant_s_exact_age",
        "family_id": "selection_core_reduction",
        "recording_model": "constant",
        "reduction_model": "year",
        "age_model": "single_year",
        "recorded_definition": "confirmed_or_pending",
        "theta_model": "morris_double_logistic_by_age_code",
        "age_endpoint_convention": {
            "12": "10-12; Morris evaluated at age 12",
            "50": "50+; Morris evaluated at age 50",
        },
        "year_range": [2020, 2021],
        "priors": {
            "theta_lb_age": [0.0007] * 7,
            "theta_lb_age_used": False,
            "reduction_mean": [0.35, 0.40],
            "reduction_logit": [-0.619, -0.405],
            "reduction_sigma": [observed_sigma, extrapolated_sigma],
            "recording_s_logit": 0.0,
            "recording_s_sigma": 1.0,
            "recording_s_year_sigma": 0.35,
            "reduction_age_step_sigma": 0.1,
            "false_positive_rate": false_positive_rate,
            "reduction_source": "fake-reduction.csv",
            "extrapolated_reduction_start": 2021,
        },
    }
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "name": "reporting",
                "draws": draws,
                "tune": draws,
                "chains": 2,
                "target_accept": 0.95,
                "posterior_predictive": True,
                "nuts_sampler": "nutpie",
                "random_seed": draws,
            }
        ),
        encoding="utf-8",
    )

    cells = pd.DataFrame(
        {
            "year_idx": [0, 0, 1, 1],
            "age_idx": [0, 1, 0, 1],
            "maternal_age": [20, 30, 20, 30],
            "N_cell": [250_000, 250_000, 250_000, 250_000],
            "R_cell": [2, 3, 4, 5],
        }
    )
    cells.to_parquet(run_dir / "cells.parquet", index=False)

    recording_s = 0.40 + false_positive_rate * 100.0
    pd.DataFrame(
        [
            {
                "metric": "livebirths",
                "mean": 1_000_000.0,
                "lo": np.nan,
                "hi": np.nan,
            },
            {
                "metric": "recorded_ds",
                "mean": 14.0,
                "lo": np.nan,
                "hi": np.nan,
            },
            {
                "metric": "natural_expected_ds",
                "mean": 70_000.0,
                "lo": np.nan,
                "hi": np.nan,
            },
            {
                "metric": "true_ds_livebirths",
                "mean": true_total,
                "lo": true_total - 1_000.0,
                "hi": true_total + 1_000.0,
            },
            {
                "metric": "aggregate_reduction",
                "mean": 1.0 - true_total / 70_000.0,
                "lo": np.nan,
                "hi": np.nan,
            },
            {
                "metric": "recording_s",
                "mean": recording_s,
                "lo": recording_s - 0.02,
                "hi": recording_s + 0.02,
            },
        ]
    ).to_csv(tables / "core_headlines.csv", index=False)

    accounting_rows = []
    reduction_rows = []
    recording_rows = []
    for year_idx, year in enumerate((2020, 2021)):
        rho_mean = 0.35 + 0.05 * year_idx
        accounting_rows.append(
            {
                "year": year,
                "births": 500_000,
                "recorded_ds": 5 + 4 * year_idx,
                "natural_expected_ds": 35_000.0,
                "true_count_year_mean": true_total / 2.0,
                "true_count_year_lo": true_total / 2.0 - 600.0,
                "true_count_year_hi": true_total / 2.0 + 600.0,
                "recorded_count_year_mu_mean": 5.0 + 4.0 * year_idx,
                "recorded_count_year_mu_lo": 4.0 + 4.0 * year_idx,
                "recorded_count_year_mu_hi": 6.0 + 4.0 * year_idx,
                "interval_prob": 0.89,
            }
        )
        sigma = observed_sigma if year == 2020 else extrapolated_sigma
        reduction_rows.append(
            {
                "year": year,
                "rho_prior_mean": rho_mean,
                "rho_prior_lo": rho_mean - sigma / 10.0,
                "rho_prior_hi": rho_mean + sigma / 10.0,
                "rho_prior_sigma_logit": sigma,
                "rho_year_mean": rho_mean,
                "rho_year_lo": rho_mean - 0.03,
                "rho_year_hi": rho_mean + 0.03,
                "eta_year_mean": 1.0 - rho_mean,
                "extrapolated": year == 2021,
            }
        )
        recording_rows.append(
            {
                "year": year,
                "posterior_mean": recording_s,
                "posterior_lo": recording_s - 0.02,
                "posterior_hi": recording_s + 0.02,
                "interval_prob": 0.89,
            }
        )
    pd.DataFrame(accounting_rows).to_csv(
        tables / "core_accounting_by_year.csv", index=False
    )
    pd.DataFrame(reduction_rows).to_csv(
        tables / "core_reduction_prior_posterior.csv", index=False
    )
    pd.DataFrame(recording_rows).to_csv(
        tables / "core_recording_s_by_year.csv", index=False
    )

    age_year_rows = []
    observed_values = [2, 3, 4, 5]
    for row, observed in zip(
        cells.itertuples(index=False), observed_values, strict=True
    ):
        predicted = observed + predictive_shift
        age_year_rows.append(
            {
                "year_idx": row.year_idx,
                "year": 2020 + row.year_idx,
                "age_idx": row.age_idx,
                "age": str(row.maternal_age),
                "births": row.N_cell,
                "observed": observed,
                "predicted_mean": predicted,
                "predicted_lo": max(0.0, predicted - 2.0),
                "predicted_hi": predicted + 2.0,
                "posterior_predictive_sd": 1.0,
                "residual_observed_minus_predicted": observed - predicted,
                "standardized_residual": observed - predicted,
                "relative_residual": (observed - predicted) / predicted,
                "interval_prob": 0.89,
                "observed_in_interval": abs(predictive_shift) <= 2.0,
            }
        )
    pd.DataFrame(age_year_rows).to_csv(tables / "core_ppc_by_age_year.csv", index=False)
    band_rows = []
    for idx, (label, observed) in enumerate((("20-24", 6), ("30-34", 8))):
        predicted = observed + 2.0 * predictive_shift
        band_rows.append(
            {
                "age_band_idx": idx,
                "label": label,
                "observed": observed,
                "predicted_mean": predicted,
                "predicted_lo": max(0.0, predicted - 3.0),
                "predicted_hi": predicted + 3.0,
                "interval_prob": 0.89,
                "observed_in_interval": abs(predictive_shift) <= 1.5,
            }
        )
    pd.DataFrame(band_rows).to_csv(tables / "core_ppc_by_age_band.csv", index=False)

    pd.DataFrame(
        {
            "mean": [0.4, 0.35],
            "ess_bulk": [800.0, 900.0],
            "ess_tail": [700.0, 750.0],
            "r_hat": [rhat, rhat],
        },
        index=["recording_s", "rho_year[0]"],
    ).to_csv(run_dir / "summary.csv")
    diverging = np.zeros((2, draws), dtype=bool)
    diverging.flat[:divergences] = True
    sample_stats = xr.Dataset(
        {"diverging": (("chain", "draw"), diverging)},
        coords={"chain": [0, 1], "draw": np.arange(draws)},
    )
    xr.DataTree.from_dict({"/sample_stats": sample_stats}).to_netcdf(
        run_dir / "idata.nc"
    )


def _update_config(run_dir: Path, update) -> None:
    path = run_dir / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    update(config)
    path.write_text(json.dumps(config), encoding="utf-8")


def test_comparison_writes_incomplete_grid_outputs(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    f_zero = tmp_path / "f_zero"
    wide = tmp_path / "wide"
    _write_run(
        reference,
        false_positive_rate=7.8e-5,
        observed_sigma=0.20,
        extrapolated_sigma=0.45,
        true_total=44_000.0,
        draws=120,
    )
    _write_run(
        f_zero,
        false_positive_rate=0.0,
        observed_sigma=0.20,
        extrapolated_sigma=0.45,
        true_total=40_000.0,
        draws=80,
        predictive_shift=1.0,
    )
    _write_run(
        wide,
        false_positive_rate=7.8e-5,
        observed_sigma=0.40,
        extrapolated_sigma=0.90,
        true_total=45_000.0,
        draws=200,
        predictive_shift=0.5,
    )

    paths = sensitivity.compare_core_reduction_sensitivities(
        reference,
        [f_zero, wide],
        tmp_path / "comparison",
    )

    for key in (
        "scenarios",
        "summary",
        "by_year",
        "age_band",
        "age_year",
        "contrasts",
        "envelope",
        "config",
        "headlines_plot_png",
        "headlines_plot_svg",
        "ppc_plot_png",
        "ppc_plot_svg",
        "age_band_residuals_plot_png",
        "age_band_residuals_plot_svg",
    ):
        assert paths[key].is_file()

    scenarios = pd.read_csv(paths["scenarios"])
    assert list(scenarios["draws"]) == [120, 80, 200]
    assert scenarios["fit_healthy"].all()
    assert (scenarios["divergences"] == 0).all()

    summary = pd.read_csv(paths["summary"])
    reference_row = summary.loc[summary["is_reference"]].iloc[0]
    assert reference_row["true_ds_mean_difference_from_reference"] == pytest.approx(0.0)
    assert reference_row["expected_false_positive_flags_mean"] == pytest.approx(
        7.8e-5 * (1_000_000.0 - 44_000.0)
    )
    zero_row = summary.loc[summary["false_positive_rate"] == 0.0].iloc[0]
    assert zero_row["expected_false_positive_flags_mean"] == 0.0

    assert len(pd.read_csv(paths["by_year"])) == 6
    assert len(pd.read_csv(paths["age_band"])) == 6
    assert len(pd.read_csv(paths["age_year"])) == 12
    contrasts = pd.read_csv(paths["contrasts"])
    assert set(contrasts["varied_factor"]) == {
        "false_positive_rate",
        "reduction_prior_width",
    }
    envelope = pd.read_csv(paths["envelope"])
    total_envelope = envelope.query("metric == 'true_ds_livebirths'").iloc[0]
    assert total_envelope["minimum_mean"] == 40_000.0
    assert total_envelope["maximum_mean"] == 45_000.0
    assert "not a posterior" in total_envelope["envelope_definition"]

    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    assert config["factorial_grid_complete"] is False
    assert config["incomplete_factorial_grid_allowed"] is True
    assert len(config["missing_factor_combinations"]) == 1
    assert config["sampling_budgets_may_differ"] is True
    assert config["raw_loo_or_waic_compared"] is False


def test_comparison_records_unhealthy_fit_without_rejecting_it(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    unhealthy = tmp_path / "unhealthy"
    _write_run(
        reference,
        false_positive_rate=7.8e-5,
        observed_sigma=0.20,
        extrapolated_sigma=0.45,
        true_total=44_000.0,
    )
    _write_run(
        unhealthy,
        false_positive_rate=0.0,
        observed_sigma=0.20,
        extrapolated_sigma=0.45,
        true_total=40_000.0,
        rhat=1.02,
        divergences=2,
    )

    paths = sensitivity.compare_core_reduction_sensitivities(
        reference,
        [unhealthy],
        tmp_path / "comparison",
    )

    scenarios = pd.read_csv(paths["scenarios"])
    row = scenarios.loc[~scenarios["is_reference"]].iloc[0]
    assert row["divergences"] == 2
    assert not bool(row["convergence_ok"])
    assert not bool(row["fit_healthy"])
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    assert config["all_fits_healthy"] is False


def test_comparison_rejects_changed_invariant_prior(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    changed = tmp_path / "changed"
    _write_run(
        reference,
        false_positive_rate=7.8e-5,
        observed_sigma=0.20,
        extrapolated_sigma=0.45,
        true_total=44_000.0,
    )
    _write_run(
        changed,
        false_positive_rate=0.0,
        observed_sigma=0.20,
        extrapolated_sigma=0.45,
        true_total=40_000.0,
    )
    _update_config(
        changed,
        lambda config: config["priors"].update({"recording_s_sigma": 2.0}),
    )

    with pytest.raises(ValueError, match="changes a prior other than"):
        sensitivity.compare_core_reduction_sensitivities(
            reference,
            [changed],
            tmp_path / "comparison",
        )


def test_comparison_rejects_changed_cells(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    changed = tmp_path / "changed"
    _write_run(
        reference,
        false_positive_rate=7.8e-5,
        observed_sigma=0.20,
        extrapolated_sigma=0.45,
        true_total=44_000.0,
    )
    _write_run(
        changed,
        false_positive_rate=0.0,
        observed_sigma=0.20,
        extrapolated_sigma=0.45,
        true_total=40_000.0,
    )
    cells = pd.read_parquet(changed / "cells.parquet")
    cells.loc[0, "N_cell"] += 1
    cells.to_parquet(changed / "cells.parquet", index=False)

    with pytest.raises(ValueError, match="same model cells"):
        sensitivity.compare_core_reduction_sensitivities(
            reference,
            [changed],
            tmp_path / "comparison",
        )


def test_comparison_rejects_duplicate_factor_combination(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    duplicate = tmp_path / "duplicate"
    _write_run(
        reference,
        false_positive_rate=7.8e-5,
        observed_sigma=0.20,
        extrapolated_sigma=0.45,
        true_total=44_000.0,
    )
    _write_run(
        duplicate,
        false_positive_rate=7.8e-5,
        observed_sigma=0.20,
        extrapolated_sigma=0.45,
        true_total=44_100.0,
    )

    with pytest.raises(ValueError, match="factor combination"):
        sensitivity.compare_core_reduction_sensitivities(
            reference,
            [duplicate],
            tmp_path / "comparison",
        )
