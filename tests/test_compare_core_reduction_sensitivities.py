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
import matplotlib.pyplot as plt  # noqa: E402

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
    reduction_error_correlation: float | None = None,
    reduction_calibration_shift_logit: float | None = None,
    recording_s_mean: float | None = None,
    true_interval_width: float = 2_000.0,
    include_posterior: bool = False,
    include_year_specific_posterior: bool = True,
    recording_model: str = "constant",
    posterior_recording_s_slope: float = 0.02,
) -> None:
    tables = run_dir / "tables"
    tables.mkdir(parents=True)
    config = {
        "model_id": "DSP004",
        "model_slug": "constant_s_exact_age",
        "family_id": "selection_core_reduction",
        "recording_model": recording_model,
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
    if reduction_error_correlation is not None:
        config["priors"]["reduction_error_correlation"] = reduction_error_correlation
    if reduction_calibration_shift_logit is not None:
        config["priors"]["reduction_calibration_shift_logit"] = (
            reduction_calibration_shift_logit
        )
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

    recording_s = (
        recording_s_mean
        if recording_s_mean is not None
        else 0.40 + false_positive_rate * 100.0
    )
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
                "lo": true_total - true_interval_width / 2.0,
                "hi": true_total + true_interval_width / 2.0,
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
    groups = {"/sample_stats": sample_stats}
    if include_posterior:
        index = np.linspace(-1.0, 1.0, 2 * draws).reshape(2, draws)
        recording_draws = recording_s + posterior_recording_s_slope * index
        shift = reduction_calibration_shift_logit or 0.0
        location = np.asarray(config["priors"]["reduction_logit"]) + shift
        scale = np.asarray(config["priors"]["reduction_sigma"])
        rho_logit = location[None, None, :] + index[:, :, None] * scale
        true_total_draws = true_total + 500.0 * index
        posterior_variables = {
            "rho_logit_year": (("chain", "draw", "year"), rho_logit),
            "recording_s": (("chain", "draw"), recording_draws),
            "recording_s_logit": (
                ("chain", "draw"),
                np.log(recording_draws / (1.0 - recording_draws)),
            ),
            "true_count_total": (("chain", "draw"), true_total_draws),
        }
        if include_year_specific_posterior:
            posterior_variables.update(
                {
                    "true_count_year": (
                        ("chain", "draw", "year"),
                        np.repeat(
                            (true_total_draws / 2.0)[:, :, None],
                            2,
                            axis=2,
                        ),
                    ),
                    "recording_s_year": (
                        ("chain", "draw", "year"),
                        np.repeat(recording_draws[:, :, None], 2, axis=2),
                    ),
                }
            )
        groups["/posterior"] = xr.Dataset(
            posterior_variables,
            coords={
                "chain": [0, 1],
                "draw": np.arange(draws),
                "year": [0, 1],
            },
        )
    xr.DataTree.from_dict(groups).to_netcdf(run_dir / "idata.nc")


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
    assert (scenarios["reduction_error_correlation"] == 0.0).all()
    assert (scenarios["reduction_calibration_shift_logit"] == 0.0).all()

    summary = pd.read_csv(paths["summary"])
    reference_row = summary.loc[summary["is_reference"]].iloc[0]
    assert reference_row["true_ds_mean_difference_from_reference"] == pytest.approx(0.0)
    assert reference_row["expected_false_positive_flags_mean"] == pytest.approx(
        7.8e-5 * (1_000_000.0 - 44_000.0)
    )
    zero_row = summary.loc[summary["false_positive_rate"] == 0.0].iloc[0]
    assert zero_row["expected_false_positive_flags_mean"] == 0.0
    legacy_figure = sensitivity._headline_plot(summary)
    assert "false-positive probability" in legacy_figure.axes[0].get_xlabel()
    plt.close(legacy_figure)

    assert len(pd.read_csv(paths["by_year"])) == 6
    age_band = pd.read_csv(paths["age_band"])
    assert len(age_band) == 6
    assert len(pd.read_csv(paths["age_year"])) == 12
    legacy_heatmap = sensitivity._age_band_residual_plot(age_band)
    assert [
        label.get_text() for label in legacy_heatmap.axes[0].get_yticklabels()
    ] == list(age_band["scenario_id"].drop_duplicates())
    assert legacy_heatmap.axes[0].get_ylabel() == "assumption scenario"
    plt.close(legacy_heatmap)
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
    assert config["reduction_error_correlation_levels"] == [0.0]
    assert config["reduction_calibration_shift_logit_levels"] == [0.0]
    assert config["sampling_budgets_may_differ"] is True
    assert config["raw_loo_or_waic_compared"] is False


def test_comparison_reports_coherent_calibration_and_posterior_diagnostics(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    correlated = tmp_path / "correlated"
    shifted = tmp_path / "shifted"
    _write_run(
        reference,
        false_positive_rate=0.0,
        observed_sigma=0.20,
        extrapolated_sigma=0.45,
        true_total=44_000.0,
        recording_s_mean=0.40,
        include_posterior=True,
    )
    _write_run(
        correlated,
        false_positive_rate=0.0,
        observed_sigma=0.20,
        extrapolated_sigma=0.45,
        true_total=47_000.0,
        recording_s_mean=0.46,
        true_interval_width=2_600.0,
        reduction_error_correlation=0.5,
        reduction_calibration_shift_logit=0.0,
        include_posterior=True,
    )
    _write_run(
        shifted,
        false_positive_rate=0.0,
        observed_sigma=0.20,
        extrapolated_sigma=0.45,
        true_total=50_000.0,
        recording_s_mean=0.40,
        reduction_error_correlation=0.0,
        reduction_calibration_shift_logit=0.2,
        include_posterior=True,
    )

    paths = sensitivity.compare_core_reduction_sensitivities(
        reference,
        [correlated, shifted],
        tmp_path / "comparison",
    )

    scenarios = pd.read_csv(paths["scenarios"])
    assert scenarios["standardized_trajectory_error_index_available"].all()
    assert (
        scenarios["standardized_trajectory_error_index_recording_s_correlation"] > 0.999
    ).all()
    assert (
        scenarios["standardized_trajectory_error_index_recording_s_logit_correlation"]
        > 0.999
    ).all()
    assert scenarios["model_implied_expected_missed_true_cases_available"].all()
    assert set(scenarios["model_implied_expected_missed_true_cases_method"]) == {
        "sum_y true_count_year * (1 - recording_s_year)"
    }

    summary = pd.read_csv(paths["summary"])
    correlated_row = summary.loc[summary["reduction_error_correlation"] == 0.5].iloc[0]
    assert correlated_row["true_ds_mean_material_change_from_reference"]
    assert correlated_row["true_ds_interval_width_material_change_from_reference"]
    assert correlated_row["recording_s_mean_material_change_from_reference"]
    assert correlated_row["aggregate_material_change_from_reference"]
    assert correlated_row["decomposition_material_change_from_reference"]
    shifted_row = summary.loc[summary["reduction_calibration_shift_logit"] == 0.2].iloc[
        0
    ]
    assert shifted_row[
        "model_implied_expected_missed_true_cases_mean_material_change_from_reference"
    ]
    assert (
        shifted_row["model_implied_expected_missed_true_cases_hi"]
        > shifted_row["model_implied_expected_missed_true_cases_lo"]
    )
    calibration_figure = sensitivity._headline_plot(summary)
    assert [
        label.get_text() for label in calibration_figure.axes[0].get_yticklabels()
    ] == list(summary["calibration_id"])
    assert all(
        "f " not in label.get_text()
        for label in calibration_figure.axes[0].get_yticklabels()
    )
    plt.close(calibration_figure)
    age_band = pd.read_csv(paths["age_band"])
    calibration_heatmap = sensitivity._age_band_residual_plot(age_band)
    assert [
        label.get_text() for label in calibration_heatmap.axes[0].get_yticklabels()
    ] == list(age_band.drop_duplicates("scenario_id")["calibration_id"])
    assert calibration_heatmap.axes[0].get_ylabel() == ("coherent calibration scenario")
    plt.close(calibration_heatmap)

    mixed = age_band.copy()
    shifted_scenario = mixed["scenario_id"] == mixed["scenario_id"].iloc[-1]
    mixed.loc[shifted_scenario, "false_positive_rate"] = 1e-4
    mixed.loc[shifted_scenario, "false_positive_rate_per_100k"] = 10.0
    mixed_labels, mixed_axis_label = sensitivity._age_band_scenario_labels(mixed)
    assert mixed_axis_label == "mixed assumption scenario"
    assert all("sigma " in label and "corr " in label for label in mixed_labels)

    contrasts = pd.read_csv(paths["contrasts"])
    assert set(contrasts["varied_factor"]) == {
        "reduction_error_correlation",
        "reduction_calibration_shift_logit",
    }
    assert len(contrasts) == 2 * len(sensitivity.CONTRAST_METRICS)
    material_total = contrasts.loc[
        (contrasts["varied_factor"] == "reduction_error_correlation")
        & (contrasts["metric"] == "true_ds_mean")
    ].iloc[0]
    assert material_total["materiality_evaluated"]
    assert material_total["material_change"]

    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    assert config["factorial_grid_complete"] is False
    assert len(config["missing_factor_combinations"]) == 1
    assert "horizontal calibration_id" in config["plot_layout_rule"]
    assert (
        config["materiality_thresholds"][
            "model_implied_expected_missed_true_cases_mean_absolute_percent_difference"
        ]
        == 10.0
    )
    assert config["materiality_proximity_within_two_combined_mcse_evaluated"] is False


def test_posterior_missed_true_cases_falls_back_to_constant_s_formula(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        false_positive_rate=0.0,
        observed_sigma=0.20,
        extrapolated_sigma=0.45,
        true_total=44_000.0,
        recording_s_mean=0.40,
        include_posterior=True,
        include_year_specific_posterior=False,
    )

    run = sensitivity.load_sensitivity_run(run_dir)

    assert run.model_implied_expected_missed_true_cases_method == (
        "true_count_total * (1 - recording_s)"
    )
    assert run.model_implied_expected_missed_true_cases_mean == pytest.approx(
        26_396.633,
        rel=1e-4,
    )
    assert run.model_implied_expected_missed_true_cases_hi > (
        run.model_implied_expected_missed_true_cases_lo
    )


def test_posterior_missed_true_cases_does_not_use_scalar_fallback_for_year_s(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        false_positive_rate=0.0,
        observed_sigma=0.20,
        extrapolated_sigma=0.45,
        true_total=44_000.0,
        include_posterior=True,
        include_year_specific_posterior=False,
        recording_model="year",
    )

    run = sensitivity.load_sensitivity_run(run_dir, is_reference=True)
    scenarios = sensitivity.scenario_table([run])
    summary = sensitivity.sensitivity_summary_table([run])

    assert run.model_implied_expected_missed_true_cases_mean is None
    assert run.model_implied_expected_missed_true_cases_method is None
    assert "requires posterior true_count_year" in (
        run.model_implied_expected_missed_true_cases_unavailable_reason
    )
    assert not scenarios["model_implied_expected_missed_true_cases_available"].iloc[0]
    assert not summary[
        "model_implied_expected_missed_true_cases_mean_materiality_evaluated_from_reference"
    ].iloc[0]
    assert pd.isna(
        summary[
            "model_implied_expected_missed_true_cases_mean_material_change_from_reference"
        ].iloc[0]
    )
    assert (
        summary[
            "model_implied_expected_missed_true_cases_mean_materiality_non_evaluation_reason"
        ].iloc[0]
        == "reference model-implied expected missed true cases is unavailable"
    )
    assert not summary["decomposition_materiality_evaluated_from_reference"].iloc[0]
    assert pd.isna(summary["decomposition_material_change_from_reference"].iloc[0])
    assert (
        "reference model-implied expected missed true cases is unavailable"
        in (summary["decomposition_materiality_non_evaluation_reason"].iloc[0])
    )


def test_nonfinite_posterior_correlations_are_not_marked_available(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        false_positive_rate=0.0,
        observed_sigma=0.20,
        extrapolated_sigma=0.45,
        true_total=44_000.0,
        include_posterior=True,
        posterior_recording_s_slope=0.0,
    )

    run = sensitivity.load_sensitivity_run(run_dir, is_reference=True)
    scenarios = sensitivity.scenario_table([run])
    summary = sensitivity.sensitivity_summary_table([run])

    assert np.isnan(run.standardized_trajectory_error_index_recording_s_correlation)
    assert np.isnan(
        run.standardized_trajectory_error_index_recording_s_logit_correlation
    )
    assert not scenarios["standardized_trajectory_error_index_available"].iloc[0]
    assert not scenarios[
        "standardized_trajectory_error_index_recording_s_correlation_available"
    ].iloc[0]
    assert not summary["standardized_trajectory_error_index_available"].iloc[0]
    assert summary["model_implied_expected_missed_true_cases_available"].iloc[0]


def test_reference_unavailable_missed_cases_propagate_through_decomposition(
    tmp_path: Path,
) -> None:
    reference_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"
    _write_run(
        reference_dir,
        false_positive_rate=0.0,
        observed_sigma=0.20,
        extrapolated_sigma=0.45,
        true_total=44_000.0,
        recording_s_mean=0.40,
        include_posterior=True,
        include_year_specific_posterior=False,
        recording_model="year",
    )
    _write_run(
        candidate_dir,
        false_positive_rate=1e-5,
        observed_sigma=0.20,
        extrapolated_sigma=0.45,
        true_total=44_000.0,
        recording_s_mean=0.50,
        include_posterior=True,
        include_year_specific_posterior=True,
        recording_model="year",
    )
    runs = [
        sensitivity.load_sensitivity_run(reference_dir, is_reference=True),
        sensitivity.load_sensitivity_run(candidate_dir),
    ]

    summary = sensitivity.sensitivity_summary_table(runs)
    reference = summary.loc[summary["is_reference"]].iloc[0]
    candidate = summary.loc[~summary["is_reference"]].iloc[0]

    assert pd.isna(reference["decomposition_material_change_from_reference"])
    assert not reference["decomposition_materiality_evaluated_from_reference"]
    assert not candidate[
        "model_implied_expected_missed_true_cases_mean_materiality_evaluated_from_reference"
    ]
    assert pd.isna(
        candidate[
            "model_implied_expected_missed_true_cases_mean_material_change_from_reference"
        ]
    )
    assert (
        candidate[
            "model_implied_expected_missed_true_cases_mean_materiality_non_evaluation_reason"
        ]
        == "reference model-implied expected missed true cases is unavailable"
    )
    assert candidate["recording_s_mean_material_change_from_reference"]
    assert candidate["decomposition_material_change_from_reference"]
    assert candidate["decomposition_materiality_evaluated_from_reference"]
    assert pd.isna(candidate["decomposition_materiality_non_evaluation_reason"])


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
    assert not bool(row["materiality_evaluated_against_reference"])
    assert row["materiality_non_evaluation_reason"] == "candidate fit is unhealthy"
    summary = pd.read_csv(paths["summary"])
    summary_row = summary.loc[~summary["is_reference"]].iloc[0]
    assert not bool(summary_row["materiality_evaluated_against_reference"])
    assert not bool(summary_row["aggregate_material_change_from_reference"])
    assert summary_row["materiality_non_evaluation_reason"] == (
        "candidate fit is unhealthy"
    )
    contrasts = pd.read_csv(paths["contrasts"])
    total_contrast = contrasts.loc[contrasts["metric"] == "true_ds_mean"].iloc[0]
    assert not bool(total_contrast["both_fits_healthy"])
    assert not bool(total_contrast["materiality_evaluated"])
    assert not bool(total_contrast["material_change"])
    assert total_contrast["materiality_non_evaluation_reason"] == (
        "candidate fit is unhealthy"
    )
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    assert config["all_fits_healthy"] is False


def test_materiality_is_not_evaluated_when_reference_fit_is_unhealthy(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_run(
        reference,
        false_positive_rate=7.8e-5,
        observed_sigma=0.20,
        extrapolated_sigma=0.45,
        true_total=44_000.0,
        rhat=1.02,
    )
    _write_run(
        candidate,
        false_positive_rate=0.0,
        observed_sigma=0.20,
        extrapolated_sigma=0.45,
        true_total=40_000.0,
    )

    paths = sensitivity.compare_core_reduction_sensitivities(
        reference,
        [candidate],
        tmp_path / "comparison",
    )

    scenarios = pd.read_csv(paths["scenarios"])
    assert not scenarios["reference_fit_healthy"].any()
    assert not scenarios["materiality_evaluated_against_reference"].any()
    candidate_row = scenarios.loc[~scenarios["is_reference"]].iloc[0]
    assert candidate_row["materiality_non_evaluation_reason"] == (
        "reference fit is unhealthy"
    )
    summary = pd.read_csv(paths["summary"])
    assert not summary["aggregate_material_change_from_reference"].any()
    contrasts = pd.read_csv(paths["contrasts"])
    total_contrast = contrasts.loc[contrasts["metric"] == "true_ds_mean"].iloc[0]
    assert not bool(total_contrast["materiality_evaluated"])
    assert total_contrast["materiality_non_evaluation_reason"] == (
        "reference fit is unhealthy"
    )


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
