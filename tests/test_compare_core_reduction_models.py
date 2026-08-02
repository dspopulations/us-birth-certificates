"""Tests for ``scripts/compare_core_reduction_models.py``."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dspopulations_us_birth_certificates.chance import get_ds_lb_nt_probability_array

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import compare_core_reduction_models  # noqa: E402


def _write_fake_core_run(
    run_dir: Path,
    *,
    model_id: str,
    true_total: float,
    recording_s: float,
    age_reduction: bool = False,
) -> None:
    tables = run_dir / "tables"
    tables.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps({"model_id": model_id}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "metric": "true_ds_livebirths",
                "mean": true_total,
                "lo": true_total - 10,
                "hi": true_total + 10,
                "notes": "fake total",
            },
            {
                "metric": "aggregate_reduction",
                "mean": 0.4,
                "lo": float("nan"),
                "hi": float("nan"),
                "notes": "fake reduction",
            },
            {
                "metric": "recording_s",
                "mean": recording_s,
                "lo": recording_s - 0.01,
                "hi": recording_s + 0.01,
                "notes": "fake recording",
            },
        ]
    ).to_csv(tables / "core_headlines.csv", index=False)
    pd.DataFrame(
        [
            {
                "year": 2020,
                "posterior_mean": recording_s,
                "posterior_lo": recording_s - 0.01,
                "posterior_hi": recording_s + 0.01,
            },
            {
                "year": 2021,
                "posterior_mean": recording_s + 0.02,
                "posterior_lo": recording_s + 0.01,
                "posterior_hi": recording_s + 0.03,
            },
        ]
    ).to_csv(tables / "core_recording_s_by_year.csv", index=False)
    pd.DataFrame(
        [
            {
                "year": 2020,
                "rho_year_mean": 0.35,
                "rho_year_lo": 0.30,
                "rho_year_hi": 0.40,
            },
            {
                "year": 2021,
                "rho_year_mean": 0.40,
                "rho_year_lo": 0.35,
                "rho_year_hi": 0.45,
            },
        ]
    ).to_csv(tables / "core_reduction_prior_posterior.csv", index=False)
    if age_reduction:
        rows = []
        for year, marginal in ((2020, 0.35), (2021, 0.40)):
            for age_idx, (age, shift) in enumerate((("<20", -0.10), ("45+", 0.10))):
                mean = marginal + shift
                rows.append(
                    {
                        "year": year,
                        "age_idx": age_idx,
                        "age": age,
                        "natural_expected_ds": 50.0,
                        "natural_ds_weight_share": 0.5,
                        "rho_year_age_mean": mean,
                        "rho_year_age_lo": mean - 0.04,
                        "rho_year_age_hi": mean + 0.04,
                        "rho_year_marginal_mean": marginal,
                        "rho_year_marginal_lo": marginal - 0.03,
                        "rho_year_marginal_hi": marginal + 0.03,
                        "rho_year_marginal_max_abs_draw_difference": 1e-12,
                    }
                )
        pd.DataFrame(rows).to_csv(
            tables / "core_reduction_by_age_year.csv",
            index=False,
        )


def _exact_cells() -> pd.DataFrame:
    ages = [18, 22, 32, 50]
    labels = ["18", "22", "32", "50+"]
    rows = []
    for year_idx in range(2):
        for age_idx, (age, label) in enumerate(zip(ages, labels, strict=True)):
            rows.append(
                {
                    "year_idx": year_idx,
                    "age_idx": age_idx,
                    "maternal_age": age,
                    "maternal_age_label": label,
                    "N_cell": 10_000 - 500 * age_idx,
                    "R_cell": 2 + year_idx + age_idx,
                }
            )
    return pd.DataFrame(rows)


def _write_common_grid_run(
    run_dir: Path,
    *,
    model_id: str,
    age_model: str,
    recording_model: str,
    age_reduction: bool = False,
) -> None:
    _write_fake_core_run(
        run_dir,
        model_id=model_id,
        true_total=100.0,
        recording_s=0.4,
        age_reduction=age_reduction,
    )
    exact = _exact_cells()
    theta_band = np.array([0.0007, 0.0008, 0.0010, 0.0015, 0.0047, 0.015, 0.03])
    config = {
        "model_id": model_id,
        "age_model": age_model,
        "recording_model": recording_model,
        "reduction_model": "year_age" if age_reduction else "year",
        "recorded_definition": "confirmed_or_pending",
        "year_range": [2020, 2021],
        "priors": {
            "theta_lb_age": theta_band.tolist(),
            "false_positive_rate": 0.0,
        },
    }
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if age_model == "single_year":
        cells = exact
    else:
        band_idx = np.select(
            [
                exact["maternal_age"] < 20,
                exact["maternal_age"] < 25,
                exact["maternal_age"] < 30,
                exact["maternal_age"] < 35,
                exact["maternal_age"] < 40,
                exact["maternal_age"] < 45,
            ],
            [0, 1, 2, 3, 4, 5],
            default=6,
        )
        cells = (
            exact.assign(age_idx=band_idx)
            .groupby(["year_idx", "age_idx"], as_index=False)[["N_cell", "R_cell"]]
            .sum()
        )
    cells.to_parquet(run_dir / "cells.parquet", index=False)

    draw = np.arange(12)
    rho = np.stack(
        [0.30 + 0.002 * draw, 0.38 + 0.002 * draw],
        axis=1,
    )[None, :, :]
    recording = np.stack(
        [0.40 + 0.001 * draw, 0.42 + 0.001 * draw],
        axis=1,
    )[None, :, :]
    posterior_variables: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
        "rho_year": (("chain", "draw", "year"), rho),
        "recording_s_year": (("chain", "draw", "year"), recording),
    }
    if age_reduction:
        offset = np.array([-0.08, -0.02, 0.03, 0.10])
        posterior_variables["rho_year_age"] = (
            ("chain", "draw", "year", "age"),
            np.clip(rho[..., None] + offset, 0.01, 0.99),
        )
    posterior = xr.Dataset(
        posterior_variables,
        coords={
            "chain": [0],
            "draw": draw,
            "year": [0, 1],
            **({"age": [18, 22, 32, 50]} if age_reduction else {}),
        },
    )
    if age_model == "single_year":
        theta = np.asarray(get_ds_lb_nt_probability_array(np.array([18, 22, 32, 50])))
        constant_data = xr.Dataset(
            {"theta_lb_age": (("age",), theta)},
            coords={"age": [18, 22, 32, 50]},
        )
    else:
        constant_data = xr.Dataset(
            {"theta_lb_age": (("age",), theta_band)},
            coords={"age": np.arange(7)},
        )
    xr.DataTree.from_dict(
        {
            "/posterior": posterior,
            "/constant_data": constant_data,
        }
    ).to_netcdf(run_dir / "idata.nc")


def test_compare_core_model_outputs_writes_tables_and_plot(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    extension = tmp_path / "extension"
    output = tmp_path / "comparison"
    _write_fake_core_run(
        baseline,
        model_id="DSP001",
        true_total=100.0,
        recording_s=0.40,
    )
    _write_fake_core_run(
        extension,
        model_id="DSP002",
        true_total=105.0,
        recording_s=0.42,
    )

    paths = compare_core_reduction_models.compare_core_model_outputs(
        baseline,
        extension,
        output,
    )

    assert paths["headline"].is_file()
    assert paths["recording"].is_file()
    assert paths["recording_plot_png"].is_file()
    assert paths["recording_plot_svg"].is_file()
    headlines = pd.read_csv(paths["headline"])
    recording = pd.read_csv(paths["recording"])
    assert list(headlines["metric"]) == [
        "true_ds_livebirths",
        "aggregate_reduction",
        "recording_s",
    ]
    assert headlines.loc[0, "extension_minus_baseline_mean"] == 5.0
    assert list(recording["year"]) == [2020, 2021]
    assert recording.loc[0, "extension_minus_baseline_mean"] == pytest.approx(0.02)
    assert "age_reduction" not in paths


def test_compare_core_model_outputs_adds_age_reduction_when_available(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    extension = tmp_path / "extension"
    output = tmp_path / "comparison"
    _write_fake_core_run(
        baseline,
        model_id="DSP001",
        true_total=100.0,
        recording_s=0.40,
    )
    _write_fake_core_run(
        extension,
        model_id="DSP003",
        true_total=102.0,
        recording_s=0.41,
        age_reduction=True,
    )

    paths = compare_core_reduction_models.compare_core_model_outputs(
        baseline,
        extension,
        output,
    )

    assert paths["age_reduction"].is_file()
    assert paths["age_reduction_plot_png"].is_file()
    assert paths["age_reduction_plot_svg"].is_file()
    comparison = pd.read_csv(paths["age_reduction"])
    assert len(comparison) == 4
    assert set(comparison["baseline_quantity"]) == {"common reduction across age cells"}
    younger_2020 = comparison.query("year == 2020 and age_idx == 0").iloc[0]
    older_2020 = comparison.query("year == 2020 and age_idx == 1").iloc[0]
    assert younger_2020["baseline_mean"] == pytest.approx(0.35)
    assert younger_2020["extension_minus_baseline_mean"] == pytest.approx(-0.10)
    assert older_2020["extension_minus_baseline_mean"] == pytest.approx(0.10)
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    assert config["age_reduction_comparison"] is True


@pytest.mark.parametrize(
    ("baseline_id", "extension_id", "recording_model"),
    [
        ("DSP001", "DSP004", "constant"),
        ("DSP002", "DSP005", "year"),
    ],
)
def test_compare_exact_age_ablation_on_reconstructed_common_grid(
    tmp_path: Path,
    baseline_id: str,
    extension_id: str,
    recording_model: str,
) -> None:
    baseline = tmp_path / "baseline"
    extension = tmp_path / "extension"
    output = tmp_path / "comparison"
    _write_common_grid_run(
        baseline,
        model_id=baseline_id,
        age_model="band",
        recording_model=recording_model,
    )
    _write_common_grid_run(
        extension,
        model_id=extension_id,
        age_model="single_year",
        recording_model=recording_model,
    )

    paths = compare_core_reduction_models.compare_core_model_outputs(
        baseline,
        extension,
        output,
    )

    for key in (
        "common_grid_ppc",
        "common_grid_ppc_summary",
        "common_grid_ppc_plot_png",
        "common_grid_ppc_plot_svg",
        "common_age_band_ppc",
        "common_age_band_ppc_plot_png",
        "common_age_band_ppc_plot_svg",
    ):
        assert paths[key].is_file()
    common_grid = pd.read_csv(paths["common_grid_ppc"])
    assert len(common_grid) == len(_exact_cells())
    assert set(common_grid["theta_resolution_baseline"]) == {"seven_band"}
    assert set(common_grid["theta_resolution_extension"]) == {"single_year"}
    assert (
        "extension_minus_baseline_absolute_standardized_residual" in common_grid.columns
    )
    summary = pd.read_csv(paths["common_grid_ppc_summary"])
    assert list(summary["comparison_role"]) == ["baseline", "extension"]
    assert (summary["n_exact_age_year_cells"] == len(_exact_cells())).all()
    assert summary["coverage_fraction"].between(0.0, 1.0).all()
    assert summary["evidence_scope"].str.contains("not held-out").all()
    comparison_config = json.loads(paths["config"].read_text(encoding="utf-8"))
    assert comparison_config["common_grid_ppc_comparison"] is True
    assert comparison_config["raw_loo_or_waic_compared"] is False


def test_compare_dsp004_with_dsp003_on_native_exact_grid(tmp_path: Path) -> None:
    baseline = tmp_path / "dsp004"
    extension = tmp_path / "dsp003"
    output = tmp_path / "comparison"
    _write_common_grid_run(
        baseline,
        model_id="DSP004",
        age_model="single_year",
        recording_model="constant",
    )
    _write_common_grid_run(
        extension,
        model_id="DSP003",
        age_model="single_year",
        recording_model="constant",
        age_reduction=True,
    )

    paths = compare_core_reduction_models.compare_core_model_outputs(
        baseline,
        extension,
        output,
    )

    assert paths["age_reduction"].is_file()
    assert paths["common_grid_ppc"].is_file()
    common_grid = pd.read_csv(paths["common_grid_ppc"])
    assert set(common_grid["theta_resolution_baseline"]) == {"single_year"}
    assert set(common_grid["theta_resolution_extension"]) == {"single_year"}
    assert set(common_grid["comparison_grid"]) == {
        "shared exact maternal-age by year cells"
    }
