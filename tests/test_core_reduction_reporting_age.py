"""Focused reporting tests for the DSP003 age-reduction extension."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dspopulations_us_birth_certificates.chance import get_ds_lb_nt_probability_array
from dspopulations_us_birth_certificates.selection.core_reduction import (
    CoreReductionPriors,
)
from dspopulations_us_birth_certificates.selection.core_reporting import (
    _prior_rho_table,
    age_reduction_by_year_table,
    reduction_prior_posterior_table,
    render_core_all,
)
from dspopulations_us_birth_certificates.selection.priors import AGE_LEVELS, logit


def test_prior_rho_table_reports_shifted_marginals_and_source_anchor() -> None:
    from scipy.integrate import quad
    from scipy.special import expit
    from scipy.stats import norm

    source = np.array([0.35, 0.40])
    shift = 0.2
    table = _prior_rho_table(
        {
            "reduction_mean": source,
            "reduction_logit": logit(source),
            "reduction_sigma": np.array([0.20, 0.45]),
            "reduction_calibration_shift_logit": shift,
            "reduction_error_correlation": 0.9,
        },
        interval_prob=0.89,
    )

    expected = 1.0 / (1.0 + np.exp(-(logit(source) + shift)))
    means = [
        quad(lambda x, mu=mu, sd=sd: expit(mu + sd * x) * norm.pdf(x), -10, 10)[0]
        for mu, sd in zip(logit(source) + shift, [0.20, 0.45], strict=True)
    ]
    assert table["rho_prior_mean"].to_numpy() == pytest.approx(means)
    assert table["rho_prior_centre"].to_numpy() == pytest.approx(expected)
    assert table["rho_surveillance_anchor_mean"].to_numpy() == pytest.approx(source)
    assert (table["reduction_calibration_shift_logit"] == shift).all()
    assert (table["reduction_error_correlation"] == 0.9).all()

    accounting = table.assign(
        year=[2020, 2021],
        rho_year_mean=[0.36, 0.41],
        rho_year_lo=[0.32, 0.36],
        rho_year_hi=[0.40, 0.46],
        eta_year_mean=[0.64, 0.59],
    )
    propagated = reduction_prior_posterior_table(accounting)
    assert "rho_surveillance_anchor_mean" in propagated
    assert "rho_prior_centre" in propagated
    assert "reduction_calibration_shift_odds_multiplier" in propagated


def _age_reduction_idata(variable: str) -> SimpleNamespace:
    rho_year = np.array([[[0.30, 0.40], [0.32, 0.42], [0.34, 0.44]]])
    offset = np.array([-0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06])
    rho_year_age = rho_year[..., None] + offset
    posterior = xr.Dataset(
        {
            "rho_year": (("chain", "draw", "year"), rho_year),
            "rho_year_anchor": (("chain", "draw", "year"), rho_year),
            variable: (
                ("chain", "draw", "year", "age"),
                rho_year_age,
            ),
            "rho_age_offset": (
                ("chain", "draw", "age"),
                np.broadcast_to(offset, (1, 3, len(AGE_LEVELS))),
            ),
        },
        coords={
            "chain": [0],
            "draw": np.arange(3),
            "year": np.arange(2),
            "age": np.arange(len(AGE_LEVELS)),
        },
    )
    return SimpleNamespace(posterior=posterior)


def _age_year_cells() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "year_idx": year,
                "age_idx": age,
                "N_cell": 1_000,
                "R_cell": 1,
            }
            for year in range(2)
            for age in range(len(AGE_LEVELS))
        ]
    )


@pytest.mark.parametrize("variable", ["rho_year_age", "rho_age_year"])
def test_age_reduction_table_reports_weighted_marginal(variable: str) -> None:
    priors = CoreReductionPriors(
        theta_lb_age=np.full(len(AGE_LEVELS), 0.01),
        reduction_mean=np.array([0.35, 0.40]),
        reduction_logit=logit(np.array([0.35, 0.40])),
        reduction_sigma=np.array([0.20, 0.20]),
    )

    table = age_reduction_by_year_table(
        _age_reduction_idata(variable),
        _age_year_cells(),
        priors.to_dict(),
        year_range=(2020, 2021),
    )

    assert len(table) == 2 * len(AGE_LEVELS)
    assert list(table["age"].drop_duplicates()) == AGE_LEVELS
    assert table.groupby("year")["natural_ds_weight_share"].sum().to_numpy() == (
        pytest.approx([1.0, 1.0])
    )
    assert table["rho_year_marginal_max_abs_draw_difference"].max() < 1e-12
    first = table.query("year == 2020 and age_idx == 0").iloc[0]
    assert first["rho_year_age_mean"] == pytest.approx(0.26)
    assert first["rho_year_marginal_mean"] == pytest.approx(0.32)
    assert first["rho_year_recomputed_mean"] == pytest.approx(0.32)
    assert first["rho_year_anchor_mean"] == pytest.approx(0.32)
    assert first["rho_age_offset_mean"] == pytest.approx(-0.06)


def test_age_reduction_table_uses_exact_maternal_age_labels() -> None:
    idata = _age_reduction_idata("rho_year_age")
    maternal_ages = np.array([12, 13, 14, 30, 48, 49, 50])
    maternal_age_labels = ["10-12", "13", "14", "30", "48", "49", "50+"]
    idata.posterior = idata.posterior.assign_coords(age=maternal_ages)
    cells = _age_year_cells()
    cells["maternal_age"] = maternal_ages[cells["age_idx"]]
    cells["maternal_age_label"] = np.asarray(maternal_age_labels)[cells["age_idx"]]
    priors = CoreReductionPriors(
        theta_lb_age=np.full(len(AGE_LEVELS), 0.01),
        reduction_mean=np.array([0.35, 0.40]),
        reduction_logit=logit(np.array([0.35, 0.40])),
        reduction_sigma=np.array([0.20, 0.20]),
    )

    table = age_reduction_by_year_table(
        idata,
        cells,
        priors.to_dict(),
        year_range=(2020, 2021),
    )

    assert list(table["age"].drop_duplicates()) == maternal_age_labels
    by_age = table.drop_duplicates("age_idx").set_index("maternal_age")
    assert bool(by_age.loc[12, "maternal_age_endpoint_capped"])
    assert bool(by_age.loc[50, "maternal_age_endpoint_capped"])
    assert not bool(by_age.loc[30, "sparse_boundary_age"])
    assert bool(by_age.loc[48, "sparse_boundary_age"])


def test_age_reduction_table_rejects_missing_age_reduction() -> None:
    idata = SimpleNamespace(
        posterior=xr.Dataset(
            {
                "rho_year": (
                    ("chain", "draw", "year"),
                    np.full((1, 2, 2), 0.4),
                )
            }
        )
    )
    priors = CoreReductionPriors(
        theta_lb_age=np.full(len(AGE_LEVELS), 0.01),
        reduction_mean=np.array([0.35, 0.40]),
        reduction_logit=logit(np.array([0.35, 0.40])),
        reduction_sigma=np.array([0.20, 0.20]),
    )

    with pytest.raises(ValueError, match="no age-specific reduction variable"):
        age_reduction_by_year_table(
            idata,
            _age_year_cells(),
            priors.to_dict(),
            year_range=(2020, 2021),
        )


def test_render_core_all_adds_exact_age_artifacts(tmp_path) -> None:
    maternal_ages = np.array([20, 30, 50])
    cells = pd.DataFrame(
        {
            "year_idx": [0, 0, 0, 1, 1, 1],
            "age_idx": [0, 1, 2, 0, 1, 2],
            "maternal_age": [20, 30, 50, 20, 30, 50],
            "maternal_age_label": ["20", "30", "50+", "20", "30", "50+"],
            "N_cell": [1_000, 800, 100, 900, 700, 120],
            "R_cell": [1, 2, 2, 1, 2, 3],
        }
    )
    theta = np.asarray(get_ds_lb_nt_probability_array(maternal_ages))
    year_age_n = np.array([[1_000, 800, 100], [900, 700, 120]], dtype=float)
    natural = year_age_n * theta[None, :]
    weights = natural / natural.sum(axis=1, keepdims=True)
    rho_year_age = np.array(
        [
            [
                [[0.20, 0.30, 0.50], [0.22, 0.32, 0.52]],
                [[0.21, 0.31, 0.51], [0.23, 0.33, 0.53]],
            ]
        ]
    )
    rho_year = (rho_year_age * weights[None, None, :, :]).sum(axis=-1)
    eta_year = 1.0 - rho_year
    true_count_year = natural.sum(axis=1)[None, None, :] * eta_year
    posterior = xr.Dataset(
        {
            "rho_year_age": (
                ("chain", "draw", "year", "age"),
                rho_year_age,
            ),
            "rho_year": (("chain", "draw", "year"), rho_year),
            "rho_year_anchor": (("chain", "draw", "year"), rho_year),
            "rho_age_offset": (
                ("chain", "draw", "age"),
                np.zeros((1, 2, 3)),
            ),
            "eta_year": (("chain", "draw", "year"), eta_year),
            "recording_s": (("chain", "draw"), np.full((1, 2), 0.4)),
            "recording_s_year": (
                ("chain", "draw", "year"),
                np.full((1, 2, 2), 0.4),
            ),
            "true_count_year": (
                ("chain", "draw", "year"),
                true_count_year,
            ),
            "recorded_count_year_mu": (
                ("chain", "draw", "year"),
                true_count_year * 0.4,
            ),
            "true_count_total": (
                ("chain", "draw"),
                true_count_year.sum(axis=-1),
            ),
        },
        coords={
            "chain": [0],
            "draw": [0, 1],
            "year": [0, 1],
            "age": maternal_ages,
        },
    )
    posterior_predictive = xr.Dataset(
        {
            "R_obs": (
                ("chain", "draw", "cell"),
                np.array([[[1, 2, 2, 1, 2, 3], [1, 1, 3, 1, 2, 4]]]),
            )
        },
        coords={"chain": [0], "draw": [0, 1], "cell": np.arange(len(cells))},
    )
    constant_data = xr.Dataset(
        {"theta_lb_age": (("age",), theta)},
        coords={"age": maternal_ages},
    )
    idata = SimpleNamespace(
        posterior=posterior,
        posterior_predictive=posterior_predictive,
        constant_data=constant_data,
    )
    priors = CoreReductionPriors(
        reduction_mean=np.array([0.35, 0.40]),
        reduction_logit=logit(np.array([0.35, 0.40])),
        reduction_sigma=np.array([0.20, 0.20]),
    )

    tables = render_core_all(
        idata,
        cells,
        tmp_path,
        priors_config=priors.to_dict(),
        year_range=(2020, 2021),
    )

    assert "core_reduction_by_age_year" in tables
    assert (tmp_path / "tables" / "core_reduction_by_age_year.csv").is_file()
    assert (tmp_path / "plots" / "core_reduction_by_age_year.png").is_file()
    assert (tmp_path / "plots" / "core_reduction_by_age_year.svg").is_file()
    assert (tmp_path / "tables" / "core_ppc_by_age_year.csv").is_file()
    assert (tmp_path / "plots" / "core_ppc_by_age_year.png").is_file()
    assert (tmp_path / "plots" / "core_ppc_by_age_year.svg").is_file()
    assert (tmp_path / "tables" / "core_ppc_by_age_band.csv").is_file()
    assert (tmp_path / "plots" / "core_ppc_by_age_band.png").is_file()
    assert (tmp_path / "plots" / "core_ppc_by_age_band.svg").is_file()
    age_table = tables["core_reduction_by_age_year"]
    assert list(age_table["age"].drop_duplicates()) == ["20", "30", "50+"]
    ppc_age = tables["core_ppc_by_age"]
    assert list(ppc_age["label"]) == ["20", "30", "50+"]
    ppc_age_year = tables["core_ppc_by_age_year"]
    assert len(ppc_age_year) == 6
    assert {
        "residual_observed_minus_predicted",
        "standardized_residual",
        "relative_residual",
        "observed_in_interval",
    }.issubset(ppc_age_year.columns)
    assert list(tables["core_ppc_by_age_band"]["label"]) == [
        "20-24",
        "30-34",
        "45+",
    ]
    assert age_table["rho_year_marginal_max_abs_draw_difference"].max() < 1e-12

    common_rho_idata = SimpleNamespace(
        posterior=posterior.drop_vars(
            ["rho_year_age", "rho_year_anchor", "rho_age_offset"]
        ),
        posterior_predictive=posterior_predictive,
        constant_data=constant_data,
    )
    common_rho_output = tmp_path / "common_rho"
    common_rho_tables = render_core_all(
        common_rho_idata,
        cells,
        common_rho_output,
        priors_config=priors.to_dict(),
        year_range=(2020, 2021),
    )
    assert "core_reduction_by_age_year" not in common_rho_tables
    assert "core_ppc_by_age_year" in common_rho_tables
    assert "core_ppc_by_age_band" in common_rho_tables
    assert (common_rho_output / "plots" / "core_ppc_by_age_year.png").is_file()
