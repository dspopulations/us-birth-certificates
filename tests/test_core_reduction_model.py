"""Tests for the core age-reduction-recording model."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import duckdb
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dspopulations_us_birth_certificates.selection.core_models import (
    CORE_MODEL_REGISTRY,
    core_model_names,
    get_core_model_definition,
)
from dspopulations_us_birth_certificates.selection.core_reduction import (
    CoreReductionPriors,
    build_core_reduction_model,
    prepare_core_age_year_cells,
)
from dspopulations_us_birth_certificates.selection.core_reporting import render_core_all
from dspopulations_us_birth_certificates.selection.priors import logit


def test_core_model_registry_indexes_dsp_models() -> None:
    assert core_model_names() == ("DSP001", "DSP002")
    assert set(CORE_MODEL_REGISTRY) == {"dsp001", "dsp002"}
    assert get_core_model_definition("dsp001").recording_model == "constant"
    assert get_core_model_definition("DSP002").recording_model == "year"
    assert get_core_model_definition("DSP002").comparison_parent == "DSP001"


def _make_row(
    *,
    year: int,
    mage_c: int | None,
    down_ind: int | None = 0,
) -> dict[str, int | None]:
    return {"year": year, "mage_c": mage_c, "down_ind": down_ind}


def test_prepare_core_age_year_cells_keeps_clinical_missing_rows(
    tmp_path: Path,
) -> None:
    rows = [
        _make_row(year=2020, mage_c=19, down_ind=0),
        _make_row(year=2020, mage_c=32, down_ind=1),
        _make_row(year=2020, mage_c=47, down_ind=0),
        _make_row(year=2020, mage_c=None, down_ind=1),
        _make_row(year=2020, mage_c=30, down_ind=None),
        _make_row(year=2019, mage_c=30, down_ind=1),
    ]
    db = tmp_path / "core.db"
    con = duckdb.connect(str(db))
    con.register("_rows", pd.DataFrame(rows))
    con.execute("CREATE TABLE us_births AS SELECT * FROM _rows")
    con.unregister("_rows")
    con.close()

    con = duckdb.connect(str(db), read_only=True)
    try:
        cells = prepare_core_age_year_cells(con, year_range=(2020, 2020))
    finally:
        con.close()

    assert set(cells.columns) == {"year_idx", "age_idx", "N_cell", "R_cell"}
    assert cells.attrs["n_year"] == 1
    assert cells["N_cell"].sum() == 3
    assert cells["R_cell"].sum() == 1
    assert sorted(cells["age_idx"].tolist()) == [0, 3, 6]


def test_core_reduction_priors_from_csv(tmp_path: Path) -> None:
    path = tmp_path / "reduction.csv"
    pd.DataFrame({"year": [2020, 2021], "reduction": [0.35, 0.40]}).to_csv(
        path, index=False
    )

    priors = CoreReductionPriors.from_reduction_csv(
        year_range=(2020, 2021),
        path=path,
        observed_logit_sigma=0.2,
        extrapolated_logit_sigma=0.5,
        extrapolated_start=2021,
    )

    assert np.allclose(priors.reduction_mean, [0.35, 0.40])
    assert np.allclose(priors.reduction_sigma, [0.2, 0.5])


def test_core_reduction_priors_allow_zero_s_year_sigma_for_constant_model(
    tmp_path: Path,
) -> None:
    path = tmp_path / "reduction.csv"
    pd.DataFrame({"year": [2020, 2021], "reduction": [0.35, 0.40]}).to_csv(
        path, index=False
    )

    priors = CoreReductionPriors.from_reduction_csv(
        year_range=(2020, 2021),
        path=path,
        recording_s_year_sigma=0.0,
    )

    assert priors.recording_s_year_sigma == 0.0


def test_build_core_reduction_model_and_prior_predictive() -> None:
    pm = pytest.importorskip("pymc")

    cells = pd.DataFrame(
        {
            "year_idx": [0, 0, 1, 1],
            "age_idx": [2, 4, 2, 4],
            "N_cell": [1000, 800, 900, 700],
            "R_cell": [1, 3, 1, 4],
        }
    )
    cells.attrs["n_year"] = 2
    cells.attrs["year_range"] = (2020, 2021)
    priors = CoreReductionPriors(
        reduction_mean=np.array([0.35, 0.40]),
        reduction_logit=logit(np.array([0.35, 0.40])),
        reduction_sigma=np.array([0.25, 0.35]),
    )

    model = build_core_reduction_model(cells, priors, n_year=2)
    named = {rv.name for rv in model.free_RVs}
    assert named == {"rho_logit_year", "recording_s_logit"}

    with model:
        prior = pm.sample_prior_predictive(draws=5, random_seed=0)

    r = np.asarray(prior.prior_predictive["R_obs"].values)
    assert r.shape[-1] == len(cells)
    assert (r >= 0).all()
    assert (r <= cells["N_cell"].to_numpy()[None, None, :]).all()


def test_build_core_reduction_model_with_s_year_extension() -> None:
    pm = pytest.importorskip("pymc")

    cells = pd.DataFrame(
        {
            "year_idx": [0, 0, 1, 1],
            "age_idx": [2, 4, 2, 4],
            "N_cell": [1000, 800, 900, 700],
            "R_cell": [1, 3, 1, 4],
        }
    )
    cells.attrs["n_year"] = 2
    cells.attrs["year_range"] = (2020, 2021)
    priors = CoreReductionPriors(
        reduction_mean=np.array([0.35, 0.40]),
        reduction_logit=logit(np.array([0.35, 0.40])),
        reduction_sigma=np.array([0.25, 0.35]),
        recording_s_year_sigma=0.25,
    )

    model = build_core_reduction_model(
        cells,
        priors,
        n_year=2,
        recording_model="year",
    )
    named = {rv.name for rv in model.free_RVs}
    assert named == {
        "rho_logit_year",
        "recording_s_logit",
        "recording_s_year_offset_raw",
    }
    assert "recording_s_year" in model.named_vars
    assert "recording_s_year_offset" in model.named_vars

    with model:
        prior = pm.sample_prior_predictive(draws=5, random_seed=0)

    r = np.asarray(prior.prior_predictive["R_obs"].values)
    s_year = np.asarray(prior.prior["recording_s_year"].values)
    assert r.shape[-1] == len(cells)
    assert s_year.shape[-1] == 2
    assert ((0.0 < s_year) & (s_year < 1.0)).all()


def test_build_core_reduction_model_s_year_requires_positive_year_sigma() -> None:
    cells = pd.DataFrame(
        {
            "year_idx": [0, 0, 1, 1],
            "age_idx": [2, 4, 2, 4],
            "N_cell": [1000, 800, 900, 700],
            "R_cell": [1, 3, 1, 4],
        }
    )
    cells.attrs["n_year"] = 2
    priors = CoreReductionPriors(
        reduction_mean=np.array([0.35, 0.40]),
        reduction_logit=logit(np.array([0.35, 0.40])),
        reduction_sigma=np.array([0.25, 0.35]),
        recording_s_year_sigma=0.0,
    )

    with pytest.raises(ValueError, match="recording_s_year_sigma must be positive"):
        build_core_reduction_model(
            cells,
            priors,
            n_year=2,
            recording_model="year",
        )


def test_render_core_report_outputs(tmp_path: Path) -> None:
    cells = pd.DataFrame(
        {
            "year_idx": [0, 0, 1, 1],
            "age_idx": [2, 4, 2, 4],
            "N_cell": [1000, 800, 900, 700],
            "R_cell": [1, 3, 1, 4],
        }
    )
    chain = np.arange(2)
    draw = np.arange(3)
    year = np.arange(2)
    cell = np.arange(len(cells))
    rho = np.array(
        [
            [[0.30, 0.35], [0.31, 0.36], [0.32, 0.37]],
            [[0.29, 0.34], [0.30, 0.35], [0.31, 0.36]],
        ]
    )
    eta = 1.0 - rho
    posterior = xr.Dataset(
        {
            "rho_year": (("chain", "draw", "year"), rho),
            "eta_year": (("chain", "draw", "year"), eta),
            "recording_s": (("chain", "draw"), np.full((2, 3), 0.4)),
            "recording_s_year": (
                ("chain", "draw", "year"),
                np.full((2, 3, 2), 0.4),
            ),
            "true_count_year": (("chain", "draw", "year"), 1000 * eta),
            "recorded_count_year_mu": (("chain", "draw", "year"), 400 * eta),
            "true_count_total": (("chain", "draw"), (1000 * eta).sum(axis=2)),
        },
        coords={"chain": chain, "draw": draw, "year": year},
    )
    ppc = xr.Dataset(
        {
            "R_obs": (
                ("chain", "draw", "cell"),
                np.array(
                    [
                        [[1, 3, 1, 4], [2, 2, 2, 4], [1, 4, 1, 5]],
                        [[1, 3, 2, 3], [1, 2, 1, 4], [2, 3, 1, 4]],
                    ]
                ),
            )
        },
        coords={"chain": chain, "draw": draw, "cell": cell},
    )
    idata = SimpleNamespace(posterior=posterior, posterior_predictive=ppc)
    priors = CoreReductionPriors(
        reduction_mean=np.array([0.35, 0.40]),
        reduction_logit=logit(np.array([0.35, 0.40])),
        reduction_sigma=np.array([0.20, 0.45]),
    )

    tables = render_core_all(
        idata,
        cells,
        tmp_path,
        priors_config=priors.to_dict(),
        year_range=(2020, 2021),
    )

    expected = {
        "core_headlines",
        "core_accounting_by_year",
        "core_reduction_prior_posterior",
        "core_recording_s",
        "core_recording_s_by_year",
        "core_ppc_by_year",
        "core_ppc_by_age",
    }
    assert expected == set(tables)
    for stem in expected:
        assert (tmp_path / "tables" / f"{stem}.csv").is_file()
        assert len(pd.read_csv(tmp_path / "tables" / f"{stem}.csv")) >= 1
    for stem in (
        "core_accounting_by_year",
        "core_reduction_prior_posterior",
        "core_recording_s",
        "core_recording_s_by_year",
        "core_ppc_by_year",
        "core_ppc_by_age",
    ):
        assert (tmp_path / "plots" / f"{stem}.png").is_file()
        assert (tmp_path / "plots" / f"{stem}.svg").is_file()
