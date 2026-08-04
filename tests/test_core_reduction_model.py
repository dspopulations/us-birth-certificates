"""Tests for the core age-reduction-recording model."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dspopulations_us_birth_certificates.chance import get_ds_lb_nt_probability_array
from dspopulations_us_birth_certificates.selection.core_models import (
    CORE_MODEL_REGISTRY,
    DSP003,
    DSP004,
    DSP005,
    DSP008,
    DSP009,
    CoreModelDefinition,
    core_model_names,
    get_core_model_definition,
    validate_core_model_definition,
)
from dspopulations_us_birth_certificates.selection.core_reduction import (
    CoreReductionModelConfig,
    CoreReductionPriors,
    SurveillanceAnchor,
    _reduction_error_covariance,
    build_core_reduction_model,
    prepare_core_age_year_cells,
)
from dspopulations_us_birth_certificates.selection.core_reporting import render_core_all
from dspopulations_us_birth_certificates.selection.priors import logit


def test_core_model_registry_indexes_dsp_models() -> None:
    assert core_model_names() == (
        "DSP001",
        "DSP002",
        "DSP003",
        "DSP004",
        "DSP005",
        "DSP006",
        "DSP007",
        "DSP008",
        "DSP009",
    )
    assert set(CORE_MODEL_REGISTRY) == {
        "dsp001",
        "dsp002",
        "dsp003",
        "dsp004",
        "dsp005",
        "dsp006",
        "dsp007",
        "dsp008",
        "dsp009",
    }
    assert get_core_model_definition("dsp001").recording_model == "constant"
    assert get_core_model_definition("DSP002").recording_model == "year"
    assert get_core_model_definition("DSP002").comparison_parent == "DSP001"
    assert get_core_model_definition("DSP003").recording_model == "constant"
    assert get_core_model_definition("DSP003").reduction_model == "year_age"
    assert get_core_model_definition("DSP003").age_model == "single_year"
    assert get_core_model_definition("DSP003").comparison_parent == "DSP001"
    assert DSP004.recording_model == "constant"
    assert DSP004.reduction_model == "year"
    assert DSP004.age_model == "single_year"
    assert DSP004.comparison_parent == "DSP001"
    assert DSP005.recording_model == "year"
    assert DSP005.reduction_model == "year"
    assert DSP005.age_model == "single_year"
    assert DSP005.comparison_parent == "DSP004"
    # DSP009 is DSP008 plus post-window recording drift, and nothing else.
    assert DSP009.recording_model == DSP008.recording_model == "revision"
    assert DSP009.reduction_model == DSP008.reduction_model == "anchor"
    assert DSP009.age_model == DSP008.age_model == "single_year"
    assert DSP008.recording_drift == "none"
    assert DSP009.recording_drift == "post_anchor"
    assert DSP009.comparison_parent == "DSP008"
    # Every other model must stay undrifted, so the drift cannot be introduced
    # silently by editing a shared default.
    assert {
        definition.model_id
        for definition in CORE_MODEL_REGISTRY.values()
        if definition.recording_drift == "post_anchor"
    } == {"DSP009"}


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"recording_drift": "sometimes"}, "recording_drift must be"),
        # Without an anchor there is no last covered year to drift from.
        ({"reduction_model": "year"}, "reduction_model='anchor'"),
        # Centred year offsets and a post-anchor walk are the same parameter.
        ({"recording_model": "year"}, "cannot combine"),
    ],
)
def test_validate_core_model_rejects_incoherent_drift(
    overrides: dict[str, str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_core_model_definition(replace(DSP009, **overrides))


def _make_row(
    *,
    year: int,
    mage_c: int | None,
    down_ind: int | None = 0,
    ca_down_c: str | None = None,
    ca_down: str | None = None,
    ca_downs: str | None = None,
) -> dict[str, int | str | None]:
    """One synthetic birth record.

    ``ca_down``/``ca_downs`` are the 2003-revision anomaly checkbox fields; a
    record with either populated is a revised certificate. Leaving both None
    represents an unrevised record.
    """
    return {
        "year": year,
        "mage_c": mage_c,
        "down_ind": down_ind,
        "ca_down_c": ca_down_c,
        "ca_down": ca_down,
        "ca_downs": ca_downs,
    }


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


def test_prepare_core_single_year_cells_pools_endpoint_age_codes(
    tmp_path: Path,
) -> None:
    rows = [
        _make_row(year=2020, mage_c=10, down_ind=0),
        _make_row(year=2020, mage_c=11, down_ind=1),
        _make_row(year=2020, mage_c=12, down_ind=0),
        _make_row(year=2020, mage_c=13, down_ind=1),
        _make_row(year=2020, mage_c=50, down_ind=0),
        _make_row(year=2020, mage_c=51, down_ind=1),
        _make_row(year=2020, mage_c=54, down_ind=0),
    ]
    db = tmp_path / "core.db"
    con = duckdb.connect(str(db))
    con.register("_rows", pd.DataFrame(rows))
    con.execute("CREATE TABLE us_births AS SELECT * FROM _rows")
    con.unregister("_rows")
    con.close()

    con = duckdb.connect(str(db), read_only=True)
    try:
        cells = prepare_core_age_year_cells(
            con,
            year_range=(2020, 2020),
            age_model="single_year",
        )
    finally:
        con.close()

    assert cells["maternal_age"].tolist() == [12, 13, 50]
    assert cells["age_idx"].tolist() == [0, 1, 2]
    assert cells["maternal_age_label"].tolist() == ["10-12", "13", "50+"]
    assert cells["N_cell"].tolist() == [3, 1, 3]
    assert cells["R_cell"].tolist() == [1, 1, 1]
    assert cells.attrs["age_values"] == [12, 13, 50]
    assert cells.attrs["age_labels"] == ["10-12", "13", "50+"]


def test_prepare_core_confirmed_only_keeps_denominator(tmp_path: Path) -> None:
    rows = [
        _make_row(year=2020, mage_c=30, down_ind=1, ca_down_c="C"),
        _make_row(year=2020, mage_c=30, down_ind=1, ca_down_c="P"),
        _make_row(year=2020, mage_c=30, down_ind=0, ca_down_c="N"),
    ]
    db = tmp_path / "core.db"
    con = duckdb.connect(str(db))
    con.register("_rows", pd.DataFrame(rows))
    con.execute("CREATE TABLE us_births AS SELECT * FROM _rows")
    con.unregister("_rows")
    con.close()

    con = duckdb.connect(str(db), read_only=True)
    try:
        combined = prepare_core_age_year_cells(con, year_range=(2020, 2020))
        confirmed = prepare_core_age_year_cells(
            con,
            year_range=(2020, 2020),
            recorded_definition="confirmed_only",
        )
    finally:
        con.close()

    assert combined["N_cell"].sum() == confirmed["N_cell"].sum() == 3
    assert combined["R_cell"].sum() == 2
    assert confirmed["R_cell"].sum() == 1
    assert confirmed.attrs["recorded_definition"] == "confirmed_only"


def test_prepare_core_cells_split_revision_preserves_totals(tmp_path: Path) -> None:
    rows = [
        _make_row(year=2010, mage_c=30, down_ind=1, ca_down="Y"),
        _make_row(year=2010, mage_c=30, down_ind=0, ca_downs="N"),
        _make_row(year=2010, mage_c=30, down_ind=1),
        _make_row(year=2010, mage_c=30, down_ind=0),
    ]
    db = tmp_path / "core.db"
    con = duckdb.connect(str(db))
    con.register("_rows", pd.DataFrame(rows))
    con.execute("CREATE TABLE us_births AS SELECT * FROM _rows")
    con.unregister("_rows")
    con.close()

    con = duckdb.connect(str(db), read_only=True)
    try:
        pooled = prepare_core_age_year_cells(con, year_range=(2010, 2010))
        split = prepare_core_age_year_cells(
            con,
            year_range=(2010, 2010),
            split_revision=True,
        )
    finally:
        con.close()

    # Splitting must reallocate rows, never create or drop them.
    assert pooled["N_cell"].sum() == split["N_cell"].sum() == 4
    assert pooled["R_cell"].sum() == split["R_cell"].sum() == 2
    assert "revised" not in pooled
    assert pooled.attrs["split_revision"] is False
    assert split.attrs["split_revision"] is True
    assert sorted(split["revised"].tolist()) == [0, 1]
    by_revision = split.set_index("revised")
    assert by_revision.loc[1, "N_cell"] == 2
    assert by_revision.loc[1, "R_cell"] == 1
    assert by_revision.loc[0, "N_cell"] == 2
    assert by_revision.loc[0, "R_cell"] == 1


def _revision_cells() -> pd.DataFrame:
    cells = pd.DataFrame(
        {
            "year_idx": [0, 0, 0, 0, 1, 1, 1, 1],
            "age_idx": [2, 2, 4, 4, 2, 2, 4, 4],
            "revised": [0, 1, 0, 1, 0, 1, 0, 1],
            "N_cell": [500, 500, 400, 400, 300, 600, 200, 500],
            "R_cell": [1, 2, 1, 3, 1, 2, 1, 3],
        }
    )
    cells.attrs["n_year"] = 2
    cells.attrs["year_range"] = (2020, 2021)
    return cells


def test_build_core_reduction_model_with_revision_split() -> None:
    pm = pytest.importorskip("pymc")

    cells = _revision_cells()
    priors = CoreReductionPriors(
        reduction_mean=np.array([0.35, 0.40]),
        reduction_logit=logit(np.array([0.35, 0.40])),
        reduction_sigma=np.array([0.25, 0.35]),
    )

    model = build_core_reduction_model(
        cells,
        priors,
        n_year=2,
        recording_model="revision",
    )
    named = {rv.name for rv in model.free_RVs}
    assert named == {
        "rho_logit_year",
        "recording_s_logit",
        "recording_s_unrevised_offset",
    }
    assert "recording_s_unrevised" in model.named_vars

    with model:
        prior = pm.sample_prior_predictive(draws=8, random_seed=0)

    recorded = np.asarray(prior.prior_predictive["R_obs"].values)
    assert recorded.shape[-1] == len(cells)
    for name in ("recording_s", "recording_s_unrevised"):
        drawn = np.asarray(prior.prior[name].values)
        assert ((0.0 < drawn) & (drawn < 1.0)).all()

    # Cells 0-3 are (age 2, unrevised), (age 2, revised), (age 4, unrevised),
    # (age 4, revised) within year 0. The revision must shift p_recorded within
    # an age, in the same direction at both ages, since one offset drives both.
    p_recorded = np.asarray(prior.prior["p_recorded"].values)[0, 0]
    assert p_recorded[0] != p_recorded[1]
    assert np.sign(p_recorded[1] - p_recorded[0]) == np.sign(
        p_recorded[3] - p_recorded[2]
    )


def test_revision_recording_model_requires_split_cells() -> None:
    pytest.importorskip("pymc")

    cells = _revision_cells().drop(columns=["revised"])
    priors = CoreReductionPriors(
        reduction_mean=np.array([0.35, 0.40]),
        reduction_logit=logit(np.array([0.35, 0.40])),
        reduction_sigma=np.array([0.25, 0.35]),
    )

    with pytest.raises(ValueError, match="split_revision=True"):
        build_core_reduction_model(
            cells,
            priors,
            n_year=2,
            recording_model="revision",
        )


def test_revision_split_leaves_true_counts_unchanged() -> None:
    """The split changes only recording, so true counts must be invariant."""
    pytest.importorskip("pymc")

    priors = CoreReductionPriors(
        reduction_mean=np.array([0.35, 0.40]),
        reduction_logit=logit(np.array([0.35, 0.40])),
        reduction_sigma=np.array([0.25, 0.35]),
    )
    split = _revision_cells()
    pooled = (
        split.groupby(["year_idx", "age_idx"], as_index=False)[["N_cell", "R_cell"]]
        .sum()
        .astype({"N_cell": "int64", "R_cell": "int64"})
    )
    pooled.attrs.update(split.attrs)

    theta = np.array([0.0007, 0.0008, 0.0009, 0.0015, 0.0047, 0.0152, 0.0307])
    totals = []
    for frame, recording_model in ((pooled, "constant"), (split, "revision")):
        model = build_core_reduction_model(
            frame,
            priors,
            n_year=2,
            recording_model=recording_model,
        )
        n_cell = frame["N_cell"].to_numpy(dtype=float)
        eta = 1.0 - priors.reduction_mean
        p = theta[frame["age_idx"].to_numpy()] * eta[frame["year_idx"].to_numpy()]
        totals.append(float(np.dot(n_cell, p)))
        assert "true_count_total" in model.named_vars
    assert np.isclose(totals[0], totals[1])


def _anchor_csv(tmp_path: Path, mid_years: list[int], prevalences: list[float]) -> Path:
    path = tmp_path / "anchor.csv"
    pd.DataFrame({"mid_year": mid_years, "prevalence_per10k": prevalences}).to_csv(
        path, index=False
    )
    return path


def test_surveillance_anchor_from_csv_selects_windows_inside_range(
    tmp_path: Path,
) -> None:
    path = _anchor_csv(
        tmp_path,
        [2002, 2004, 2006, 2020],
        [12.0, 12.5, 13.0, 14.0],
    )
    anchor = SurveillanceAnchor.from_csv(year_range=(2004, 2010), path=path)

    # 2002 and 2020 are centred outside the range and must be dropped rather
    # than partially applied.
    assert anchor.mid_years == (2004, 2006)
    assert anchor.mid_year_idx.tolist() == [0, 2]
    # Prevalence is converted from per-10,000 to per live birth.
    assert np.allclose(np.exp(anchor.log_prevalence), [12.5e-4, 13.0e-4])
    assert anchor.to_dict()["n_windows"] == 2


def test_surveillance_anchor_rejects_empty_selection(tmp_path: Path) -> None:
    path = _anchor_csv(tmp_path, [1990, 1991], [10.0, 10.5])
    with pytest.raises(ValueError, match="no surveillance window centred"):
        SurveillanceAnchor.from_csv(year_range=(2016, 2024), path=path)


def test_surveillance_anchor_discounts_overlapping_windows(tmp_path: Path) -> None:
    """Seventeen overlapping five-year windows are not seventeen observations."""
    mid_years = list(range(2000, 2017))
    path = _anchor_csv(tmp_path, mid_years, [12.5] * len(mid_years))
    anchor = SurveillanceAnchor.from_csv(year_range=(2000, 2016), path=path)
    # 17 mid-years span 2000-2016, plus two padding years either side = 21
    # birth-years, over a five-year window width.
    assert anchor.to_dict()["effective_independent_windows"] == pytest.approx(21 / 5)


def _anchor_cells(n_year: int) -> pd.DataFrame:
    rows = []
    for year in range(n_year):
        for age_idx, maternal_age in enumerate((30, 40)):
            rows.append(
                {
                    "year_idx": year,
                    "age_idx": age_idx,
                    "maternal_age": maternal_age,
                    "N_cell": 200_000 if maternal_age == 30 else 50_000,
                    "R_cell": 30 if maternal_age == 30 else 60,
                }
            )
    cells = pd.DataFrame(rows)
    cells.attrs["n_year"] = n_year
    cells.attrs["year_range"] = (2004, 2004 + n_year - 1)
    return cells


def test_anchored_model_derives_eta_from_prevalence(tmp_path: Path) -> None:
    """eta must equal anchored prevalence over the Morris expectation."""
    pm = pytest.importorskip("pymc")

    n_year = 9
    cells = _anchor_cells(n_year)
    path = _anchor_csv(tmp_path, [2006, 2008], [12.5, 13.0])
    anchor = SurveillanceAnchor.from_csv(year_range=(2004, 2012), path=path)
    priors = CoreReductionPriors()

    model = build_core_reduction_model(
        cells,
        priors,
        n_year=n_year,
        recording_model="constant",
        reduction_model="anchor",
        anchor=anchor,
    )
    named = {rv.name for rv in model.free_RVs}
    # No rho_logit_year: the reduction prior does not enter an anchored fit.
    assert "rho_logit_year" not in named
    assert {
        "anchor_log_level_start",
        "anchor_log_slope_start",
        "anchor_level_sigma",
        "anchor_trend_sigma",
        "anchor_obs_sigma",
        "anchor_level_innovation_raw",
        "anchor_trend_innovation_raw",
        "recording_s_logit",
    } <= named

    with model:
        prior = pm.sample_prior_predictive(draws=6, random_seed=0)

    prevalence = np.asarray(prior.prior["prevalence_year"].values)
    eta = np.asarray(prior.prior["eta_year"].values)
    natural = np.asarray(
        model.named_vars["natural_prevalence_year"].get_value()  # type: ignore[union-attr]
    )
    assert prevalence.shape[-1] == n_year
    assert np.allclose(eta, prevalence / natural, rtol=1e-10)
    assert np.allclose(
        np.asarray(prior.prior["rho_year"].values), 1.0 - eta, rtol=1e-10
    )
    # The latent series is padded so an edge-centred window is still usable.
    latent = np.asarray(prior.prior["anchor_log_prevalence_latent"].values)
    assert latent.shape[-1] == n_year + 2 * anchor.half_width


def test_anchored_window_prevalence_is_the_centred_mean(tmp_path: Path) -> None:
    """Each window must constrain the mean of its own five latent years."""
    pm = pytest.importorskip("pymc")

    n_year = 9
    path = _anchor_csv(tmp_path, [2006, 2008], [12.5, 13.0])
    anchor = SurveillanceAnchor.from_csv(year_range=(2004, 2012), path=path)
    model = build_core_reduction_model(
        _anchor_cells(n_year),
        CoreReductionPriors(),
        n_year=n_year,
        recording_model="constant",
        reduction_model="anchor",
        anchor=anchor,
    )
    with model:
        prior = pm.sample_prior_predictive(draws=4, random_seed=1)

    latent = np.exp(np.asarray(prior.prior["anchor_log_prevalence_latent"].values))
    windows = np.asarray(prior.prior["anchor_window_prevalence"].values)
    width = 2 * anchor.half_width + 1
    for position, start in enumerate(anchor.mid_year_idx):
        expected = latent[..., start : start + width].mean(axis=-1)
        assert np.allclose(windows[..., position], expected, rtol=1e-10)


def test_anchored_model_requires_an_anchor() -> None:
    pytest.importorskip("pymc")
    with pytest.raises(ValueError, match="requires a SurveillanceAnchor"):
        build_core_reduction_model(
            _anchor_cells(5),
            CoreReductionPriors(
                reduction_mean=np.full(5, 0.35),
                reduction_logit=logit(np.full(5, 0.35)),
                reduction_sigma=np.full(5, 0.2),
            ),
            n_year=5,
            reduction_model="anchor",
        )


def test_unanchored_model_rejects_a_stray_anchor(tmp_path: Path) -> None:
    pytest.importorskip("pymc")
    path = _anchor_csv(tmp_path, [2006], [12.5])
    anchor = SurveillanceAnchor.from_csv(year_range=(2004, 2012), path=path)
    with pytest.raises(ValueError, match="pass reduction_model='anchor'"):
        build_core_reduction_model(
            _anchor_cells(9),
            CoreReductionPriors(
                reduction_mean=np.full(9, 0.35),
                reduction_logit=logit(np.full(9, 0.35)),
                reduction_sigma=np.full(9, 0.2),
            ),
            n_year=9,
            reduction_model="year",
            anchor=anchor,
        )


def test_anchor_obs_sigma_can_be_fixed(tmp_path: Path) -> None:
    """Fixing the observation SD is the sensitivity axis for accuracy."""
    pytest.importorskip("pymc")
    path = _anchor_csv(tmp_path, [2006, 2008], [12.5, 13.0])
    anchor = SurveillanceAnchor.from_csv(year_range=(2004, 2012), path=path)
    model = build_core_reduction_model(
        _anchor_cells(9),
        CoreReductionPriors(),
        n_year=9,
        reduction_model="anchor",
        anchor=anchor,
        anchor_obs_sigma_fixed=0.1,
    )
    named = {rv.name for rv in model.free_RVs}
    assert "anchor_obs_sigma" not in named
    assert "anchor_obs_sigma" in model.named_vars


def _drift_cells(n_year: int) -> pd.DataFrame:
    """Anchored exact-age cells split by certificate revision, as DSP009 needs."""
    rows = []
    for year in range(n_year):
        for age_idx, maternal_age in enumerate((30, 40)):
            for revised in (0, 1):
                rows.append(
                    {
                        "year_idx": year,
                        "age_idx": age_idx,
                        "maternal_age": maternal_age,
                        "revised": revised,
                        "N_cell": 100_000 if maternal_age == 30 else 25_000,
                        "R_cell": 15 if maternal_age == 30 else 30,
                    }
                )
    cells = pd.DataFrame(rows)
    cells.attrs["n_year"] = n_year
    cells.attrs["year_range"] = (2004, 2004 + n_year - 1)
    return cells


def _drift_model(
    tmp_path: Path,
    *,
    n_year: int = 13,
    mid_years: list[int] | None = None,
    drift_sigma: float = 0.06,
    recording_model: str = "revision",
    anchor_forecast_flat: bool = False,
    recording_drift: str = "post_anchor",
) -> tuple[Any, SurveillanceAnchor]:
    path = _anchor_csv(tmp_path, mid_years or [2006, 2008], [12.5, 13.0])
    anchor = SurveillanceAnchor.from_csv(
        year_range=(2004, 2004 + n_year - 1), path=path
    )
    model = build_core_reduction_model(
        _drift_cells(n_year),
        CoreReductionPriors(recording_s_drift_sigma=drift_sigma),
        n_year=n_year,
        recording_model=recording_model,
        reduction_model="anchor",
        recording_drift=recording_drift,
        anchor=anchor,
        anchor_forecast_flat=anchor_forecast_flat,
    )
    return model, anchor


def test_post_anchor_drift_starts_after_the_last_covered_year(tmp_path: Path) -> None:
    """The drift must be exactly zero for every year a window still reaches."""
    pytest.importorskip("pymc")

    n_year = 13
    model, anchor = _drift_model(tmp_path, n_year=n_year)
    named = {rv.name for rv in model.free_RVs}
    assert "recording_s_drift_innovation_raw" in named

    # Windows centred on 2006 and 2008 with half-width 2 reach 2010, model-year
    # index 6, so 2011-2016 (indices 7-12) are the unanchored tail.
    last_anchored = int(anchor.mid_year_idx.max()) + anchor.half_width
    assert last_anchored == 6
    n_drift = n_year - 1 - last_anchored
    assert n_drift == 6
    assert model["recording_s_drift_innovation_raw"].eval().shape == (n_drift,)

    drift = np.asarray(model["recording_s_drift_logit"].eval())
    assert drift.shape == (n_year,)
    assert np.array_equal(drift[: last_anchored + 1], np.zeros(last_anchored + 1))
    # The tail is a random walk, so it must not be identically zero as well.
    assert np.any(drift[last_anchored + 1 :] != 0.0)

    s_year = np.asarray(model["recording_s_year"].eval())
    s_anchored = float(model["recording_s"].eval())
    assert np.allclose(s_year[: last_anchored + 1], s_anchored)
    assert float(model["recording_s_drift_ratio"].eval()) == pytest.approx(
        s_year[-1] / s_anchored
    )


def test_zero_drift_sigma_reproduces_the_undrifted_model(tmp_path: Path) -> None:
    """The all-prevalence corner must be DSP008 exactly, not merely close."""
    pytest.importorskip("pymc")

    undrifted, _ = _drift_model(tmp_path, recording_drift="none")
    zero_drift, _ = _drift_model(tmp_path, drift_sigma=0.0)

    assert {rv.name for rv in undrifted.free_RVs} == {
        rv.name for rv in zero_drift.free_RVs
    }
    assert set(undrifted.named_vars) == set(zero_drift.named_vars)
    assert "recording_s_drift_logit" not in zero_drift.named_vars
    point = undrifted.initial_point()
    assert float(zero_drift.compile_logp()(point)) == float(
        undrifted.compile_logp()(point)
    )


def test_drift_moves_both_certificate_versions_together(tmp_path: Path) -> None:
    """The drift is a recording-behaviour trend, not a change in the version gap."""
    pm = pytest.importorskip("pymc")

    model, _ = _drift_model(tmp_path)
    with model:
        prior = pm.sample_prior_predictive(draws=6, random_seed=3)

    # Cells are ordered (year, age, revised), so within a year each unrevised
    # cell is immediately followed by its revised counterpart.
    p_recorded = np.asarray(prior.prior["p_recorded"].values)
    assert p_recorded.shape[-1] == 4 * 13
    drawn = np.asarray(prior.prior["recording_s_drift_logit"].values)
    assert drawn.shape[-1] == 13
    assert np.allclose(drawn[..., :7], 0.0)
    for name in ("recording_s", "recording_s_unrevised", "recording_s_year"):
        values = np.asarray(prior.prior[name].values)
        assert ((0.0 < values) & (values < 1.0)).all()


def test_anchor_forecast_flat_holds_prevalence_past_the_last_window(
    tmp_path: Path,
) -> None:
    """The all-recording corner pins prevalence instead of forecasting it."""
    pytest.importorskip("pymc")

    model, anchor = _drift_model(tmp_path, anchor_forecast_flat=True)
    last_anchored = int(anchor.mid_year_idx.max()) + anchor.half_width

    latent = np.asarray(model["anchor_log_prevalence_latent"].eval())
    last_anchored_latent = int(anchor.mid_year_idx.max()) + 2 * anchor.half_width
    assert np.allclose(latent[last_anchored_latent:], latent[last_anchored_latent])
    # The anchored years themselves must still move.
    assert not np.allclose(latent[:last_anchored_latent], latent[0])

    prevalence = np.asarray(model["prevalence_year"].eval())
    assert np.allclose(prevalence[last_anchored:], prevalence[last_anchored])

    # Without the flag the same graph forecasts rather than pinning.
    forecasting, _ = _drift_model(tmp_path)
    forecast_prevalence = np.asarray(forecasting["prevalence_year"].eval())
    assert not np.allclose(
        forecast_prevalence[last_anchored:], forecast_prevalence[last_anchored]
    )


def test_post_anchor_drift_requires_an_unanchored_year(tmp_path: Path) -> None:
    """A window reaching the final modelled year leaves nothing to drift over."""
    pytest.importorskip("pymc")

    # Mid-2008 with half-width 2 covers 2010, the last year of a 2004-2010 range.
    with pytest.raises(ValueError, match="at least one modelled year beyond"):
        _drift_model(tmp_path, n_year=7, mid_years=[2006, 2008])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"recording_drift": "annual"}, "recording_drift must be"),
        ({"recording_model": "year"}, "cannot be combined"),
    ],
)
def test_build_rejects_incoherent_drift_configurations(
    tmp_path: Path,
    kwargs: dict[str, str],
    message: str,
) -> None:
    pytest.importorskip("pymc")
    with pytest.raises(ValueError, match=message):
        _drift_model(tmp_path, **kwargs)


def test_unanchored_model_rejects_drift_and_flat_forecast() -> None:
    """Both drift controls are meaningless without a surveillance anchor."""
    pytest.importorskip("pymc")

    priors = CoreReductionPriors(
        reduction_mean=np.full(9, 0.35),
        reduction_logit=logit(np.full(9, 0.35)),
        reduction_sigma=np.full(9, 0.2),
    )
    with pytest.raises(ValueError, match="requires reduction_model='anchor'"):
        build_core_reduction_model(
            _drift_cells(9),
            priors,
            n_year=9,
            recording_model="revision",
            reduction_model="year",
            recording_drift="post_anchor",
        )
    with pytest.raises(ValueError, match="anchor_forecast_flat requires"):
        build_core_reduction_model(
            _drift_cells(9),
            priors,
            n_year=9,
            recording_model="revision",
            reduction_model="year",
            anchor_forecast_flat=True,
        )


def test_drift_config_records_whether_the_prior_is_live(tmp_path: Path) -> None:
    """A reader must not have to infer a live drift from the sigma alone."""
    path = tmp_path / "reduction.csv"
    years = list(range(2004, 2017))
    pd.DataFrame({"year": years, "reduction": [0.35] * len(years)}).to_csv(
        path, index=False
    )
    anchor = SurveillanceAnchor.from_csv(
        year_range=(2004, 2016),
        path=_anchor_csv(tmp_path, [2006, 2008], [12.5, 13.0]),
    )

    for drift_sigma, expected in ((0.06, True), (0.0, False)):
        priors = CoreReductionPriors.from_reduction_csv(
            year_range=(2004, 2016),
            path=path,
            recording_s_drift_sigma=drift_sigma,
        )
        config = CoreReductionModelConfig.from_priors(
            year_range=(2004, 2016),
            priors_obj=priors,
            model_definition=DSP009,
            anchor=anchor,
        ).to_dict()
        assert config["model_id"] == "DSP009"
        assert config["recording_drift"] == "post_anchor"
        assert config["priors"]["recording_s_drift_sigma"] == drift_sigma
        assert config["priors"]["recording_drift_enters_likelihood"] is expected
        # An anchored fit never consumes the reduction prior, drifted or not.
        assert config["priors"]["reduction_prior_enters_likelihood"] is False

    undrifted = CoreReductionModelConfig.from_priors(
        year_range=(2004, 2016),
        priors_obj=CoreReductionPriors.from_reduction_csv(
            year_range=(2004, 2016), path=path
        ),
        model_definition=DSP008,
        anchor=anchor,
    ).to_dict()
    assert undrifted["recording_drift"] == "none"
    assert undrifted["priors"]["recording_drift_enters_likelihood"] is False


def test_core_reduction_priors_reject_negative_drift_sigma(tmp_path: Path) -> None:
    path = tmp_path / "reduction.csv"
    pd.DataFrame({"year": [2020], "reduction": [0.35]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="recording_s_drift_sigma"):
        CoreReductionPriors.from_reduction_csv(
            year_range=(2020, 2020), path=path, recording_s_drift_sigma=-0.01
        )


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
        reduction_error_correlation=0.6,
        reduction_calibration_shift_logit=0.2,
        extrapolated_start=2021,
    )

    assert np.allclose(priors.reduction_mean, [0.35, 0.40])
    assert np.allclose(priors.reduction_sigma, [0.2, 0.5])
    assert priors.reduction_error_correlation == 0.6
    assert priors.reduction_calibration_shift_logit == 0.2


@pytest.mark.parametrize(
    ("correlation", "shift", "message"),
    [
        (-0.1, 0.0, "reduction_error_correlation"),
        (1.0, 0.0, "reduction_error_correlation"),
        (0.0, np.inf, "reduction_calibration_shift_logit"),
    ],
)
def test_core_reduction_priors_reject_invalid_calibration_parameters(
    tmp_path: Path,
    correlation: float,
    shift: float,
    message: str,
) -> None:
    path = tmp_path / "reduction.csv"
    pd.DataFrame({"year": [2020], "reduction": [0.35]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match=message):
        CoreReductionPriors.from_reduction_csv(
            year_range=(2020, 2020),
            path=path,
            reduction_error_correlation=correlation,
            reduction_calibration_shift_logit=shift,
        )


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


def test_core_reduction_config_serialises_dsp003_sensitivities(tmp_path: Path) -> None:
    path = tmp_path / "reduction.csv"
    pd.DataFrame({"year": [2020], "reduction": [0.35]}).to_csv(path, index=False)
    priors = CoreReductionPriors.from_reduction_csv(
        year_range=(2020, 2020),
        path=path,
        reduction_age_step_sigma=0.08,
        reduction_error_correlation=0.5,
        reduction_calibration_shift_logit=-0.2,
        false_positive_rate=0.0,
    )

    config = CoreReductionModelConfig.from_priors(
        year_range=(2020, 2020),
        priors_obj=priors,
        model_definition=DSP003,
        recorded_definition="confirmed_only",
    ).to_dict()

    assert config["model_id"] == "DSP003"
    assert config["recording_model"] == "constant"
    assert config["reduction_model"] == "year_age"
    assert config["age_model"] == "single_year"
    assert config["recorded_definition"] == "confirmed_only"
    assert config["theta_model"] == "morris_double_logistic_by_age_code"
    assert config["age_endpoint_convention"] == {
        "12": "10-12; Morris evaluated at age 12",
        "50": "50+; Morris evaluated at age 50",
    }
    assert config["priors"]["theta_lb_age_used"] is False
    assert config["priors"]["reduction_age_step_sigma"] == 0.08
    assert config["priors"]["reduction_error_correlation"] == 0.5
    assert config["priors"]["reduction_calibration_shift_logit"] == -0.2
    assert config["priors"]["false_positive_rate"] == 0.0


@pytest.mark.parametrize("model_definition", [DSP004, DSP005])
def test_exact_age_ablation_config_records_theta_provenance(
    model_definition: CoreModelDefinition,
) -> None:
    priors = CoreReductionPriors(
        reduction_mean=np.array([0.35]),
        reduction_logit=logit(np.array([0.35])),
        reduction_sigma=np.array([0.20]),
    )

    config = CoreReductionModelConfig.from_priors(
        year_range=(2020, 2020),
        priors_obj=priors,
        model_definition=model_definition,
    ).to_dict()

    assert config["age_model"] == "single_year"
    assert config["reduction_model"] == "year"
    assert config["theta_model"] == "morris_double_logistic_by_age_code"
    assert config["age_endpoint_convention"]["12"].startswith("10-12")
    assert config["age_endpoint_convention"]["50"].startswith("50+")
    assert config["priors"]["theta_lb_age_used"] is False


def test_core_reduction_priors_reject_invalid_false_positive_rate(
    tmp_path: Path,
) -> None:
    path = tmp_path / "reduction.csv"
    pd.DataFrame({"year": [2020], "reduction": [0.35]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="false_positive_rate"):
        CoreReductionPriors.from_reduction_csv(
            year_range=(2020, 2020), path=path, false_positive_rate=1.0
        )


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


def test_reduction_error_covariance_preserves_marginals_and_correlation() -> None:
    sigma = np.array([0.20, 0.45, 0.30])
    correlation = 0.6

    covariance = _reduction_error_covariance(sigma, correlation)
    cholesky = np.linalg.cholesky(covariance)
    reconstructed = cholesky @ cholesky.T
    standardised = covariance / np.outer(sigma, sigma)

    assert reconstructed == pytest.approx(covariance)
    assert np.diag(covariance) == pytest.approx(sigma**2)
    assert standardised[np.triu_indices(len(sigma), k=1)] == pytest.approx(correlation)


def test_correlated_reduction_prior_applies_whitened_shifted_transform() -> None:
    pm = pytest.importorskip("pymc")

    cells = pd.DataFrame(
        {
            "year_idx": [0, 1],
            "age_idx": [2, 2],
            "N_cell": [1000, 900],
            "R_cell": [1, 1],
        }
    )
    cells.attrs["n_year"] = 2
    anchor_logit = logit(np.array([0.35, 0.40]))
    sigma = np.array([0.20, 0.45])
    priors = CoreReductionPriors(
        reduction_mean=np.array([0.35, 0.40]),
        reduction_logit=anchor_logit,
        reduction_sigma=sigma,
        reduction_error_correlation=0.6,
        reduction_calibration_shift_logit=0.2,
    )

    model = build_core_reduction_model(cells, priors, n_year=2)
    assert {rv.name for rv in model.free_RVs} == {
        "rho_logit_year_raw",
        "recording_s_logit",
    }
    with model:
        prior = pm.sample_prior_predictive(
            draws=20,
            var_names=["rho_logit_year", "rho_logit_year_raw"],
            random_seed=11,
        )

    rho_logit = np.asarray(prior.prior["rho_logit_year"]).reshape(-1, 2)
    raw = np.asarray(prior.prior["rho_logit_year_raw"]).reshape(-1, 2)
    covariance = _reduction_error_covariance(sigma, 0.6)
    expected = anchor_logit + 0.2 + raw @ np.linalg.cholesky(covariance).T
    assert rho_logit == pytest.approx(expected)


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


@pytest.mark.parametrize(
    ("recording_model", "expected_free_rvs"),
    [
        ("constant", {"rho_logit_year", "recording_s_logit"}),
        (
            "year",
            {
                "rho_logit_year",
                "recording_s_logit",
                "recording_s_year_offset_raw",
            },
        ),
    ],
)
def test_build_exact_age_ablation_uses_morris_without_age_reduction(
    recording_model: str,
    expected_free_rvs: set[str],
) -> None:
    cells = pd.DataFrame(
        {
            "year_idx": [0, 0, 0, 1, 1, 1],
            "age_idx": [0, 1, 2, 0, 1, 2],
            "maternal_age": [20, 30, 40, 20, 30, 40],
            "N_cell": [1000, 800, 200, 900, 700, 300],
            "R_cell": [1, 2, 3, 1, 2, 4],
        }
    )
    cells.attrs.update(
        {
            "n_year": 2,
            "n_age": 3,
            "age_model": "single_year",
            "age_values": [20, 30, 40],
            "year_range": (2020, 2021),
        }
    )
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
        recording_model=recording_model,
        reduction_model="year",
    )

    assert {rv.name for rv in model.free_RVs} == expected_free_rvs
    assert np.array_equal(model.coords["age"], (20, 30, 40))
    assert np.allclose(
        model["theta_lb_age"].get_value(),
        get_ds_lb_nt_probability_array(np.array([20, 30, 40])),
    )
    assert "rho_year_age" not in model.named_vars
    assert "rho_age_offset" not in model.named_vars
    assert model["p_ds_lb"].eval().shape == (6,)


def test_build_dsp003_preserves_weighted_reduction_margin() -> None:
    pm = pytest.importorskip("pymc")

    cells = pd.DataFrame(
        {
            "year_idx": [0, 0, 0, 1, 1, 1],
            "age_idx": [0, 1, 2, 0, 1, 2],
            "maternal_age": [20, 30, 40, 20, 30, 40],
            "N_cell": [1000, 800, 200, 900, 700, 300],
            "R_cell": [1, 2, 3, 1, 2, 4],
        }
    )
    cells.attrs.update(
        {
            "n_year": 2,
            "n_age": 3,
            "age_model": "single_year",
            "age_values": [20, 30, 40],
            "year_range": (2020, 2021),
        }
    )
    priors = CoreReductionPriors(
        reduction_mean=np.array([0.35, 0.40]),
        reduction_logit=logit(np.array([0.35, 0.40])),
        reduction_sigma=np.array([0.25, 0.35]),
        reduction_age_step_sigma=0.15,
    )

    model = build_core_reduction_model(
        cells,
        priors,
        n_year=2,
        recording_model="constant",
        reduction_model="year_age",
    )
    named = {rv.name for rv in model.free_RVs}
    assert named == {"rho_logit_year", "rho_age_step", "recording_s_logit"}
    assert model["rho_year_age"].eval().shape == (2, 3)
    assert np.allclose(
        model["theta_lb_age"].get_value(),
        get_ds_lb_nt_probability_array(np.array([20, 30, 40])),
    )

    with model:
        prior = pm.sample_prior_predictive(draws=20, random_seed=0)

    rho_year_age = np.asarray(prior.prior["rho_year_age"])
    rho_year = np.asarray(prior.prior["rho_year"])
    rho_anchor = np.asarray(prior.prior["rho_year_anchor"])
    theta = np.asarray(get_ds_lb_nt_probability_array(np.array([20, 30, 40])))
    year_age_n = np.array([[1000, 800, 200], [900, 700, 300]], dtype=float)
    natural = year_age_n * theta[None, :]
    weights = natural / natural.sum(axis=1, keepdims=True)
    margin = (rho_year_age * weights[None, None, :, :]).sum(axis=-1)

    assert np.allclose(rho_year, margin, atol=1e-10)
    assert np.allclose(rho_year, rho_anchor, atol=1e-10)
    assert np.allclose(
        np.asarray(prior.prior["rho_age_offset"]).mean(axis=-1), 0.0, atol=1e-12
    )
    gradient = np.asarray(model.compile_dlogp()(model.initial_point()))
    assert np.isfinite(gradient).all()


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
