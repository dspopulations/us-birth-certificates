"""Compile + prior-predictive smoke tests for the staged model builds.

Each of the four specs (``theta_only``, ``theta_s``, ``single_eta``,
``full``) must build, draw from its prior predictive, and produce
valid ``R_obs`` draws (non-negative, <= N_cell) on a tiny synthetic
cell frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pymc")

from dspopulations_us_birth_certificates.selection import (  # noqa: E402
    TrueParams,
    build_model,
    simulate_cells,
    variant_C_default,
)

N_YEAR = 9


@pytest.fixture(scope="module")
def tiny_cells() -> pd.DataFrame:
    truth = TrueParams.from_priors(
        variant_C_default(), n_year=N_YEAR, seed=0
    )
    cells = simulate_cells(
        truth,
        n_cells_per_month=2,
        n_year=N_YEAR,
        n_cells_mean=500,
        seed=0,
    )
    return cells


@pytest.mark.parametrize(
    "spec", ["theta_only", "theta_s", "single_eta", "full"]
)
def test_build_model_and_prior_predict(
    spec: str, tiny_cells: pd.DataFrame
) -> None:
    import pymc as pm

    model = build_model(
        tiny_cells, variant_C_default(), spec=spec, n_year=N_YEAR
    )
    with model:
        prior = pm.sample_prior_predictive(draws=5, random_seed=0)

    assert "R_obs" in prior.prior_predictive

    r = np.asarray(prior.prior_predictive["R_obs"].values)
    N = tiny_cells["N_cell"].to_numpy()
    assert (r >= 0).all(), f"{spec}: negative R_obs sampled"
    assert (r <= N[None, None, :]).all(), f"{spec}: R_obs > N_cell"
    assert r.shape[-1] == len(tiny_cells)


def test_build_model_rejects_unknown_spec(tiny_cells: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Unknown spec"):
        build_model(
            tiny_cells,
            variant_C_default(),
            spec="bogus",  # type: ignore[arg-type]
            n_year=N_YEAR,
        )


def test_full_spec_has_detect_and_term(tiny_cells: pd.DataFrame) -> None:
    """Only the ``full`` spec should emit the decomposed eta_detect/eta_term."""
    model = build_model(
        tiny_cells, variant_C_default(), spec="full", n_year=N_YEAR
    )
    named = {rv.name for rv in model.free_RVs}
    assert "eta_detect_int" in named
    assert "eta_term_int" in named
    assert "eta_term_year" in named
    assert "eta_term_ry" not in named


def test_full_margin_anchor_adds_prevalence_observation(
    tiny_cells: pd.DataFrame,
) -> None:
    """``prev_margin`` ties the marginal p_ds_lb per race×year to a de Graaf target."""
    import pymc as pm

    n_race = 6
    # One anchored target per race×year for the named races; Unknown (idx 5) is NaN.
    target = np.full((n_race, N_YEAR), 12.0)
    target[5, :] = np.nan
    sigma = np.full((n_race, N_YEAR), 1.0)

    model = build_model(
        tiny_cells,
        variant_C_default(),
        spec="full",
        n_year=N_YEAR,
        prev_margin=(target, sigma),
    )
    obs = {rv.name for rv in model.observed_RVs}
    det = {d.name for d in model.deterministics}
    assert "prev_margin_obs" in obs and "prev_margin" in det
    # Only the 5 named races are anchored -> 5 × N_YEAR groups (Unknown excluded).
    assert model["prev_margin"].eval().shape == (5 * N_YEAR,)

    with model:
        prior = pm.sample_prior_predictive(draws=5, random_seed=0)
    assert "prev_margin_obs" in prior.prior_predictive
    # Without prev_margin the observation must be absent (default path unchanged).
    plain = build_model(tiny_cells, variant_C_default(), spec="full", n_year=N_YEAR)
    assert "prev_margin_obs" not in {rv.name for rv in plain.observed_RVs}


def test_theta_only_omits_eta_and_s(tiny_cells: pd.DataFrame) -> None:
    model = build_model(
        tiny_cells, variant_C_default(), spec="theta_only", n_year=N_YEAR
    )
    named = {rv.name for rv in model.free_RVs}
    assert "theta_lb_age" in named
    assert "eta_detect_int" not in named
    assert "eta_term_int" not in named
    assert "eta_int" not in named
    assert "s_int" not in named
