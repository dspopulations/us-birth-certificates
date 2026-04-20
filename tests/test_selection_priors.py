"""Lock-down tests for the selection-model priors.

These tests guard against silent drift in the published-literature values
that anchor the three-stage model. If any numeric is changed here, the
corresponding section of ``plans/docs/bayesian_selection_model.md`` (and
``plans/20260420-selection-model.md`` §10) should be updated in the same
commit.
"""

from __future__ import annotations

import numpy as np
import pytest

from dspopulations_us_birth_certificates.selection import priors as P


def test_morris_age_rates_monotone() -> None:
    """Morris rates should increase with maternal age."""
    rates = P.MORRIS_THETA_LB_PER_1000
    diffs = np.diff(rates)
    assert (diffs >= 0).all(), f"Morris rates not monotone: {rates!r}"


def test_morris_rates_match_de_graaf_2015() -> None:
    """Values must match the de Graaf 2015 corrigendum (EJHG)."""
    expected = np.array([0.66, 0.70, 0.84, 1.48, 4.72, 15.22, 30.71])
    assert np.allclose(P.MORRIS_THETA_LB_PER_1000, expected)


def test_factor_level_lengths_match_arrays() -> None:
    """Prior arrays must match their factor-level vocabularies in length."""
    assert len(P.RACE_LEVELS) == P.N_RACE == len(P.ETA_DETECT_RACE)
    assert len(P.RACE_LEVELS) == len(P.ETA_TERM_RACE) == len(P.S_RACE)
    assert len(P.EDU_LEVELS) == P.N_EDU == len(P.ETA_DETECT_EDU)
    assert len(P.EDU_LEVELS) == len(P.ETA_TERM_EDU) == len(P.S_EDU)
    assert len(P.PAYER_LEVELS) == P.N_PAYER == len(P.ETA_DETECT_PAYER)
    assert len(P.AGE_LEVELS) == P.N_AGE == len(P.MORRIS_THETA_LB_PER_1000)


def test_reference_levels_are_zero() -> None:
    """Reference levels must have a 0.0 prior mean to keep intercept
    interpretation stable (plan §10 #4)."""
    assert P.ETA_DETECT_RACE[0] == 0.0  # NH White
    assert P.ETA_TERM_RACE[0] == 0.0
    assert P.S_RACE[0] == 0.0
    assert P.ETA_DETECT_EDU[2] == 0.0  # Some college
    assert P.ETA_TERM_EDU[2] == 0.0
    assert P.S_EDU[2] == 0.0
    assert P.ETA_DETECT_PAYER[1] == 0.0  # Private


def test_sensitivity_variants_differ_as_expected() -> None:
    """Each variant moves the named priors in a specific direction."""
    c = P.variant_C_default()
    a = P.variant_A_tight_s()
    b = P.variant_B_tight_eta_term()
    d = P.variant_D_dobbs_only()

    # Variant A: tighter s, looser eta_term.
    assert a.s_race_sigma < c.s_race_sigma
    assert a.eta_term_race_sigma > c.eta_term_race_sigma

    # Variant B: tighter eta_term, looser s.
    assert b.eta_term_race_sigma < c.eta_term_race_sigma
    assert b.s_race_sigma > c.s_race_sigma

    # Variant D: informative race priors on eta_term shrunk to zero with
    # wide sigma; post-Dobbs year sigma widened.
    assert np.allclose(d.eta_term_race, 0.0)
    assert d.eta_term_race_sigma > c.eta_term_race_sigma
    assert (
        d.eta_term_year_sigma_post_dobbs > c.eta_term_year_sigma_post_dobbs
    )


def test_logit_round_trip() -> None:
    """``logit`` and ``inv_logit`` are inverse."""
    for p in (0.01, 0.1, 0.5, 0.9, 0.99):
        assert np.isclose(P.inv_logit(P.logit(p)), p)


def test_morris_sigma_is_tight() -> None:
    """Morris sigma must remain tight (plan §10 #1: loosening lets the data
    drag theta_LB around, absorbing variation that belongs to eta/s)."""
    assert P.MORRIS_SIGMA <= 0.15


def test_false_positive_rate_fixed() -> None:
    """The Ohio/NY false-positive value is fixed, not estimated (plan §10 #3)."""
    assert P.FALSE_POSITIVE_RATE == pytest.approx(7.8e-5)
    for factory in P.VARIANTS.values():
        assert factory().false_positive_rate == pytest.approx(7.8e-5)


def test_variants_registry_contains_abcd() -> None:
    assert set(P.VARIANTS) == {"A", "B", "C", "D"}


def test_post_dobbs_sigma_wider_than_pre() -> None:
    """The pre/post-Dobbs asymmetry is load-bearing for identification."""
    c = P.variant_C_default()
    assert (
        c.eta_term_year_sigma_post_dobbs > c.eta_term_year_sigma_pre_dobbs
    )
