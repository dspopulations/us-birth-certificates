# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Checks of likelihood equivalence, uncertainty and age standardisation."""

import numpy as np
import pytest
import statsmodels.api as sm
from scipy.linalg import null_space
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests

from dspopulations_us_birth_certificates.recorded_trends import (
    bootstrap_test,
    fit_trends,
    fitted_proportions,
    gamma_interval,
    holm,
    profile_contrast,
    simulate_conditional,
    trend_designs,
)


@pytest.fixture
def cells():
    rng = np.random.default_rng(7182)
    times = np.arange(8) - 3.5
    ages = np.array([19, 20, 25, 30, 35, 40, 45])
    low, high, _ = trend_designs(range(3), ages)
    births = rng.integers(9000, 20000, size=(3, 7, 8))
    beta = np.array([0.08, 0.05, 0.03, 0.0, -0.02, -0.05, -0.07, 0.03, -0.01])
    mean = births * np.exp(-7 + (high @ beta)[..., None] * times)
    counts = rng.poisson(mean)
    counts[2, 0] = 0  # A zero-total age stratum is at the intercept boundary.
    return counts, births, times, low, high


def test_conditional_fit_matches_full_poisson_glm(cells):
    y, n, t, _, z = cells
    fit = fit_trends(y, n, t, z)
    active = y.sum(-1) > 0
    intercept = np.repeat(np.eye(active.sum()), len(t), axis=0)
    trend = (z[..., None, :] * t[None, None, :, None])[active].reshape(-1, z.shape[-1])
    glm = sm.GLM(
        y[active].ravel(),
        np.column_stack([intercept, trend]),
        family=sm.families.Poisson(),
        offset=np.log(n[active].ravel()),
    ).fit(tol=1e-11)
    np.testing.assert_allclose(fit["beta"][0], glm.params[-z.shape[-1] :], atol=1e-8)
    np.testing.assert_allclose(
        np.linalg.inv(fit["information"][0]),
        glm.cov_params()[-z.shape[-1] :, -z.shape[-1] :],
        atol=1e-7,
    )
    expected = y.sum(-1)[..., None] * fit["p"][0]
    np.testing.assert_allclose(expected[active].ravel(), glm.fittedvalues, atol=1e-7)
    np.testing.assert_allclose(expected.sum(-1), y.sum(-1), atol=1e-10)
    assert np.all(expected[2, 0] == 0)


def test_profile_endpoints_reach_likelihood_cutoff(cells):
    y, n, t, _, z = cells
    fit = fit_trends(y, n, t, z)
    # A sum of age and race slopes, not just one parameter.
    contrast = np.zeros(z.shape[-1])
    contrast[3] = contrast[7] = 1
    lower, upper = profile_contrast(y, n, t, z, fit, contrast)
    assert lower < np.exp(contrast @ fit["beta"][0]) < upper
    basis = null_space(contrast[None, :])
    for value in (lower, upper):
        shift = (z @ (contrast / (contrast @ contrast)))[..., None] * np.log(value) * t
        constrained = fit_trends(y, n * np.exp(shift), t, z @ basis)
        np.testing.assert_allclose(
            2 * (fit["ll"] - constrained["ll"]), chi2.ppf(0.95, 1), atol=1e-7
        )


def test_standardisation_preserves_age_dependent_trends(cells):
    y, n, t, _, z = cells
    fit = fit_trends(y, n, t, z)
    weights = np.arange(1, 8) / 28
    observed_model_rates = y.sum(-1)[..., None] * fit["p"][0] / n
    expected = (observed_model_rates * weights[None, :, None]).sum(1) * 10000
    rates = fitted_proportions(y, n, t, z, fit, weights)
    np.testing.assert_allclose(rates, expected, atol=1e-12)
    # A weighted sum of age curves generally has changing annual multipliers.
    assert np.ptp(rates[0, 1:] / rates[0, :-1]) > 1e-5
    # But the relative annual multipliers between groups at a given age agree.
    annual = np.exp(z @ fit["beta"][0])
    np.testing.assert_allclose(annual[1] / annual[0], np.exp(fit["beta"][0, 7]))


def test_conditional_simulations_preserve_counts_and_bootstrap_seed(cells):
    y, n, t, low, high = cells
    null = fit_trends(y, n, t, low)
    draws = simulate_conditional(np.random.default_rng(20), y.sum(-1), null["p"][0], 31)
    np.testing.assert_array_equal(
        draws.sum(-1), np.broadcast_to(y.sum(-1), (31, *y.shape[:-1]))
    )
    assert np.all(draws[:, 2, 0] == 0)
    a = bootstrap_test(y, n, t, low, high, 31, 10)
    b = bootstrap_test(y, n, t, low, high, 31, 10)
    assert a == b
    assert a["bootstrap_p"] == (a["exceedances"] + 1) / 32


@pytest.mark.parametrize("count", [0, 1, 20])
def test_gamma_limits_reduce_to_poisson_when_coefficients_equal(count):
    rate, lower, upper = gamma_interval(
        np.array([count, 0]), np.array([100, 100]), np.array([0.5, 0.5])
    )
    assert rate == count * 50
    expected_low = 0 if count == 0 else 25 * chi2.ppf(0.025, 2 * count)
    np.testing.assert_allclose(
        [lower, upper], [expected_low, 25 * chi2.ppf(0.975, 2 * (count + 1))]
    )


def test_gamma_upper_limit_includes_zero_case_strata():
    _, _, upper = gamma_interval(
        np.array([10, 0]), np.array([10000, 10]), np.array([0.5, 0.5])
    )
    _, _, without_small_cell = gamma_interval(
        np.array([10, 0]), np.array([10000, 10000]), np.array([0.5, 0.5])
    )
    assert upper > 100 * without_small_cell


def test_holm_matches_library_with_ties():
    p = np.array([0.015, 0.001, 0.9, 0.001, 0.04, 0.3])
    np.testing.assert_allclose(holm(p), multipletests(p, method="holm")[1])


def test_age_restriction_drops_absent_band_parameters():
    low, high, _ = trend_designs(range(7), np.arange(20, 45))
    assert low.shape == (7, 25, 5)
    assert high.shape == (7, 25, 11)


def test_nonpositive_denominator_rejected(cells):
    y, n, t, _, z = cells
    n[0, 0, 0] = 0
    with pytest.raises(ValueError, match="positive birth"):
        fit_trends(y, n, t, z)
