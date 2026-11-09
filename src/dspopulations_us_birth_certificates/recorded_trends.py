# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Age-dependent trends in recorded Down syndrome counts.

AI-assisted revision by Codex (GPT-6).

Arrays have axes race, age stratum, year. A leading axis batches simulations.
Conditioning on each race-age case total removes its free Poisson intercept.
The fitted slopes and their profile likelihood equal those of a Poisson GLM
with log(births) offsets and separate race-age intercepts. These are models of
recorded status, not true prevalence or recording sensitivity.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import brentq
from scipy.special import logsumexp
from scipy.stats import chi2, gamma


def gamma_interval(counts, births, weights, alpha=0.05):
    """Fay-Feuer interval, including the maximum weight over zero-count cells."""
    coefficients = np.asarray(weights) / np.asarray(births) * 10000
    rate = float(coefficients @ counts)
    variance = float(coefficients**2 @ counts)
    maximum = float(coefficients.max())
    lower = (
        0.0
        if rate == 0
        else gamma.ppf(alpha / 2, rate**2 / variance, scale=variance / rate)
    )
    upper_variance = variance + maximum**2
    upper_mean = rate + maximum
    upper = gamma.ppf(
        1 - alpha / 2, upper_mean**2 / upper_variance, scale=upper_variance / upper_mean
    )
    return rate, float(lower), float(upper)


def holm(p):
    """Independent implementation of Holm's step-down adjustment."""
    p = np.asarray(p)
    order = np.argsort(p)
    adjusted = np.minimum(
        1, np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p))))
    )
    result = np.empty_like(p)
    result[order] = adjusted
    return result


def fit_trends(y, n, t, z):
    """Profile Poisson MLE using race-age-specific slopes specified by z.

    alpha[r,a] is eliminated analytically. All derivatives use the first and
    second moments of year, independently of the earlier derivative code.
    A leading dimension permits batching simulations.
    """
    y = np.asarray(y, dtype=float)
    n, t, z = (np.asarray(value, dtype=float) for value in (n, t, z))
    if y.ndim == 3:
        y = y[None, ...]
    if (
        y.ndim != 4
        or n.shape != y.shape[1:]
        or t.shape != (y.shape[-1],)
        or z.shape[:2] != y.shape[1:3]
        or not all(np.isfinite(value).all() for value in (y, n, t, z))
        or (n <= 0).any()
        or (y < 0).any()
    ):
        raise ValueError("Require finite non-negative counts and positive birth cells")
    case_totals = y.sum(-1)
    case_times = (y * t).sum(-1)
    beta = np.zeros((len(y), z.shape[-1]))
    for _ in range(100):
        slopes = np.einsum("rap,bp->bra", z, beta)
        logits = np.log(n) + slopes[..., None] * t
        logp = logits - logsumexp(logits, axis=-1, keepdims=True)
        p = np.exp(logp)
        mean = (p * t).sum(-1)
        variance = (p * (t - mean[..., None]) ** 2).sum(-1)
        scores = case_times - case_totals * mean
        score = np.einsum("bra,rap->bp", scores, z)
        information = np.einsum("bra,rap,raq->bpq", case_totals * variance, z, z)
        step = np.linalg.solve(information, score[..., None])[..., 0]
        if np.max(np.abs(step)) < 1e-10:
            break
        step *= np.minimum(1, 0.5 / np.maximum(np.max(np.abs(step), axis=-1), 1e-30))[
            :, None
        ]
        beta += step
    else:
        raise RuntimeError("Trend fit did not converge")
    return {
        "beta": beta,
        "information": information,
        "p": p,
        "logp": logp,
        "ll": (y * logp).sum((-3, -2, -1)),
        "score_max": float(np.abs(score).max()),
    }


def simulate_conditional(rng, totals, p, batch):
    draws = np.zeros((batch, *p.shape), dtype=np.int64)
    # Draw the first seven years sequentially from binomials. This independently
    # checks the earlier use of the library multinomial sampler.
    remaining = np.broadcast_to(totals, (batch, *totals.shape)).copy().astype(np.int64)
    survival = np.ones(p.shape[:-1])
    for j in range(p.shape[-1] - 1):
        probability = np.clip(p[..., j] / survival, 0, 1)
        draws[..., j] = rng.binomial(remaining, probability)
        remaining -= draws[..., j]
        survival -= p[..., j]
    draws[..., -1] = remaining
    assert np.array_equal(draws.sum(-1), np.broadcast_to(totals, draws.shape[:-1]))
    return draws


def bootstrap_test(y, n, t, low, high, repetitions, seed):
    """Conditional null simulation, with plus-one p-value and Monte Carlo SE."""
    if repetitions < 1:
        raise ValueError("At least one bootstrap repetition is required")
    null = fit_trends(y, n, t, low)
    alt = fit_trends(y, n, t, high)
    observed = float(2 * (alt["ll"] - null["ll"])[0])
    exceed = 0
    rng = np.random.default_rng(seed)
    for start in range(0, repetitions, 128):
        drawn = simulate_conditional(
            rng, y.sum(-1), null["p"][0], min(128, repetitions - start)
        )
        a = fit_trends(drawn, n, t, low)
        b = fit_trends(drawn, n, t, high)
        statistic = 2 * (b["ll"] - a["ll"])
        assert statistic.min() > -1e-7
        exceed += int(np.sum(statistic >= observed))
    p = (exceed + 1) / (repetitions + 1)
    return {
        "lrt": observed,
        "df": high.shape[-1] - low.shape[-1],
        "repetitions": repetitions,
        "exceedances": exceed,
        "seed": seed,
        "bootstrap_p": p,
        "mc_se": float(np.sqrt(p * (1 - p) / repetitions)),
    }


def constrained_contrast(y, n, t, z, fit, contrast):
    basis = null_space(contrast[None, :])
    null_z = z @ basis
    null = fit_trends(y, n, t, null_z)
    estimate = float(contrast @ fit["beta"][0])
    statistic = max(0, float(2 * (fit["ll"] - null["ll"])[0]))
    return {
        "log_annual_factor_ratio": estimate,
        "annual_factor_ratio": float(np.exp(estimate)),
        "lrt": statistic,
        "p_value": float(chi2.sf(statistic, 1)),
    }, null_z


def profile_contrast(y, n, t, z, fit, contrast):
    """Pointwise 95% likelihood interval for exp(contrast @ beta).

    The cutoff is asymptotic chi-square(1). These intervals are not adjusted
    for multiple comparisons, even when accompanying p-values are adjusted.
    """
    basis = null_space(contrast[None, :])
    null_z = z @ basis
    direction = contrast / (contrast @ contrast)
    multiplier = z @ direction
    estimate = float(contrast @ fit["beta"][0])
    best = fit["ll"][0]

    def objective(value):
        adjusted = n * np.exp(multiplier[..., None] * value * t)
        restricted = fit_trends(y, adjusted, t, null_z)
        return 2 * (best - restricted["ll"][0]) - chi2.ppf(0.95, 1)

    step = 0.1
    while objective(estimate - step) < 0:
        step *= 2
    lower = brentq(objective, estimate - step, estimate)
    step = 0.1
    while objective(estimate + step) < 0:
        step *= 2
    upper = brentq(objective, estimate, estimate + step)
    return float(np.exp(lower)), float(np.exp(upper))


def trend_designs(races, ages):
    """Age-only null and age-plus-race alternative with first race as reference.

    The seven bands are under 20, 20-24, 25-29, 30-34, 35-39, 40-44, 45+.
    Only bands present in ages enter the design, permitting age restrictions.
    """
    bands = np.searchsorted([20, 25, 30, 35, 40, 45], ages, side="right")
    present, codes = np.unique(bands, return_inverse=True)
    age = np.broadcast_to(
        np.eye(len(present))[codes], (len(races), len(ages), len(present))
    )
    race = np.broadcast_to(
        np.eye(len(races))[:, None, :], (len(races), len(ages), len(races))
    )
    return age, np.concatenate([age, race[..., 1:]], axis=-1), race


def fitted_proportions(y, n, t, z, fit, weights):
    """Return modelled proportions per 10,000 at fixed age weights.

    Recover intercepts from observed stratum totals and fitted slopes. A
    stratum with no cases throughout has fitted mean zero at the boundary.
    These are point estimates, not interval estimates. In particular, sums
    of age-specific exponential curves need not have one annual multiplier.
    """
    slopes = z @ fit["beta"][0]
    relative = np.exp(slopes[..., None] * t)
    denominator = (n * relative).sum(-1)
    intercept = y.sum(-1) / denominator
    return np.einsum("rat,a->rt", intercept[..., None] * relative, weights) * 10000
