"""Forward simulator for the three-stage selection model.

Generates synthetic cell-level data from a known ground truth so the
Bayesian fit can be validated by parameter-recovery. If the model cannot
recover truth from data it generated itself, it should not be trusted on
real data.

Example
-------
    from dspopulations_us_birth_certificates.selection import (
        TrueParams, simulate_cells, variant_C_default, build_model,
    )
    import pymc as pm

    truth = TrueParams.from_priors(variant_C_default(), seed=0)
    cells = simulate_cells(
        truth, n_cells_per_month=400,
        n_year=9, post_dobbs_year_start=6, seed=0,
    )
    model = build_model(
        cells, variant_C_default(), spec="full",
        n_year=9, post_dobbs_year_start=6,
    )
    with model:
        idata = pm.sample(1000, tune=1000, chains=4)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dspopulations_us_birth_certificates.selection.priors import (
    N_AGE,
    N_EDU,
    N_PAYER,
    N_RACE,
    ModelPriors,
    inv_logit,
)


@dataclass
class TrueParams:
    """Concrete ground-truth parameter values for forward simulation."""

    theta_lb_age_logit: np.ndarray  # (N_AGE,)

    eta_det_int: float
    eta_det_year: np.ndarray  # (n_year,)
    eta_det_age: np.ndarray  # (N_AGE,)
    eta_det_race: np.ndarray  # (N_RACE,)
    eta_det_edu: np.ndarray  # (N_EDU,)
    eta_det_payer: np.ndarray  # (N_PAYER,)

    eta_term_int: float
    eta_term_race: np.ndarray  # (N_RACE,)
    eta_term_edu: np.ndarray  # (N_EDU,)
    eta_term_year: np.ndarray  # (n_year,)

    s_int: float
    s_race: np.ndarray  # (N_RACE,)
    s_edu: np.ndarray  # (N_EDU,)
    s_preterm: float
    s_cchd: float
    s_nicu: float
    s_aven: float

    false_positive_rate: float

    @classmethod
    def from_priors(
        cls,
        priors: ModelPriors,
        *,
        n_year: int = 9,
        post_dobbs_year_start: int = 6,
        seed: int | None = None,
    ) -> TrueParams:
        """Draw one plausible parameter set from the prior distribution."""
        rng = np.random.default_rng(seed)

        def draw(mu, sigma, size=None):
            return rng.normal(mu, sigma, size=size)

        year_offsets = priors.eta_detect_year_offsets[:n_year]

        sigma_year = np.where(
            np.arange(n_year) >= post_dobbs_year_start,
            priors.eta_term_year_sigma_post_dobbs,
            priors.eta_term_year_sigma_pre_dobbs,
        )
        eta_term_year = rng.normal(0.0, sigma_year)

        return cls(
            theta_lb_age_logit=draw(
                priors.theta_lb_logit, priors.theta_lb_sigma
            ),
            eta_det_int=draw(
                priors.eta_detect_logit, priors.eta_detect_sigma
            ),
            eta_det_year=draw(year_offsets, priors.eta_detect_year_sigma),
            eta_det_age=draw(0.0, priors.eta_detect_age_sigma, size=N_AGE),
            eta_det_race=draw(
                priors.eta_detect_race, priors.eta_detect_race_sigma
            ),
            eta_det_edu=draw(
                priors.eta_detect_edu, priors.eta_detect_edu_sigma
            ),
            eta_det_payer=draw(
                priors.eta_detect_payer, priors.eta_detect_payer_sigma
            ),
            eta_term_int=draw(priors.eta_term_logit, priors.eta_term_sigma),
            eta_term_race=draw(
                priors.eta_term_race, priors.eta_term_race_sigma
            ),
            eta_term_edu=draw(priors.eta_term_edu, priors.eta_term_edu_sigma),
            eta_term_year=eta_term_year,
            s_int=draw(priors.s_logit, priors.s_sigma),
            s_race=draw(priors.s_race, priors.s_race_sigma),
            s_edu=draw(priors.s_edu, priors.s_edu_sigma),
            s_preterm=draw(priors.s_preterm_mu, priors.s_preterm_sigma),
            s_cchd=draw(priors.s_cchd_mu, priors.s_cchd_sigma),
            s_nicu=draw(priors.s_nicu_mu, priors.s_nicu_sigma),
            s_aven=draw(priors.s_aven_mu, priors.s_aven_sigma),
            false_positive_rate=priors.false_positive_rate,
        )


def simulate_cells(
    truth: TrueParams,
    *,
    n_cells_per_month: int = 400,
    n_year: int,
    post_dobbs_year_start: int,  # noqa: ARG001 — kept for CLI symmetry
    n_cells_mean: int = 12_000,
    seed: int | None = None,
) -> pd.DataFrame:
    """Forward-simulate a synthetic cell-level dataset.

    Args:
        truth: Ground-truth parameter values.
        n_cells_per_month: Cells per (year, month). Covariate profiles are
            sampled uniformly at random.
        n_year: Number of years (must match ``build_model``).
        post_dobbs_year_start: Retained for CLI symmetry with ``build_model``
            — the Dobbs sigma is baked into ``truth.eta_term_year`` already.
        n_cells_mean: Mean of the Poisson cell-size distribution.
        seed: RNG seed.

    Returns:
        DataFrame ready for ``build_model``, with the observed columns
        plus ``true_*`` columns carrying the per-cell ground-truth
        probabilities.
    """
    rng = np.random.default_rng(seed)

    n_cells = n_cells_per_month * 12 * n_year

    age_idx = rng.integers(0, N_AGE, size=n_cells)
    race_idx = rng.integers(0, N_RACE, size=n_cells)
    edu_idx = rng.integers(0, N_EDU, size=n_cells)
    payer_idx = rng.integers(0, N_PAYER, size=n_cells)
    year_idx = rng.integers(0, n_year, size=n_cells)
    preterm = rng.integers(0, 2, size=n_cells)
    cchd = rng.integers(0, 2, size=n_cells)
    nicu = rng.integers(0, 2, size=n_cells)
    aven = rng.integers(0, 2, size=n_cells)

    N_cell = rng.poisson(n_cells_mean, size=n_cells)
    N_cell = np.clip(N_cell, 100, None)

    theta_lb = inv_logit(truth.theta_lb_age_logit[age_idx])

    eta_detect = inv_logit(
        truth.eta_det_int
        + truth.eta_det_year[year_idx]
        + truth.eta_det_age[age_idx]
        + truth.eta_det_race[race_idx]
        + truth.eta_det_edu[edu_idx]
        + truth.eta_det_payer[payer_idx]
    )
    eta_term = inv_logit(
        truth.eta_term_int
        + truth.eta_term_race[race_idx]
        + truth.eta_term_edu[edu_idx]
        + truth.eta_term_year[year_idx]
    )
    eta = 1.0 - eta_detect * eta_term

    s = inv_logit(
        truth.s_int
        + truth.s_race[race_idx]
        + truth.s_edu[edu_idx]
        + truth.s_preterm * preterm
        + truth.s_cchd * cchd
        + truth.s_nicu * nicu
        + truth.s_aven * aven
    )

    p_ds_lb = theta_lb * eta
    p_recorded = (
        p_ds_lb * s + (1.0 - p_ds_lb) * truth.false_positive_rate
    )

    R_cell = rng.binomial(N_cell, p_recorded)

    return pd.DataFrame(
        {
            "age_idx": age_idx,
            "race_idx": race_idx,
            "edu_idx": edu_idx,
            "payer_idx": payer_idx,
            "year_idx": year_idx,
            "preterm": preterm,
            "cchd": cchd,
            "nicu": nicu,
            "aven": aven,
            "N_cell": N_cell,
            "R_cell": R_cell,
            "true_theta_lb": theta_lb,
            "true_eta_detect": eta_detect,
            "true_eta_term": eta_term,
            "true_eta": eta,
            "true_s": s,
            "true_p_ds_lb": p_ds_lb,
            "true_p_recorded": p_recorded,
        }
    )
