"""
simulate.py
===========

Forward-simulate synthetic cell-level data from the three-stage model.
Used to validate that the Bayesian fit recovers known parameter values.

If the model cannot recover ground truth from data it generated itself,
it cannot be trusted on real data.  Run this before fitting on NCHS data.

Usage
-----
    from simulate import simulate_cells, TrueParams
    from priors import variant_C_default
    from model import build_model
    import pymc as pm

    # Choose a "ground truth" parameter set
    truth = TrueParams.from_priors(variant_C_default(), seed=0)

    # Simulate cells
    cells = simulate_cells(truth, n_cells_per_month=400,
                           n_year=9, n_region=4,
                           post_dobbs_year_start=6, seed=0)

    # Fit and compare
    model = build_model(cells, variant_C_default(), spec="full",
                        n_year=9, n_region=4, post_dobbs_year_start=6)
    with model:
        idata = pm.sample(1000, tune=1000, chains=4)

    # Compare posterior to `truth`
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from priors import (
    ModelPriors, inv_logit,
    N_AGE, N_RACE, N_EDU, N_PAYER,
)


@dataclass
class TrueParams:
    """Concrete ground-truth parameter values for forward simulation."""

    # Stage 1
    theta_lb_age_logit: np.ndarray          # shape (N_AGE,)

    # Stage 2a
    eta_det_int: float
    eta_det_year: np.ndarray                # shape (n_year,)
    eta_det_age: np.ndarray                 # shape (N_AGE,)
    eta_det_race: np.ndarray                # shape (N_RACE,)
    eta_det_edu: np.ndarray                 # shape (N_EDU,)
    eta_det_payer: np.ndarray               # shape (N_PAYER,)

    # Stage 2b
    eta_term_int: float
    eta_term_race: np.ndarray               # shape (N_RACE,)
    eta_term_edu: np.ndarray                # shape (N_EDU,)
    eta_term_ry: np.ndarray                 # shape (n_region, n_year)

    # Stage 3
    s_int: float
    s_race: np.ndarray                      # shape (N_RACE,)
    s_edu: np.ndarray                       # shape (N_EDU,)
    s_preterm: float
    s_cchd: float
    s_nicu: float
    s_aven: float

    # False positive rate
    false_positive_rate: float

    @classmethod
    def from_priors(
        cls,
        priors: ModelPriors,
        *,
        n_year: int = 9,
        n_region: int = 4,
        post_dobbs_year_start: int = 6,
        seed: int | None = None,
    ) -> "TrueParams":
        """
        Draw a single parameter set from the prior distribution.

        Useful for generating "plausible" ground truth consistent with the
        prior so that the fit's recovery is a meaningful validation.
        """
        rng = np.random.default_rng(seed)

        def draw(mu, sigma, size=None):
            return rng.normal(mu, sigma, size=size)

        year_offsets = priors.eta_detect_year_offsets[:n_year]

        # Region x year with heterogeneous sigma
        sigma_ry = np.where(
            np.arange(n_year) >= post_dobbs_year_start,
            priors.eta_term_ry_sigma_post_dobbs,
            priors.eta_term_ry_sigma_pre_dobbs,
        )
        eta_term_ry = rng.normal(
            0.0,
            np.broadcast_to(sigma_ry[None, :], (n_region, n_year)),
        )

        return cls(
            theta_lb_age_logit=draw(
                priors.theta_lb_logit, priors.theta_lb_sigma),
            eta_det_int=draw(priors.eta_detect_logit, priors.eta_detect_sigma),
            eta_det_year=draw(year_offsets, priors.eta_detect_year_sigma),
            eta_det_age=draw(0.0, priors.eta_detect_age_sigma, size=N_AGE),
            eta_det_race=draw(
                priors.eta_detect_race, priors.eta_detect_race_sigma),
            eta_det_edu=draw(
                priors.eta_detect_edu, priors.eta_detect_edu_sigma),
            eta_det_payer=draw(
                priors.eta_detect_payer, priors.eta_detect_payer_sigma),
            eta_term_int=draw(priors.eta_term_logit, priors.eta_term_sigma),
            eta_term_race=draw(
                priors.eta_term_race, priors.eta_term_race_sigma),
            eta_term_edu=draw(
                priors.eta_term_edu, priors.eta_term_edu_sigma),
            eta_term_ry=eta_term_ry,
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
    n_region: int,
    post_dobbs_year_start: int,
    n_cells_mean: int = 12_000,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Forward-simulate a synthetic cell-level dataset.

    Parameters
    ----------
    truth : TrueParams
        Ground truth parameter values.
    n_cells_per_month : int
        Number of distinct cells per (year, month) bin.  Cells are drawn by
        sampling covariate profiles uniformly at random.
    n_year, n_region, post_dobbs_year_start
        Must match what will be passed to build_model().
    n_cells_mean : int
        Mean of Poisson-distributed cell size (N_cell).

    Returns
    -------
    pd.DataFrame
        A cells DataFrame ready for build_model().
    """
    rng = np.random.default_rng(seed)

    # Draw covariate profiles at random for each cell
    n_cells = n_cells_per_month * 12 * n_year

    age_idx = rng.integers(0, N_AGE, size=n_cells)
    race_idx = rng.integers(0, N_RACE, size=n_cells)
    edu_idx = rng.integers(0, N_EDU, size=n_cells)
    payer_idx = rng.integers(0, N_PAYER, size=n_cells)
    region_idx = rng.integers(0, n_region, size=n_cells)
    year_idx = rng.integers(0, n_year, size=n_cells)
    preterm = rng.integers(0, 2, size=n_cells)
    cchd = rng.integers(0, 2, size=n_cells)
    nicu = rng.integers(0, 2, size=n_cells)
    aven = rng.integers(0, 2, size=n_cells)

    # Cell population sizes
    N_cell = rng.poisson(n_cells_mean, size=n_cells)
    N_cell = np.clip(N_cell, 100, None)

    # Compute per-cell probabilities using the TRUE parameters
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
        + truth.eta_term_ry[region_idx, year_idx]
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
    p_recorded = p_ds_lb * s + (1.0 - p_ds_lb) * truth.false_positive_rate

    # Draw observed recorded counts
    R_cell = rng.binomial(N_cell, p_recorded)

    cells = pd.DataFrame({
        "age_idx": age_idx,
        "race_idx": race_idx,
        "edu_idx": edu_idx,
        "payer_idx": payer_idx,
        "region_idx": region_idx,
        "year_idx": year_idx,
        "preterm": preterm,
        "cchd": cchd,
        "nicu": nicu,
        "aven": aven,
        "N_cell": N_cell,
        "R_cell": R_cell,
        # Store the true per-cell probabilities for downstream comparison
        "true_theta_lb": theta_lb,
        "true_eta_detect": eta_detect,
        "true_eta_term": eta_term,
        "true_eta": eta,
        "true_s": s,
        "true_p_ds_lb": p_ds_lb,
        "true_p_recorded": p_recorded,
    })
    return cells


if __name__ == "__main__":
    from priors import variant_C_default

    truth = TrueParams.from_priors(variant_C_default(), seed=42)
    cells = simulate_cells(
        truth,
        n_cells_per_month=50,
        n_year=9,
        n_region=4,
        post_dobbs_year_start=6,
        seed=42,
    )
    print(f"Simulated {len(cells):,} cells")
    print(f"Total livebirths: {cells['N_cell'].sum():,}")
    print(f"Total recorded DS: {cells['R_cell'].sum():,}")
    print(f"Recorded rate: {cells['R_cell'].sum() / cells['N_cell'].sum():.2e}")
    print()
    print("True rate summary (across cells):")
    print(f"  θ_LB mean:       {cells['true_theta_lb'].mean():.4f}")
    print(f"  η_detect mean:   {cells['true_eta_detect'].mean():.4f}")
    print(f"  η_term mean:     {cells['true_eta_term'].mean():.4f}")
    print(f"  η (pass) mean:   {cells['true_eta'].mean():.4f}")
    print(f"  s mean:          {cells['true_s'].mean():.4f}")
    print(f"  Implied recorded rate: "
          f"{(cells['true_theta_lb'] * cells['true_eta'] * cells['true_s']).mean():.2e}")
