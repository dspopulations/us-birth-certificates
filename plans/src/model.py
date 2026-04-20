"""
model.py
========

PyMC implementation of the three-stage Bayesian selection model for Down
syndrome livebirth ascertainment, 2016-2024.

The model factors observed birth-certificate DS recording as

    P(R = 1 | X) = θ_LB(age) · η(X) · s(X) + (1 − θ_LB · η) · f

where
    θ_LB  - baseline livebirth rate in absence of screening (Morris 2002)
    η     - 1 − η_detect · η_term (screening/termination pass-through)
    s     - BC sensitivity given DS livebirth (Boulet 2011 / Salemi 2017)
    f     - false-positive rate (Ohio/NY study)

Data preparation
----------------
Input is a pandas DataFrame with one row per cell, containing:
    year_idx, age_idx, race_idx, edu_idx, payer_idx, region_idx, is_post_dobbs
    preterm, cchd, nicu, aven   (0/1 flags)
    N_cell                       (total livebirths in the cell)
    R_cell                       (recorded DS count in the cell)

Cell indices must be contiguous integers starting at 0 and match the
vocabularies in priors.py.  Use data.prepare_cells() to build this frame
from raw NCHS data.

Staged builds
-------------
The build_model() function accepts a `spec` argument selecting which model
components to enable:

    "theta_only"  Stage A  - θ_LB only, η=1, s=1
    "theta_s"     Stage B  - θ_LB + s, η=1
    "single_eta"  Stage C  - θ_LB + single η + s
    "full"        Stage D  - θ_LB + η_detect × η_term + s  (default)

Usage
-----
    from model import build_model
    from priors import variant_C_default
    import pymc as pm

    priors = variant_C_default()
    model = build_model(cells_df, priors, spec="full", n_year=9,
                        n_region=50, post_dobbs_year_start=6)

    with model:
        idata = pm.sample(1000, tune=1000, chains=4, target_accept=0.9)
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from priors import (
    ModelPriors,
    N_AGE, N_RACE, N_EDU, N_PAYER,
)


Spec = Literal["theta_only", "theta_s", "single_eta", "full"]


def build_model(
    cells: pd.DataFrame,
    priors: ModelPriors,
    *,
    spec: Spec = "full",
    n_year: int,
    n_region: int,
    post_dobbs_year_start: int,
) -> pm.Model:
    """
    Build the PyMC model.

    Parameters
    ----------
    cells : pd.DataFrame
        Cell-level aggregated data.  See module docstring for required columns.
    priors : ModelPriors
        Prior specification.  Use priors.variant_C_default() or a sensitivity
        variant.
    spec : {"theta_only", "theta_s", "single_eta", "full"}
        Which model components to enable.  See module docstring.
    n_year : int
        Number of distinct years in the data.
    n_region : int
        Number of distinct regions/states.
    post_dobbs_year_start : int
        Index of the first year-level treated as "post-Dobbs" (e.g. if
        year_idx=0 is 2016 and Dobbs is mid-2022, pass 6 for 2022 if you
        treat the whole year as post, or 7 for 2023).
    """
    if spec not in ("theta_only", "theta_s", "single_eta", "full"):
        raise ValueError(f"Unknown spec: {spec!r}")

    # --- Extract arrays ---------------------------------------------------
    age_idx = cells["age_idx"].to_numpy()
    race_idx = cells["race_idx"].to_numpy()
    edu_idx = cells["edu_idx"].to_numpy()
    payer_idx = cells["payer_idx"].to_numpy()
    region_idx = cells["region_idx"].to_numpy()
    year_idx = cells["year_idx"].to_numpy()
    preterm = cells["preterm"].to_numpy().astype(float)
    cchd = cells["cchd"].to_numpy().astype(float)
    nicu = cells["nicu"].to_numpy().astype(float)
    aven = cells["aven"].to_numpy().astype(float)
    N_cell = cells["N_cell"].to_numpy()
    R_cell = cells["R_cell"].to_numpy()

    coords = {
        "age": np.arange(N_AGE),
        "race": np.arange(N_RACE),
        "edu": np.arange(N_EDU),
        "payer": np.arange(N_PAYER),
        "region": np.arange(n_region),
        "year": np.arange(n_year),
        "cell": np.arange(len(cells)),
    }

    with pm.Model(coords=coords) as model:
        # -------------------------------------------------------------- #
        # Stage 1: θ_LB  (baseline livebirth rate, Morris)               #
        # -------------------------------------------------------------- #
        theta_lb_age = pm.Normal(
            "theta_lb_age",
            mu=priors.theta_lb_logit,
            sigma=priors.theta_lb_sigma,
            dims="age",
        )
        theta_lb = pm.Deterministic(
            "theta_lb",
            pm.math.invlogit(theta_lb_age[age_idx]),
            dims="cell",
        )

        # -------------------------------------------------------------- #
        # Stage 2: η  (screening + termination)                          #
        # -------------------------------------------------------------- #
        if spec in ("single_eta", "full"):
            if spec == "single_eta":
                # One combined pass-through rate (no detect/term split)
                eta_int = pm.Normal(
                    "eta_int",
                    mu=priors.eta_term_logit,   # repurpose as combined anchor
                    sigma=priors.eta_term_sigma,
                )
                eta_race = pm.Normal(
                    "eta_race",
                    mu=priors.eta_term_race,
                    sigma=priors.eta_term_race_sigma,
                    dims="race",
                )
                eta_edu = pm.Normal(
                    "eta_edu",
                    mu=priors.eta_term_edu,
                    sigma=priors.eta_term_edu_sigma,
                    dims="edu",
                )
                # In single-η mode, η is the *pass-through fraction*
                # (analogous to 1 - detect*term in the full spec).
                eta = pm.Deterministic(
                    "eta",
                    pm.math.invlogit(
                        eta_int + eta_race[race_idx] + eta_edu[edu_idx]
                    ),
                    dims="cell",
                )
            else:
                # Full spec: detect and term separately
                # --- η_detect --------------------------------------------
                eta_det_int = pm.Normal(
                    "eta_detect_int",
                    mu=priors.eta_detect_logit,
                    sigma=priors.eta_detect_sigma,
                )
                eta_det_year = pm.Normal(
                    "eta_detect_year",
                    mu=priors.eta_detect_year_offsets[:n_year],
                    sigma=priors.eta_detect_year_sigma,
                    dims="year",
                )
                eta_det_age = pm.Normal(
                    "eta_detect_age",
                    mu=0.0,
                    sigma=priors.eta_detect_age_sigma,
                    dims="age",
                )
                eta_det_race = pm.Normal(
                    "eta_detect_race",
                    mu=priors.eta_detect_race,
                    sigma=priors.eta_detect_race_sigma,
                    dims="race",
                )
                eta_det_edu = pm.Normal(
                    "eta_detect_edu",
                    mu=priors.eta_detect_edu,
                    sigma=priors.eta_detect_edu_sigma,
                    dims="edu",
                )
                eta_det_payer = pm.Normal(
                    "eta_detect_payer",
                    mu=priors.eta_detect_payer,
                    sigma=priors.eta_detect_payer_sigma,
                    dims="payer",
                )

                eta_detect = pm.Deterministic(
                    "eta_detect",
                    pm.math.invlogit(
                        eta_det_int
                        + eta_det_year[year_idx]
                        + eta_det_age[age_idx]
                        + eta_det_race[race_idx]
                        + eta_det_edu[edu_idx]
                        + eta_det_payer[payer_idx]
                    ),
                    dims="cell",
                )

                # --- η_term ----------------------------------------------
                eta_term_int = pm.Normal(
                    "eta_term_int",
                    mu=priors.eta_term_logit,
                    sigma=priors.eta_term_sigma,
                )
                eta_term_race = pm.Normal(
                    "eta_term_race",
                    mu=priors.eta_term_race,
                    sigma=priors.eta_term_race_sigma,
                    dims="race",
                )
                eta_term_edu = pm.Normal(
                    "eta_term_edu",
                    mu=priors.eta_term_edu,
                    sigma=priors.eta_term_edu_sigma,
                    dims="edu",
                )
                # Region x year interaction: heterogeneous sigma for
                # pre vs post Dobbs.
                sigma_ry = np.where(
                    np.arange(n_year) >= post_dobbs_year_start,
                    priors.eta_term_ry_sigma_post_dobbs,
                    priors.eta_term_ry_sigma_pre_dobbs,
                )
                # Broadcast sigma to shape (n_region, n_year)
                sigma_ry_2d = np.broadcast_to(
                    sigma_ry[None, :], (n_region, n_year)
                )
                eta_term_ry = pm.Normal(
                    "eta_term_ry",
                    mu=0.0,
                    sigma=sigma_ry_2d,
                    dims=("region", "year"),
                )

                eta_term = pm.Deterministic(
                    "eta_term",
                    pm.math.invlogit(
                        eta_term_int
                        + eta_term_race[race_idx]
                        + eta_term_edu[edu_idx]
                        + eta_term_ry[region_idx, year_idx]
                    ),
                    dims="cell",
                )

                # Pass-through: a pregnancy becomes a livebirth if NOT
                # (detected AND terminated).
                eta = pm.Deterministic(
                    "eta",
                    1.0 - eta_detect * eta_term,
                    dims="cell",
                )
        else:
            # spec in ("theta_only", "theta_s"): no η component; η=1
            eta = pt.ones_like(theta_lb)

        # -------------------------------------------------------------- #
        # Stage 3: BC sensitivity s                                       #
        # -------------------------------------------------------------- #
        if spec in ("theta_s", "single_eta", "full"):
            s_int = pm.Normal(
                "s_int", mu=priors.s_logit, sigma=priors.s_sigma
            )
            s_race = pm.Normal(
                "s_race", mu=priors.s_race,
                sigma=priors.s_race_sigma, dims="race",
            )
            s_edu = pm.Normal(
                "s_edu", mu=priors.s_edu,
                sigma=priors.s_edu_sigma, dims="edu",
            )
            s_preterm = pm.Normal(
                "s_preterm", mu=priors.s_preterm_mu,
                sigma=priors.s_preterm_sigma,
            )
            s_cchd = pm.Normal(
                "s_cchd", mu=priors.s_cchd_mu,
                sigma=priors.s_cchd_sigma,
            )
            s_nicu = pm.Normal(
                "s_nicu", mu=priors.s_nicu_mu,
                sigma=priors.s_nicu_sigma,
            )
            s_aven = pm.Normal(
                "s_aven", mu=priors.s_aven_mu,
                sigma=priors.s_aven_sigma,
            )

            s = pm.Deterministic(
                "s",
                pm.math.invlogit(
                    s_int
                    + s_race[race_idx]
                    + s_edu[edu_idx]
                    + s_preterm * preterm
                    + s_cchd * cchd
                    + s_nicu * nicu
                    + s_aven * aven
                ),
                dims="cell",
            )
        else:
            # spec == "theta_only": s=1
            s = pt.ones_like(theta_lb)

        # -------------------------------------------------------------- #
        # Observed-data likelihood                                        #
        # -------------------------------------------------------------- #
        p_ds_lb = pm.Deterministic(
            "p_ds_lb", theta_lb * eta, dims="cell"
        )
        p_recorded = pm.Deterministic(
            "p_recorded",
            p_ds_lb * s + (1.0 - p_ds_lb) * priors.false_positive_rate,
            dims="cell",
        )

        pm.Binomial(
            "R_obs",
            n=N_cell,
            p=p_recorded,
            observed=R_cell,
            dims="cell",
        )

    return model


# ---------------------------------------------------------------------------- #
# Posterior extraction helpers                                                 #
# ---------------------------------------------------------------------------- #

def extract_true_counts(idata, cells: pd.DataFrame) -> pd.DataFrame:
    """
    From a sampled InferenceData, compute per-cell posterior over *true*
    DS livebirth counts (N_cell * θ_LB * η) with credible intervals.
    """
    p_ds_lb = idata.posterior["p_ds_lb"]   # (chain, draw, cell)
    N = cells["N_cell"].to_numpy()
    true_counts = p_ds_lb * N              # broadcast
    summary = pd.DataFrame({
        "true_count_mean": true_counts.mean(dim=("chain", "draw")).values,
        "true_count_lo":   true_counts.quantile(0.025, dim=("chain", "draw")).values,
        "true_count_hi":   true_counts.quantile(0.975, dim=("chain", "draw")).values,
        "N_cell": N,
        "R_cell": cells["R_cell"].to_numpy(),
    })
    for col in ("year_idx", "age_idx", "race_idx", "edu_idx", "payer_idx",
                "region_idx"):
        if col in cells.columns:
            summary[col] = cells[col].to_numpy()
    return summary


def posterior_subgroup_rate(
    idata,
    cells: pd.DataFrame,
    group_col: str,
    quantity: str = "true_rate",
) -> pd.DataFrame:
    """
    Aggregate posterior true-DS-livebirth rate by a single subgroup column.

    quantity can be one of:
        "true_rate"  -  θ_LB·η  (true DS livebirths per livebirth)
        "recorded_rate"  -  p_recorded (matches observed)
        "sensitivity"  -  s  (BC sensitivity)
    """
    key = {
        "true_rate": "p_ds_lb",
        "recorded_rate": "p_recorded",
        "sensitivity": "s",
    }[quantity]
    q = idata.posterior[key]   # (chain, draw, cell)
    N = cells["N_cell"].to_numpy()

    groups = cells[group_col].to_numpy()
    out = []
    for g in np.unique(groups):
        mask = groups == g
        # N-weighted mean across cells in the group
        group_rate = (q[:, :, mask] * N[mask]).sum(dim="cell") / N[mask].sum()
        out.append({
            group_col: g,
            "mean": group_rate.mean().item(),
            "lo":   group_rate.quantile(0.025).item(),
            "hi":   group_rate.quantile(0.975).item(),
        })
    return pd.DataFrame(out)
