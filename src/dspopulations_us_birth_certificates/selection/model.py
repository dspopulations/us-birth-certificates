"""PyMC implementation of the three-stage selection model.

The model factors observed birth-certificate DS recording as::

    P(R = 1 | X) = theta_LB(age) . eta(X) . s(X) + (1 - theta_LB . eta) . f

where ``theta_LB`` is the baseline DS livebirth rate in the absence of
screening (Morris 2002), ``eta = 1 - eta_detect . eta_term`` is the
screening/termination pass-through (Kuppermann/Natoli), ``s`` is
birth-certificate sensitivity given a DS livebirth (Boulet 2011 /
Salemi 2017), and ``f`` is the small false-positive rate pinned from
Ohio/NY validation.

Cell schema
-----------
Input is a pandas DataFrame with one row per cell containing integer
index columns matching the vocabularies in :mod:`priors`:

    ``year_idx``, ``age_idx``, ``race_idx``, ``edu_idx``, ``payer_idx``,
    ``preterm``, ``cchd``, ``nicu``, ``aven``, ``N_cell`` (total
    livebirths) and ``R_cell`` (recorded DS count).

Use :func:`dspopulations_us_birth_certificates.selection.data.prepare_cells`
to build this frame from the project's DuckDB.

Staged builds
-------------
``build_model`` accepts a ``spec`` flag selecting which stages are active:

    ``"theta_only"`` — Stage 1 only (eta=1, s=1)
    ``"theta_s"``    — Stage 1 + Stage 3 (eta=1)
    ``"single_eta"`` — Stage 1 + single combined eta + Stage 3
    ``"full"``       — Stage 1 + eta_detect × eta_term + Stage 3 (default)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from dspopulations_us_birth_certificates.selection.priors import (
    N_AGE,
    N_EDU,
    N_PAYER,
    N_RACE,
    ModelPriors,
)

if TYPE_CHECKING:
    import pymc as pm

Spec = Literal["theta_only", "theta_s", "single_eta", "full"]

SPECS: tuple[Spec, ...] = ("theta_only", "theta_s", "single_eta", "full")


def build_model(
    cells: pd.DataFrame,
    priors: ModelPriors,
    *,
    spec: Spec = "full",
    n_year: int,
) -> pm.Model:
    """Build the PyMC model for a given spec.

    Args:
        cells: Cell-level aggregated frame (see module docstring).
        priors: Prior specification — one of the ``variant_*`` factories in
            :mod:`priors`.
        spec: Which stages to enable.
        n_year: Number of year levels in the data (e.g. 9 for 2016-2024).
    """
    import pymc as pm
    import pytensor.tensor as pt

    if spec not in SPECS:
        raise ValueError(f"Unknown spec: {spec!r}. Valid: {SPECS}")

    age_idx = cells["age_idx"].to_numpy()
    race_idx = cells["race_idx"].to_numpy()
    edu_idx = cells["edu_idx"].to_numpy()
    payer_idx = cells["payer_idx"].to_numpy()
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
        "year": np.arange(n_year),
        "cell": np.arange(len(cells)),
    }

    with pm.Model(coords=coords) as model:
        # --- Stage 1: theta_LB ----------------------------------------- #
        # Note: per-cell theta_lb is NOT saved as a Deterministic — a full
        # idata for 60k cells × 6000 draws × 4 chains would exceed 30 GB.
        # Downstream diagnostics that need theta_lb per cell reconstruct it
        # from theta_lb_age[age_idx]; see selection.diagnostics.
        theta_lb_age = pm.Normal(
            "theta_lb_age",
            mu=priors.theta_lb_logit,
            sigma=priors.theta_lb_sigma,
            dims="age",
        )
        theta_lb = pm.math.invlogit(theta_lb_age[age_idx])

        # --- Stage 2: eta (screening + termination) -------------------- #
        # Same size-discipline: only the scalar/low-dim RVs named below go
        # into idata; per-cell eta_detect / eta_term / eta are inline
        # tensors.
        if spec in ("single_eta", "full"):
            if spec == "single_eta":
                eta_int = pm.Normal(
                    "eta_int",
                    mu=priors.eta_term_logit,
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
                eta = pm.math.invlogit(
                    eta_int + eta_race[race_idx] + eta_edu[edu_idx]
                )
            else:
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
                eta_detect = pm.math.invlogit(
                    eta_det_int
                    + eta_det_year[year_idx]
                    + eta_det_age[age_idx]
                    + eta_det_race[race_idx]
                    + eta_det_edu[edu_idx]
                    + eta_det_payer[payer_idx]
                )

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
                eta_term_year = pm.Normal(
                    "eta_term_year",
                    mu=0.0,
                    sigma=priors.eta_term_year_sigma,
                    dims="year",
                )
                eta_term = pm.math.invlogit(
                    eta_term_int
                    + eta_term_race[race_idx]
                    + eta_term_edu[edu_idx]
                    + eta_term_year[year_idx]
                )
                eta = 1.0 - eta_detect * eta_term
        else:
            eta = pt.ones_like(theta_lb)

        # --- Stage 3: BC sensitivity s --------------------------------- #
        if spec in ("theta_s", "single_eta", "full"):
            s_int = pm.Normal("s_int", mu=priors.s_logit, sigma=priors.s_sigma)
            s_race = pm.Normal(
                "s_race",
                mu=priors.s_race,
                sigma=priors.s_race_sigma,
                dims="race",
            )
            s_edu = pm.Normal(
                "s_edu",
                mu=priors.s_edu,
                sigma=priors.s_edu_sigma,
                dims="edu",
            )
            s_preterm = pm.Normal(
                "s_preterm",
                mu=priors.s_preterm_mu,
                sigma=priors.s_preterm_sigma,
            )
            s_cchd = pm.Normal(
                "s_cchd", mu=priors.s_cchd_mu, sigma=priors.s_cchd_sigma
            )
            s_nicu = pm.Normal(
                "s_nicu", mu=priors.s_nicu_mu, sigma=priors.s_nicu_sigma
            )
            s_aven = pm.Normal(
                "s_aven", mu=priors.s_aven_mu, sigma=priors.s_aven_sigma
            )
            s = pm.math.invlogit(
                s_int
                + s_race[race_idx]
                + s_edu[edu_idx]
                + s_preterm * preterm
                + s_cchd * cchd
                + s_nicu * nicu
                + s_aven * aven
            )
        else:
            s = pt.ones_like(theta_lb)

        # --- Likelihood ------------------------------------------------ #
        p_ds_lb = pm.Deterministic("p_ds_lb", theta_lb * eta, dims="cell")
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


# --------------------------------------------------------------------------- #
# Posterior extraction helpers                                                #
# --------------------------------------------------------------------------- #


def extract_true_counts(idata, cells: pd.DataFrame) -> pd.DataFrame:
    """Per-cell posterior mean + 95% CI of true DS livebirth counts."""
    p_ds_lb = idata.posterior["p_ds_lb"]
    N = cells["N_cell"].to_numpy()
    true_counts = p_ds_lb * N
    summary = pd.DataFrame(
        {
            "true_count_mean": true_counts.mean(dim=("chain", "draw")).values,
            "true_count_lo": true_counts.quantile(
                0.025, dim=("chain", "draw")
            ).values,
            "true_count_hi": true_counts.quantile(
                0.975, dim=("chain", "draw")
            ).values,
            "N_cell": N,
            "R_cell": cells["R_cell"].to_numpy(),
        }
    )
    for col in (
        "year_idx",
        "age_idx",
        "race_idx",
        "edu_idx",
        "payer_idx",
    ):
        if col in cells.columns:
            summary[col] = cells[col].to_numpy()
    return summary


def posterior_subgroup_rate(
    idata,
    cells: pd.DataFrame,
    group_col: str,
    quantity: str = "true_rate",
) -> pd.DataFrame:
    """N-weighted posterior rate aggregated by a single subgroup column.

    ``quantity`` is one of ``"true_rate"`` (``theta_LB . eta``) or
    ``"recorded_rate"`` (``p_recorded``). ``s`` is not retained on
    the posterior (per-cell, dropped at idata-write time for size),
    so a sensitivity option is intentionally not offered here.
    """
    key = {
        "true_rate": "p_ds_lb",
        "recorded_rate": "p_recorded",
    }[quantity]
    q = idata.posterior[key]
    N = cells["N_cell"].to_numpy()
    groups = cells[group_col].to_numpy()
    out = []
    for g in np.unique(groups):
        mask = groups == g
        group_rate = (q[:, :, mask] * N[mask]).sum(dim="cell") / N[mask].sum()
        out.append(
            {
                group_col: g,
                "mean": group_rate.mean().item(),
                "lo": group_rate.quantile(0.025).item(),
                "hi": group_rate.quantile(0.975).item(),
            }
        )
    return pd.DataFrame(out)
