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

from dspopulations_us_birth_certificates.intervals import (
    eti_quantiles,
    interval_label,
)
from dspopulations_us_birth_certificates.selection.priors import (
    N_AGE,
    N_EDU,
    N_PAYER,
    N_RACE,
    ModelPriors,
)
from dspopulations_us_birth_certificates.selection.recording_anchor import ANCHOR_YEARS

if TYPE_CHECKING:
    import pymc as pm

Spec = Literal["theta_only", "theta_s", "single_eta", "full"]

SPECS: tuple[Spec, ...] = ("theta_only", "theta_s", "single_eta", "full")
ETI_LO_Q, ETI_HI_Q = eti_quantiles()


def year_slice_for_anchor(start_year: int, n_year: int) -> slice:
    """Return the anchor-array column slice for a contiguous year window."""
    if n_year <= 0:
        raise ValueError(f"n_year must be positive, got {n_year!r}")
    first = ANCHOR_YEARS[0]
    last = ANCHOR_YEARS[-1]
    end_year = start_year + n_year - 1
    if start_year < first or end_year > last:
        raise ValueError(
            f"Year range {start_year}-{end_year} is outside available "
            f"anchor years {first}-{last}."
        )
    offset = start_year - first
    return slice(offset, offset + n_year)


def build_model(
    cells: pd.DataFrame,
    priors: ModelPriors,
    *,
    spec: Spec = "full",
    n_year: int,
    start_year: int = ANCHOR_YEARS[0],
    prev_margin: tuple[np.ndarray, np.ndarray] | None = None,
) -> pm.Model:
    """Build the PyMC model for a given spec.

    Args:
        cells: Cell-level aggregated frame (see module docstring).
        priors: Prior specification — one of the ``variant_*`` factories in
            :mod:`priors`.
        spec: Which stages to enable.
        n_year: Number of year levels in the data (e.g. 9 for 2016-2024).
        start_year: Calendar year corresponding to ``year_idx == 0``. Used to
            align year-indexed priors/anchors with subset windows.
        prev_margin: Optional full-margin anchor ``(target, sigma)``, each a
            ``[N_RACE, n_year]`` array of de Graaf TRUE prevalence per 10k (NaN where
            unanchored, e.g. the Unknown race). When given, a soft Normal observation
            ties the model's N-weighted marginal ``p_ds_lb`` per race×year to de Graaf's
            surveillance prevalence — pinning η's race×year level (θ is already pinned to
            Morris) so the literature η priors cannot drag the total below surveillance.
            See ``selection.recording_anchor`` and ``scripts/derive_recording_rates.py``.
    """
    import pymc as pm
    import pytensor.tensor as pt

    if spec not in SPECS:
        raise ValueError(f"Unknown spec: {spec!r}. Valid: {SPECS}")

    year_slice = year_slice_for_anchor(start_year, n_year)

    age_idx = cells["age_idx"].to_numpy()
    race_idx = cells["race_idx"].to_numpy()
    edu_idx = cells["edu_idx"].to_numpy()
    payer_idx = cells["payer_idx"].to_numpy()
    year_idx = cells["year_idx"].to_numpy()
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
                eta = pm.math.invlogit(eta_int + eta_race[race_idx] + eta_edu[edu_idx])
            else:
                eta_det_int = pm.Normal(
                    "eta_detect_int",
                    mu=priors.eta_detect_logit,
                    sigma=priors.eta_detect_sigma,
                )
                eta_det_year = pm.Normal(
                    "eta_detect_year",
                    mu=priors.eta_detect_year_offsets[year_slice],
                    sigma=priors.eta_detect_year_sigma,
                    dims="year",
                )
                eta_det_age = pm.Normal(
                    "eta_detect_age",
                    mu=priors.eta_detect_age,
                    sigma=priors.eta_detect_age_sigma,
                    dims="age",
                )
                # Zero-sum year-by-age interaction: lets the screening rollout
                # differ by maternal age (the "older mothers first?" question)
                # without shifting the pinned year or age main effects.
                eta_det_year_age = pm.ZeroSumNormal(
                    "eta_detect_year_age",
                    sigma=priors.eta_detect_year_age_sigma,
                    n_zerosum_axes=2,
                    dims=("year", "age"),
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
                    + eta_det_year_age[year_idx, age_idx]
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
                eta_term_age = pm.Normal(
                    "eta_term_age",
                    mu=priors.eta_term_age,
                    sigma=priors.eta_term_age_sigma,
                    dims="age",
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
                    + eta_term_age[age_idx]
                    + eta_term_year[year_idx]
                )
                eta = 1.0 - eta_detect * eta_term
        else:
            eta = pt.ones_like(theta_lb)

        # --- Stage 3: BC sensitivity s (de Graaf surveillance anchor) --- #
        # s(race, year) is anchored to recorded/true derived from de Graaf prevalence
        # (priors.s_race_year_*; see selection.recording_anchor). The external anchor
        # supplies the recording LEVEL, the racial gradient AND a year dimension, breaking
        # the eta x s ridge with data rather than the old hard pin on s_int. s_edu stays a
        # small within-cell education residual. A year-averaged ``s_race`` Deterministic is
        # retained so the identifiability / coefficient-forest / parameter-recovery
        # diagnostics that summarise s by race keep working unchanged.
        if spec in ("theta_s", "single_eta", "full"):
            s_race_year = pm.Normal(
                "s_race_year",
                mu=priors.s_race_year_logit[:, year_slice],
                sigma=priors.s_race_year_sigma[:, year_slice],
                dims=("race", "year"),
            )
            s_edu = pm.Normal(
                "s_edu",
                mu=priors.s_edu,
                sigma=priors.s_edu_sigma,
                dims="edu",
            )
            pm.Deterministic("s_race", s_race_year.mean(axis=1), dims="race")
            s = pm.math.invlogit(s_race_year[race_idx, year_idx] + s_edu[edu_idx])
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

        # --- Full-margin anchor: tie marginal p_ds_lb per race×year to de Graaf --- #
        if prev_margin is not None:
            target_mat = np.asarray(prev_margin[0], dtype=float)
            sigma_mat = np.asarray(prev_margin[1], dtype=float)
            rows, tvec, svec = [], [], []
            for r in range(target_mat.shape[0]):
                for y in range(n_year):
                    t = target_mat[r, y]
                    if not np.isfinite(t):
                        continue
                    w = np.where((race_idx == r) & (year_idx == y), N_cell, 0.0)
                    if w.sum() <= 0:
                        continue
                    rows.append(
                        w / w.sum()
                    )  # N-weighted average over the cell's age/edu/...
                    tvec.append(t)
                    svec.append(sigma_mat[r, y])
            if rows:
                W = np.asarray(rows)  # (n_group, n_cell)
                model.add_coord("prev_group", np.arange(len(rows)))
                # marginal true DS prevalence per 10k for each anchored race×year group
                margin = pm.Deterministic(
                    "prev_margin", 1e4 * pt.dot(W, p_ds_lb), dims="prev_group"
                )
                pm.Normal(
                    "prev_margin_obs",
                    mu=margin,
                    sigma=np.asarray(svec),
                    observed=np.asarray(tvec),
                    dims="prev_group",
                )

    return model


# --------------------------------------------------------------------------- #
# Posterior extraction helpers                                                #
# --------------------------------------------------------------------------- #


def extract_true_counts(idata, cells: pd.DataFrame) -> pd.DataFrame:
    """Per-cell posterior mean + project-standard ETI of true DS livebirth counts."""
    p_ds_lb = idata.posterior["p_ds_lb"]
    N = cells["N_cell"].to_numpy()
    true_counts = p_ds_lb * N
    summary = pd.DataFrame(
        {
            "true_count_mean": true_counts.mean(dim=("chain", "draw")).values,
            "true_count_lo": true_counts.quantile(
                ETI_LO_Q, dim=("chain", "draw")
            ).values,
            "true_count_hi": true_counts.quantile(
                ETI_HI_Q, dim=("chain", "draw")
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
                "lo": group_rate.quantile(ETI_LO_Q).item(),
                "hi": group_rate.quantile(ETI_HI_Q).item(),
                "interval": interval_label(),
            }
        )
    return pd.DataFrame(out)
