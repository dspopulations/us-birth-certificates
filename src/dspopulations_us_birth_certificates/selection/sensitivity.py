"""Isolated prior stress tests for the full three-stage selection model."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from dspopulations_us_birth_certificates.selection.priors import (
    ModelPriors,
    inv_logit,
    logit,
)

MODEL_COVARIATES = ("year_idx", "age_idx", "race_idx", "edu_idx", "payer_idx")
SCENARIOS = (
    "baseline",
    "baseline_rate_low",
    "baseline_rate_high",
    "baseline_rate_wide",
    "recording_wide",
    "termination_wide",
    "false_positive_zero",
    "false_positive_double",
)


def scenario_priors(name: str) -> ModelPriors:
    """Change one prior family, retaining the recorded-count target throughout.

    These are specified stress tests, not confidence bounds from external
    studies. A/B/D are not used because they change several assumptions at once.
    """
    baseline = ModelPriors()
    if name == "baseline":
        return baseline
    if name in ("baseline_rate_low", "baseline_rate_high"):
        factor = 0.8 if name.endswith("low") else 1.2
        return replace(
            baseline, theta_lb_logit=logit(inv_logit(baseline.theta_lb_logit) * factor)
        )
    if name == "baseline_rate_wide":
        return replace(baseline, theta_lb_sigma=0.2)
    if name == "recording_wide":
        return replace(
            baseline,
            s_race_year_sigma=2 * baseline.s_race_year_sigma,
            s_edu_sigma=2 * baseline.s_edu_sigma,
        )
    if name == "termination_wide":
        return replace(
            baseline,
            **{
                field: 2 * getattr(baseline, field)
                for field in (
                    "eta_term_sigma",
                    "eta_term_race_sigma",
                    "eta_term_edu_sigma",
                    "eta_term_age_sigma",
                    "eta_term_year_sigma",
                )
            },
        )
    if name in ("false_positive_zero", "false_positive_double"):
        return replace(
            baseline,
            false_positive_rate=0.0
            if name.endswith("zero")
            else 2 * baseline.false_positive_rate,
        )
    raise ValueError(f"Unknown sensitivity scenario {name!r}")


def combine_likelihood_cells(cells: pd.DataFrame) -> pd.DataFrame:
    """Combine binomials with identical probabilities in the current full model.

    Clinical flags do not enter its probability. Summing their N and R counts
    changes the likelihood only by a parameter-independent binomial coefficient.
    Clinical-stratum checks still need the original cells and are not produced
    by this sensitivity study.
    """
    result = (
        cells.groupby(list(MODEL_COVARIATES), dropna=False, sort=True)[
            ["N_cell", "R_cell"]
        ]
        .sum()
        .reset_index()
    )
    result.attrs.update(cells.attrs)
    return result


def add_headline_quantities(model, cells):
    """Include annual expected counts in trace validation as well as summaries."""
    import pymc as pm
    import pytensor.tensor as pt

    n = cells.N_cell.to_numpy()
    year = cells.year_idx.to_numpy()
    age = cells.age_idx.to_numpy()
    with model:
        p = model["p_ds_lb"]
        theta = pm.math.invlogit(model["theta_lb_age"])[age]
        totals = pt.stack(
            [
                pt.sum(p[year == y] * n[year == y])
                for y in range(len(model.coords["year"]))
            ]
        )
        natural = pt.stack(
            [
                pt.sum(theta[year == y] * n[year == y])
                for y in range(len(model.coords["year"]))
            ]
        )
        pm.Deterministic("expected_ds_count_year", totals, dims="year")
        pm.Deterministic("expected_ds_count_total", pt.sum(totals))
        pm.Deterministic("natural_ds_count_year", natural, dims="year")
        pm.Deterministic("reduction_year", 1.0 - totals / natural, dims="year")


def annual_summary(idata, cells: pd.DataFrame, scenario: str, start_year: int):
    """Separate posterior expected counts from predictive recorded counts."""
    rows = []
    for year in sorted(cells.year_idx.unique()):
        mask = cells.year_idx.to_numpy() == year
        row = {
            "scenario": scenario,
            "year": start_year + int(year),
            "births": int(cells.loc[mask, "N_cell"].sum()),
            "recorded": int(cells.loc[mask, "R_cell"].sum()),
        }
        for variable, values in (
            (
                "expected_ds",
                idata.posterior["expected_ds_count_year"].values[..., year],
            ),
            ("natural_ds", idata.posterior["natural_ds_count_year"].values[..., year]),
            ("reduction", idata.posterior["reduction_year"].values[..., year]),
            (
                "predicted_recorded",
                idata.posterior_predictive["R_obs"].values[..., mask].sum(axis=-1),
            ),
        ):
            lo, hi = np.quantile(values, [0.055, 0.945])
            row.update(
                {
                    f"{variable}_mean": float(np.mean(values)),
                    f"{variable}_lo89": float(lo),
                    f"{variable}_hi89": float(hi),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def predictive_summary(idata, cells: pd.DataFrame, scenario: str):
    """Check recorded counts against replicated outcomes, including binomial noise."""
    replicated = (
        idata.posterior_predictive["R_obs"].transpose("chain", "draw", ...).values
    )
    rows = []
    for stratum in ("year_idx", "race_idx", "age_idx"):
        for level in sorted(cells[stratum].unique()):
            mask = cells[stratum].to_numpy() == level
            draws = replicated[..., mask].sum(axis=-1)
            lo, hi = np.quantile(draws, [0.055, 0.945])
            observed = int(cells.loc[mask, "R_cell"].sum())
            rows.append(
                {
                    "scenario": scenario,
                    "stratum": stratum,
                    "level": int(level),
                    "recorded": observed,
                    "predicted_mean": float(draws.mean()),
                    "predicted_lo89": float(lo),
                    "predicted_hi89": float(hi),
                    "observed_in_interval": bool(lo <= observed <= hi),
                }
            )
    return pd.DataFrame(rows)


def total_summary(
    idata, cells: pd.DataFrame, scenario: str, false_positive_rate: float
):
    """Summarise totals from joint draws, retaining dependence between years."""
    expected = idata.posterior["expected_ds_count_total"].values
    natural = idata.posterior["natural_ds_count_year"].values.sum(axis=-1)
    corrected = cells.R_cell.sum() - false_positive_rate * (
        cells.N_cell.sum() - expected
    )
    row = {
        "scenario": scenario,
        "births": int(cells.N_cell.sum()),
        "recorded": int(cells.R_cell.sum()),
    }
    for name, draws in (
        ("expected_ds", expected),
        ("natural_ds", natural),
        ("reduction", 1 - expected / natural),
        ("recorded_fraction_adjusted", corrected / expected),
    ):
        lo, hi = np.quantile(draws, [0.055, 0.945])
        row.update(
            {
                f"{name}_mean": float(draws.mean()),
                f"{name}_lo89": float(lo),
                f"{name}_hi89": float(hi),
            }
        )
    return pd.DataFrame([row])
