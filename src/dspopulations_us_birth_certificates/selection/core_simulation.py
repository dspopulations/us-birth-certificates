"""Joint synthetic-data generation and calibration quantities for DSP models."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from dspopulations_us_birth_certificates.selection.anomaly_panel import AnomalyPanel
from dspopulations_us_birth_certificates.selection.core_models import (
    get_core_model_definition,
)
from dspopulations_us_birth_certificates.selection.core_reduction import (
    CoreReductionPriors,
    SurveillanceAnchor,
)
from dspopulations_us_birth_certificates.selection.core_specification import (
    CoreFitSpecification,
)
from dspopulations_us_birth_certificates.selection.priors import logit
from dspopulations_us_birth_certificates.selection.sampling import sample_model_prior


def simulation_design(model_id: str) -> tuple[pd.DataFrame, CoreFitSpecification]:
    """Small fixed design with overlap, a forecast year and two recording regimes."""
    definition = get_core_model_definition(model_id)
    years = np.arange(6)
    cells = pd.DataFrame(
        {
            "year_idx": np.repeat(years, 3),
            "age_idx": np.tile(np.arange(3), 6),
            "maternal_age": np.tile([20, 30, 40], 6),
            "N_cell": np.tile([80000, 150000, 80000], 6),
            "R_cell": np.zeros(18, dtype=int),
            "revised": np.repeat((years >= 2).astype(int), 3),
        }
    )
    cells.attrs.update(n_year=6, year_range=(2016, 2021))
    if definition.age_model == "band":
        cells["age_idx"] = np.tile([1, 3, 5], 6)
        cells = cells.drop(columns="maternal_age")
    priors = CoreReductionPriors(
        reduction_mean=np.full(6, 0.35),
        reduction_logit=np.full(6, logit(0.35)),
        reduction_sigma=np.full(6, 0.25),
    )
    anchor = None
    if definition.reduction_model == "anchor":
        anchor = SurveillanceAnchor(
            np.arange(4),
            np.log(np.full(4, 0.0013)),
            half_width=1,
            mid_years=tuple(range(2016, 2020)),
            window_births=np.full((4, 3), 310000),
        )
        priors = CoreReductionPriors()
    panel = None
    if definition.recording_panel == "anomaly":
        panel = AnomalyPanel(
            condition=("control_a", "control_b", "control_c"),
            year_idx=years,
            years=tuple(range(2016, 2022)),
            flags=np.full((6, 3), 300),
            births=np.full(6, 310000),
            expected_share=np.full((6, 3), 0.001),
            true_trend_log_per_year=np.zeros(3),
            reference_year_idx=0,
        )
    return cells, CoreFitSpecification(
        definition, priors, (2016, 2021), anchor=anchor, panel=panel
    )


def simulate_core(cells, specification, *, seed: int):
    """Generate all observation channels jointly, including the penalised prior.

    The starting prevalence prior is fixed in the specification. Replacing the
    synthetic surveillance observation therefore cannot change that prior.
    """
    model = specification.build(cells)
    generated = sample_model_prior(model, draws=1, random_seed=seed)
    truth = generated.prior.isel(chain=0, draw=0).to_dataset()
    observations = generated.prior_predictive.isel(chain=0, draw=0)
    simulated = cells.copy()
    simulated["R_cell"] = observations["R_obs"].values.astype(int)
    anchor = specification.anchor
    panel = specification.panel
    if anchor is not None:
        anchor = replace(anchor, log_prevalence=observations["anchor_obs"].values)
    if panel is not None:
        panel = replace(panel, flags=observations["panel_obs"].values)
    return simulated, replace(specification, anchor=anchor, panel=panel), truth


def calibration_table(idata, truth, *, seed: int, model_id: str) -> pd.DataFrame:
    """Ranks and central 89% coverage for parameters and identifiable combinations.

    Finite correlated MCMC draws affect ranks. Inspect fit health and increase
    effective sample sizes before treating a rank histogram as a calibration
    test. A few repetitions are a regression pilot, not proof of calibration.
    """
    rows = []
    for name in (
        "expected_ds_count_total",
        "recording_s",
        "rho_year",
        "recording_s_year",
        "recording_s_panel_ratio",
        "recording_s_drift_ratio",
        "panel_loading_ds",
        "p_recorded",
    ):
        if name not in truth or name not in idata.posterior:
            continue
        variable = idata.posterior[name]
        other_dims = [dim for dim in variable.dims if dim not in ("chain", "draw")]
        draws = variable.transpose("chain", "draw", *other_dims).values
        draws = draws.reshape((-1, int(np.prod(draws.shape[2:]))))
        values = np.asarray(truth[name]).reshape(-1)
        for index, value in enumerate(values):
            samples = draws[:, index]
            lower, upper = np.quantile(samples, [0.055, 0.945])
            rank = int(np.sum(samples < value))
            rows.append(
                {
                    "model": model_id,
                    "seed": seed,
                    "variable": name,
                    "index": index,
                    "truth": value,
                    "rank": rank,
                    "draws": len(samples),
                    "rank_fraction": rank / len(samples),
                    "covered_89": bool(lower <= value <= upper),
                    "posterior_mean": float(samples.mean()),
                    "posterior_lo": lower,
                    "posterior_hi": upper,
                }
            )
    return pd.DataFrame(rows)
