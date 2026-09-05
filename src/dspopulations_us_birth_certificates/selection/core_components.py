"""PyMC components for prevalence, age reduction and certificate recording.

Each function builds one named part of the DSP graph. Probability accounting
and the observed DS likelihood remain in core_reduction.py.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from dspopulations_us_birth_certificates.selection.core_math import (
    calibrated_age_intercept,
    window_design,
    window_error_correlation,
)
from dspopulations_us_birth_certificates.selection.priors import logit


def build_anchor_prevalence(
    anchor,
    *,
    n_year,
    n_latent,
    natural_prevalence_year,
    anchor_level_sigma,
    anchor_trend_sigma,
    anchor_obs_sigma,
    anchor_obs_sigma_fixed,
    anchor_forecast_flat,
    anchor_overlap_share,
    anchor_level_prior_prevalence,
):
    """Build the annual process and joint, weighted window observation model."""
    assert anchor is not None
    pm.Data("natural_prevalence_year", natural_prevalence_year, dims="year")
    half = anchor.half_width

    # Local linear trend on log prevalence, non-centred. The level
    # innovation absorbs year-to-year variation; the slope innovation lets
    # the trend itself drift, which is what makes the forecast interval
    # widen rather than extrapolate one fixed gradient forever.
    # The fixed prior median is separate from the observations. Recentring it
    # on each observed window would reuse data and change the prior in a joint
    # simulation experiment. The probability penalty handles extreme tails.
    level_start = pm.Normal(
        "anchor_log_level_start",
        mu=float(np.log(anchor_level_prior_prevalence)),
        sigma=0.25,
    )
    slope_start = pm.Normal(
        "anchor_log_slope_start", mu=0.0, sigma=anchor_trend_sigma * 5.0
    )
    sigma_level = pm.HalfNormal("anchor_level_sigma", sigma=anchor_level_sigma)
    sigma_trend = pm.HalfNormal("anchor_trend_sigma", sigma=anchor_trend_sigma)
    level_innovation = pm.Normal(
        "anchor_level_innovation_raw", mu=0.0, sigma=1.0, shape=n_latent - 1
    )
    trend_innovation = pm.Normal(
        "anchor_trend_innovation_raw", mu=0.0, sigma=1.0, shape=n_latent - 1
    )
    slope = slope_start + pt.concatenate(
        [pt.zeros((1,)), pt.cumsum(trend_innovation * sigma_trend)]
    )
    latent_increment = slope[:-1] + level_innovation * sigma_level
    if anchor_forecast_flat:
        # The corner opposite a drifting s: hold latent prevalence at its
        # last anchored value so the whole post-window decline in the
        # recorded rate has to be absorbed by recording. Increment j
        # feeds latent year j + 1, so zeroing every increment from the
        # last anchored latent index onward leaves the path flat from
        # that year. Latent index i is model year i - half_width, hence
        # the 2 * half offset.
        last_anchored_latent_idx = int(anchor.mid_year_idx.max()) + 2 * half
        latent_increment = latent_increment * pt.as_tensor_variable(
            (np.arange(n_latent - 1) < last_anchored_latent_idx).astype(float)
        )
    log_prevalence_latent = pm.Deterministic(
        "anchor_log_prevalence_latent",
        level_start + pt.concatenate([pt.zeros((1,)), pt.cumsum(latent_increment)]),
        dims="latent_year",
    )
    prevalence_latent = pt.exp(log_prevalence_latent)

    design = window_design(anchor.mid_year_idx, half, anchor.window_births)
    correlation = window_error_correlation(design, anchor_overlap_share)
    pm.Data("anchor_window_error_correlation", correlation)
    window_means = pt.dot(design, prevalence_latent[: design.shape[1]])
    pm.Deterministic("anchor_window_prevalence", window_means, dims="anchor_window")
    # Estimating the observation SD measures only whether the windows are
    # mutually consistent with a smooth path -- it cannot measure whether
    # the surveillance programmes' prevalences are *accurate*, because the
    # workbook supplies no uncertainty. Compare fixed observation scales
    # to assess sensitivity to assumed source accuracy.
    if anchor_obs_sigma_fixed is None:
        sigma_obs: Any = pm.HalfNormal("anchor_obs_sigma", sigma=anchor_obs_sigma)
    else:
        sigma_obs = anchor_obs_sigma_fixed
        pm.Deterministic(
            "anchor_obs_sigma", pt.as_tensor_variable(anchor_obs_sigma_fixed)
        )
    pm.MvNormal(
        "anchor_obs",
        mu=pt.log(window_means),
        chol=sigma_obs * np.linalg.cholesky(correlation),
        observed=anchor.log_prevalence,
        dims="anchor_window",
    )

    prevalence_year = pm.Deterministic(
        "prevalence_year",
        prevalence_latent[half : half + n_year],
        dims="year",
    )
    # The accounting identity: reduction is whatever reconciles an
    # anchored prevalence with the Morris no-reduction expectation.
    eta_year = pm.Deterministic(
        "eta_year",
        prevalence_year / natural_prevalence_year,
        dims="year",
    )
    pm.Deterministic("rho_year", 1.0 - eta_year, dims="year")
    return eta_year


def build_age_reduction(rho_logit, natural_weight, age_step_scale, priors):
    """Build a smooth age curve that preserves the sampled national margin."""
    rho_year_anchor = pm.Deterministic(
        "rho_year_anchor", pm.math.invlogit(rho_logit), dims="year"
    )
    rho_age_step = pm.Normal(
        "rho_age_step",
        mu=0.0,
        sigma=priors.reduction_age_step_sigma * age_step_scale,
        dims="age_step",
    )
    rho_age_offset_uncentred = pt.concatenate(
        [pt.zeros((1,), dtype=rho_age_step.dtype), pt.cumsum(rho_age_step)]
    )
    rho_age_offset = pm.Deterministic(
        "rho_age_offset",
        rho_age_offset_uncentred - rho_age_offset_uncentred.mean(),
        dims="age",
    )

    # Calibrate a separate intercept for each year so the smooth age
    # curve preserves the sampled surveillance-informed national
    # reduction margin. The weights are expected DS births absent
    # prenatal reduction: N(year, age) * Morris(age).
    rho_year_intercept = calibrated_age_intercept(
        rho_logit, rho_age_offset, natural_weight
    )
    rho_year_intercept = pm.Deterministic(
        "rho_year_intercept", rho_year_intercept, dims="year"
    )
    rho_year_age = pm.Deterministic(
        "rho_year_age",
        pm.math.invlogit(rho_year_intercept[:, None] + rho_age_offset[None, :]),
        dims=("year", "age"),
    )
    rho_year = pm.Deterministic(
        "rho_year",
        (natural_weight * rho_year_age).sum(axis=1),
        dims="year",
    )
    pm.Deterministic(
        "rho_year_margin_error",
        rho_year - rho_year_anchor,
        dims="year",
    )
    eta_year = pm.Deterministic("eta_year", 1.0 - rho_year, dims="year")
    pm.Deterministic("eta_year_age", 1.0 - rho_year_age, dims=("year", "age"))

    return rho_year_age, eta_year


def build_panel_offset(
    panel,
    *,
    n_year,
    panel_factor_sigma,
    panel_prevalence_trend_sigma,
    panel_condition_trend_sigma,
    panel_idiosyncratic_sigma,
    panel_loading_sigma,
    panel_loading_fixed,
):
    """Build control likelihoods and the DS sensitivity log-odds offset."""
    assert panel is not None
    n_panel_year = panel.n_year
    years_since_reference = panel.years_since_reference
    reference_idx = panel.reference_year_idx
    reference_share = panel.expected_share[reference_idx]

    # Maternal-age composition, centred on the reference year, so the
    # level parameter is a reference-year level and this offset carries
    # only composition *change*.
    log_composition = np.log(panel.expected_share / reference_share[np.newaxis, :])
    # Any externally known true-prevalence trend, entered as a fixed
    # offset rather than estimated. Zero for every curated control today:
    # pinning these against published surveillance is the open input.
    known_log_trend = (
        years_since_reference[:, np.newaxis]
        * panel.true_trend_log_per_year[np.newaxis, :]
    )

    condition_level = pm.Normal(
        "panel_condition_logit_level",
        mu=logit(reference_share),
        sigma=1.0,
        dims="panel_condition",
    )

    # The common log-rate change across the controls. This is the part the
    # panel *measures*: what every control did together, relative to the
    # reference year. Modelled as a random walk so it interpolates smoothly
    # rather than fitting one free effect per year.
    #
    # A fixed innovation scale still shrinks changes towards zero.
    # Fixing it prevents adaptive shrinkage from competing with the
    # estimated condition-trend scale. Both scales remain assumptions
    # about the allocation of common and condition-specific movement.
    common_innovation = pm.Normal(
        "panel_common_innovation_raw",
        mu=0.0,
        sigma=1.0,
        shape=n_panel_year - 1,
    )
    common_walk = pt.concatenate(
        [
            pt.zeros((1,)),
            pt.cumsum(common_innovation * panel_factor_sigma),
        ]
    )
    common_log_change = pm.Deterministic(
        "panel_common_log_change",
        common_walk - common_walk[reference_idx],
        dims="panel_year",
    )

    # The one thing the panel cannot see: a true-prevalence trend shared
    # by every control. It is perfectly confounded with a common recording
    # trend, so it is carried as a prior and *subtracted* rather than being
    # allowed to compete in the control likelihood. This parameter can
    # still be updated by the joint DS likelihood and anchor. Separating
    # the terms does not establish identification in the joint model.
    if panel_prevalence_trend_sigma > 0.0:
        prevalence_trend: Any = pm.Normal(
            "panel_prevalence_trend",
            mu=0.0,
            sigma=panel_prevalence_trend_sigma,
        )
    else:
        prevalence_trend = pt.zeros(())
        pm.Deterministic("panel_prevalence_trend", prevalence_trend)

    # What is left is the shared *recording* factor, which is the whole
    # point of the panel.
    panel_recording_log_factor = pm.Deterministic(
        "panel_recording_log_factor",
        common_log_change - prevalence_trend * years_since_reference,
        dims="panel_year",
    )
    pm.Deterministic(
        "panel_recording_factor_ratio",
        pt.exp(panel_recording_log_factor[-1]),
    )
    pm.Deterministic("panel_common_change_ratio", pt.exp(common_log_change[-1]))

    # Uncentred condition trends allow their mean to differ from zero. That
    # mean remains confounded with the common trajectory, so their priors affect
    # its interpretation and uncertainty. A sum-to-zero alternative would define
    # a different common estimand. Historical comparisons are in the DSP010 note.
    condition_trend_scale = pm.HalfNormal(
        "panel_condition_trend_scale", sigma=panel_condition_trend_sigma
    )
    condition_trend_raw = pm.Normal(
        "panel_condition_trend_raw",
        mu=0.0,
        sigma=1.0,
        dims="panel_condition",
    )
    condition_trend = pm.Deterministic(
        "panel_condition_trend",
        condition_trend_raw * condition_trend_scale,
        dims="panel_condition",
    )
    # The quantity the common trend is confounded with. Reporting it makes
    # the confounding legible instead of buried in the parameterisation:
    # if it sits far from zero, the control set is lop-sided and the
    # "common" factor is carrying a shared quirk.
    pm.Deterministic("panel_condition_trend_mean", condition_trend.mean())

    panel_idiosyncratic_scale = pm.HalfNormal(
        "panel_idiosyncratic_scale", sigma=panel_idiosyncratic_sigma
    )
    idiosyncratic_raw = pm.Normal(
        "panel_idiosyncratic_raw",
        mu=0.0,
        sigma=1.0,
        dims=("panel_year", "panel_condition"),
    )

    # Rates here are of order 5e-4, so a logit-scale offset is a
    # multiplicative change in the rate to within the rate itself.
    panel_logit = (
        condition_level[np.newaxis, :]
        + log_composition
        + known_log_trend
        + common_log_change[:, np.newaxis]
        + condition_trend[np.newaxis, :] * years_since_reference[:, np.newaxis]
        + idiosyncratic_raw * panel_idiosyncratic_scale
    )
    panel_rate = pm.Deterministic(
        "panel_condition_rate",
        pm.math.invlogit(panel_logit),
        dims=("panel_year", "panel_condition"),
    )
    pm.Binomial(
        "panel_obs",
        n=np.repeat(
            panel.births[:, np.newaxis].astype("int64"),
            panel.n_condition,
            axis=1,
        ),
        p=panel_rate,
        observed=panel.flags.astype("int64"),
        dims=("panel_year", "panel_condition"),
    )

    # Down syndrome's loading on the item-wide factor. A value of 1 shares
    # log-odds changes, not proportional sensitivity changes. Years with a
    # surveillance window and the panel are what make it estimable.
    if panel_loading_fixed is None:
        loading: Any = pm.Normal("panel_loading_ds", mu=1.0, sigma=panel_loading_sigma)
    else:
        loading = pt.as_tensor_variable(float(panel_loading_fixed))
        pm.Deterministic("panel_loading_ds", loading)

    # Scatter the panel years back onto the model's year coordinate. A
    # fixed 0/1 selection matrix rather than ``set_subtensor``: it is the
    # same idiom as ``year_by_cell_n``, and it avoids a pytensor rewrite
    # that fails noisily on every compile when a scalar is added to a
    # sparse write. Years before the panel starts stay at zero, so
    # ``recording_s`` keeps its meaning as the reference-year revised level
    # and stays comparable across the family.
    panel_year_selector = np.zeros((n_year, n_panel_year))
    panel_year_selector[panel.year_idx, np.arange(n_panel_year)] = 1.0
    s_panel_logit = pm.Deterministic(
        "recording_s_panel_logit",
        loading * pt.dot(panel_year_selector, panel_recording_log_factor),
        dims="year",
    )

    return s_panel_logit


def build_recording(
    priors,
    *,
    n_year,
    n_drift_year,
    recording_model,
    recording_drift,
    recording_panel,
    panel,
    panel_factor_sigma,
    panel_prevalence_trend_sigma,
    panel_condition_trend_sigma,
    panel_idiosyncratic_sigma,
    panel_loading_sigma,
    panel_loading_fixed,
    year_idx,
    revised_cell,
):
    """Build the recording process and evaluate sensitivity in each cell."""
    s_logit = pm.Normal(
        "recording_s_logit",
        mu=priors.recording_s_logit,
        sigma=priors.recording_s_sigma,
    )
    recording_s = pm.Deterministic("recording_s", pm.math.invlogit(s_logit))

    # ``recording_s`` stays the anchored-era level in every model, drifted or
    # not, so it remains directly comparable across the family. The drift is
    # carried as a separate per-year logit offset that is exactly zero while
    # the anchor still speaks.
    s_drift_logit: Any = None
    if recording_drift == "post_anchor" and priors.recording_s_drift_sigma > 0.0:
        drift_innovation = pm.Normal(
            "recording_s_drift_innovation_raw",
            mu=0.0,
            sigma=1.0,
            shape=n_drift_year,
        )
        s_drift_logit = pm.Deterministic(
            "recording_s_drift_logit",
            pt.concatenate(
                [
                    pt.zeros((n_year - n_drift_year,)),
                    pt.cumsum(drift_innovation * priors.recording_s_drift_sigma),
                ]
            ),
            dims="year",
        )
    # The anomaly panel: a second observation channel on conditions that share
    # the Down syndrome certificate item but have no reduction channel, so
    # their common movement reads as the item's recording sensitivity.
    s_panel_logit: Any = None
    if recording_panel == "anomaly":
        s_panel_logit = build_panel_offset(
            panel,
            n_year=n_year,
            panel_factor_sigma=panel_factor_sigma,
            panel_prevalence_trend_sigma=panel_prevalence_trend_sigma,
            panel_condition_trend_sigma=panel_condition_trend_sigma,
            panel_idiosyncratic_sigma=panel_idiosyncratic_sigma,
            panel_loading_sigma=panel_loading_sigma,
            panel_loading_fixed=panel_loading_fixed,
        )

    s_year_logit_offset = None
    for offset in (s_drift_logit, s_panel_logit):
        if offset is not None:
            s_year_logit_offset = (
                offset if s_year_logit_offset is None else s_year_logit_offset + offset
            )
    s_logit_year = (
        s_logit if s_year_logit_offset is None else s_logit + s_year_logit_offset
    )
    # Keep the undrifted, unpanelled graph byte-identical to what it was before
    # either existed, so a zero drift sigma reproduces the parent model exactly.
    recording_s_year_value = (
        pt.ones((n_year,)) * recording_s
        if s_year_logit_offset is None
        else pm.math.invlogit(s_logit_year)
    )

    if recording_model == "revision":
        # ``recording_s`` is the revised-certificate sensitivity, so it stays
        # directly comparable with fits confined to 2016 onward where every
        # record is revised. The unrevised sensitivity is a logit offset from
        # it, identified by years in which both certificate versions are in
        # use. Sum-to-zero centring would be wrong here: the two levels are
        # distinguishable measurement regimes, not exchangeable groups.
        s_unrevised_offset = pm.Normal(
            "recording_s_unrevised_offset",
            mu=0.0,
            sigma=priors.recording_s_sigma,
        )
        pm.Deterministic(
            "recording_s_unrevised",
            pm.math.invlogit(s_logit + s_unrevised_offset),
        )
        recording_s_year = pm.Deterministic(
            "recording_s_year",
            recording_s_year_value,
            dims="year",
        )
    elif recording_model == "constant":
        recording_s_year = pm.Deterministic(
            "recording_s_year",
            recording_s_year_value,
            dims="year",
        )
    else:
        s_year_offset_raw = pm.Normal(
            "recording_s_year_offset_raw",
            mu=0.0,
            sigma=priors.recording_s_year_sigma,
            dims="year",
        )
        s_year_offset = pm.Deterministic(
            "recording_s_year_offset",
            s_year_offset_raw - s_year_offset_raw.mean(),
            dims="year",
        )
        recording_s_year = pm.Deterministic(
            "recording_s_year",
            pm.math.invlogit(s_logit + s_year_offset),
            dims="year",
        )

    if s_drift_logit is not None:
        # The headline the drift exists to report: how far the final modelled
        # year's revised sensitivity has moved from its anchored-era level.
        # A value below 1 means recording has taken part of the recorded-rate
        # decline that an undrifted fit books entirely as falling prevalence.
        pm.Deterministic("recording_s_drift_ratio", recording_s_year[-1] / recording_s)
    if s_panel_logit is not None:
        # The panel's counterpart of that headline, on the same scale, so the
        # prior-driven and panel-driven allocations can be read side by side.
        pm.Deterministic("recording_s_panel_ratio", recording_s_year[-1] / recording_s)

    if recording_model == "revision":
        # The drift and the panel factor both shift the two certificate
        # versions together: they model the certificate's recording behaviour
        # over time, not a change in the gap between the versions. Unrevised
        # records only exist before 2016, so in practice both terms are zero
        # wherever ``revised_cell`` is 0 -- the drift because those years are
        # anchored, the panel because it starts at 2016 by construction.
        s_cell = pm.math.invlogit(
            (s_logit if s_year_logit_offset is None else s_logit_year[year_idx])
            + s_unrevised_offset * (1.0 - revised_cell)
        )
    else:
        s_cell = recording_s_year[year_idx]
    return recording_s_year, s_cell
