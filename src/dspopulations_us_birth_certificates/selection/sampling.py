"""Thin wrapper around ``pm.sample`` for the selection-model pipeline.

Centralises sampler choice (``nutpie`` by default, falling back to PyMC's
default NUTS), seeding, and attachment of prior- and posterior-predictive
draws to the returned ``InferenceData``.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from dspopulations_us_birth_certificates.selection.config import RunConfig

if TYPE_CHECKING:
    import pymc as pm
    import xarray as xr


def sample(
    model: pm.Model,
    *,
    config: RunConfig,
    prior_only: bool = False,
) -> xr.DataTree:
    """Run the full Bayesian workflow: prior predictive → posterior → PPC.

    Args:
        model: A PyMC model (typically from ``selection.build_model``).
        config: Run config controlling draws/tune/chains/target_accept/seed.
        prior_only: If True, only run the prior-predictive step and return
            an InferenceData carrying ``prior`` and ``prior_predictive`` —
            useful for prior-predictive checks before a full fit.

    Returns:
        ``xr.DataTree`` with ``prior``, ``prior_predictive`` and
        (unless ``prior_only``) ``posterior`` and ``posterior_predictive``.
    """
    import pymc as pm

    with model:
        idata = sample_model_prior(
            model,
            draws=config.prior_predictive_samples,
            random_seed=config.random_seed,
        )

        if prior_only:
            return idata

        posterior = _sample_posterior(config)
        idata.update(posterior)
        idata.attrs["dsp_actual_sampler"] = posterior.attrs.get(
            "dsp_actual_sampler", "unknown"
        )

        if config.posterior_predictive:
            ppc = pm.sample_posterior_predictive(
                posterior,
                random_seed=config.random_seed,
            )
            idata.update(ppc)

    return idata


def sample_model_prior(model, *, draws: int, random_seed: int):
    """Draw the actual DSP prior, including its bounded log-weight penalty.

    Forward proposals omit Potentials. For the DSP probability barrier its log
    weight is known to be <= 0, so rejection with probability exp(log_weight)
    gives exact independent draws from the normalised penalised prior. Other
    Potentials require their own valid envelope and are rejected explicitly.
    """
    import pymc as pm
    import xarray as xr

    if draws < 1:
        raise ValueError("prior draws must be positive")
    weight_name = getattr(model, "dsp_prior_log_weight", None)
    if model.potentials and ({v.name for v in model.potentials} != {weight_name}):
        raise ValueError(
            "prior sampler does not know the rejection envelope for these Potentials"
        )
    if not model.potentials:
        result = pm.sample_prior_predictive(
            draws=draws, model=model, random_seed=random_seed
        )
        result.attrs["dsp_prior_sampling"] = "forward_exact"
        return result
    rng = np.random.default_rng(random_seed)
    accepted = []
    kept = proposed = accepted_total = 0
    while kept < draws and proposed < 100 * draws:
        size = min(max(draws - kept, 32), 100 * draws - proposed)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The effect of Potentials.*")
            batch = pm.sample_prior_predictive(
                draws=size,
                model=model,
                random_seed=int(rng.integers(2**30)),
                var_names=[v.name for v in model.basic_RVs + model.deterministics]
                + [weight_name],
            )
        log_weight = np.asarray(batch.prior[weight_name]).reshape(-1)
        if np.any(np.isnan(log_weight)) or np.any(log_weight > 0):
            raise ValueError("invalid DSP prior rejection log weight")
        select = np.flatnonzero(np.log(rng.uniform(size=size)) < log_weight)
        accepted_total += len(select)
        select = select[: draws - kept]
        if len(select):
            groups = {
                name: (
                    batch[name].to_dataset().isel(draw=select)
                    if "draw" in batch[name].dims
                    else batch[name].to_dataset()
                )
                for name in batch.groups
                if name != "/"
            }
            accepted.append(xr.DataTree.from_dict(groups))
        kept += len(select)
        proposed += size
    if kept < draws:
        raise ValueError(
            "DSP prior rejection acceptance too low; revise the prior or probability parameterisation"
        )
    first = accepted[0]
    result = xr.DataTree.from_dict(
        {
            name: (
                xr.concat(
                    [part[name].to_dataset() for part in accepted], dim="draw"
                ).assign_coords(draw=np.arange(draws))
                if "draw" in first[name].dims
                else first[name].to_dataset()
            )
            for name in first.groups
            if name != "/"
        }
    )
    result.attrs.update(
        {
            "dsp_prior_sampling": "rejection_weighted",
            "dsp_prior_proposals": proposed,
            "dsp_prior_acceptance_rate": accepted_total / proposed,
        }
    )
    return result


def _sample_posterior(config: RunConfig) -> xr.DataTree:
    """Run NUTS with the requested sampler backend."""
    import pymc as pm

    sampler = config.nuts_sampler
    try:
        result = pm.sample(
            draws=config.draws,
            tune=config.tune,
            chains=config.chains,
            target_accept=config.target_accept,
            random_seed=config.random_seed,
            nuts_sampler=sampler,
            progressbar=True,
        )
        result.attrs["dsp_actual_sampler"] = sampler
        return result
    except ImportError:
        if sampler == "nutpie":
            result = pm.sample(
                draws=config.draws,
                tune=config.tune,
                chains=config.chains,
                target_accept=config.target_accept,
                random_seed=config.random_seed,
                progressbar=True,
            )
            result.attrs["dsp_actual_sampler"] = "pymc"
            result.attrs["dsp_sampler_fallback_from"] = sampler
            return result
        raise
