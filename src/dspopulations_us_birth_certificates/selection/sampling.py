"""Thin wrapper around ``pm.sample`` for the selection-model pipeline.

Centralises sampler choice (``nutpie`` by default, falling back to PyMC's
default NUTS), seeding, and attachment of prior- and posterior-predictive
draws to the returned ``InferenceData``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dspopulations_us_birth_certificates.selection.config import RunConfig

if TYPE_CHECKING:
    import arviz as az
    import pymc as pm


def sample(
    model: pm.Model,
    *,
    config: RunConfig,
    prior_only: bool = False,
) -> az.InferenceData:
    """Run the full Bayesian workflow: prior predictive → posterior → PPC.

    Args:
        model: A PyMC model (typically from ``selection.build_model``).
        config: Run config controlling draws/tune/chains/target_accept/seed.
        prior_only: If True, only run the prior-predictive step and return
            an InferenceData carrying ``prior`` and ``prior_predictive`` —
            useful for prior-predictive checks before a full fit.

    Returns:
        ``az.InferenceData`` with ``prior``, ``prior_predictive`` and
        (unless ``prior_only``) ``posterior`` and ``posterior_predictive``.
    """
    import pymc as pm

    with model:
        idata = pm.sample_prior_predictive(
            draws=config.prior_predictive_samples,
            random_seed=config.random_seed,
        )

        if prior_only:
            return idata

        posterior = _sample_posterior(config)
        idata.extend(posterior)

        if config.posterior_predictive:
            ppc = pm.sample_posterior_predictive(
                posterior,
                random_seed=config.random_seed,
            )
            idata.extend(ppc)

    return idata


def _sample_posterior(config: RunConfig) -> az.InferenceData:
    """Run NUTS with the requested sampler backend."""
    import pymc as pm

    sampler = config.nuts_sampler
    try:
        return pm.sample(
            draws=config.draws,
            tune=config.tune,
            chains=config.chains,
            target_accept=config.target_accept,
            random_seed=config.random_seed,
            nuts_sampler=sampler,
            progressbar=True,
        )
    except ImportError:
        if sampler == "nutpie":
            return pm.sample(
                draws=config.draws,
                tune=config.tune,
                chains=config.chains,
                target_accept=config.target_accept,
                random_seed=config.random_seed,
                progressbar=True,
            )
        raise
