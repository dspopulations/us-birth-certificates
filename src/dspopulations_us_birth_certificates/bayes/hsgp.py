"""Hilbert-space GP helper for 1-D smooth effects.

``make_hsgp_component`` wraps ``pm.gp.HSGP`` with sensible defaults for
the kind of 1-D smooth effects we model over year or age: centre/scale
the coord, weakly-informative priors on length-scale and amplitude, a
Matern / ExpQuad kernel, and a boundary factor of 1.5× the half-range.

Call from inside a ``pm.Model()`` context; the function returns both the
centred coord array (suitable for later evaluation) and the deterministic
``f`` variable registered with PyMC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class HSGPComponent:
    """The objects produced by ``make_hsgp_component``.

    Attributes:
        f: PyMC deterministic, shape ``(n_coord,)``, values on the outcome
            scale (typically logit probability).
        coord_centred: Centred/scaled coord, shape ``(n_coord,)``. Useful
            for plotting the smooth against the original coord.
        coord_mean: Mean of the raw coord used for centering.
        coord_scale: Scale used for normalisation (std dev, or 1.0 if zero).
    """

    f: Any
    coord_centred: np.ndarray
    coord_mean: float
    coord_scale: float


def make_hsgp_component(
    coord_values: np.ndarray,
    *,
    name: str,
    m: int = 12,
    c: float = 1.5,
    amplitude_prior_sigma: float = 1.0,
    lengthscale_prior_alpha: float = 2.0,
    lengthscale_prior_beta: float | None = None,
    kernel: str = "expquad",
) -> HSGPComponent:
    """Build an HSGP smooth ``f_name(coord)`` on the logit scale.

    Args:
        coord_values: 1-D array of coord values (year, age, ...). Will be
            centred on its mean and divided by its std dev before use.
        name: PyMC name prefix — the deterministic is registered as
            ``f"f_{name}"`` and the hyperparameters as
            ``ls_{name}``, ``eta_{name}``.
        m: Number of basis functions. 12 is adequate for smooth 1-D effects
            over short spans (e.g. 9 years). Bump to 20+ for wider spans.
        c: Boundary factor (L = c × max(|centred_coord|)). 1.5 is the
            lower end of the Riutort-Mayol et al. guidance for smooth
            approximations.
        amplitude_prior_sigma: Scale of the HalfNormal prior on kernel
            amplitude (on the logit scale, so 1.0 is weakly informative
            for rare-event rates).
        lengthscale_prior_alpha: Alpha of the InverseGamma length-scale
            prior (in centred/scaled coord units). Default 2.0 follows
            PyMC's recommended weakly-informative shape.
        lengthscale_prior_beta: Beta of the length-scale prior. If None,
            uses ``lengthscale_prior_alpha - 1`` so the mode is 1.0 in
            scaled units (≈ one std-dev of the coord).
        kernel: ``"expquad"`` (smooth) or ``"matern52"`` (less smooth).

    Returns:
        ``HSGPComponent`` with the deterministic and the centred coord.
    """
    import pymc as pm

    coord = np.asarray(coord_values, dtype=np.float64).ravel()
    mean = float(coord.mean())
    std = float(coord.std())
    scale = std if std > 0 else 1.0
    centred = (coord - mean) / scale

    if lengthscale_prior_beta is None:
        lengthscale_prior_beta = max(lengthscale_prior_alpha - 1.0, 1e-3)

    ls = pm.InverseGamma(
        f"ls_{name}",
        alpha=lengthscale_prior_alpha,
        beta=lengthscale_prior_beta,
    )
    eta = pm.HalfNormal(f"eta_{name}", sigma=amplitude_prior_sigma)

    if kernel == "expquad":
        cov = eta**2 * pm.gp.cov.ExpQuad(input_dim=1, ls=ls)
    elif kernel == "matern52":
        cov = eta**2 * pm.gp.cov.Matern52(input_dim=1, ls=ls)
    else:
        raise ValueError(f"Unknown kernel {kernel!r}; use 'expquad' or 'matern52'")

    gp = pm.gp.HSGP(m=[m], c=c, cov_func=cov)
    f = gp.prior(f"f_{name}", X=centred[:, None])

    return HSGPComponent(f=f, coord_centred=centred, coord_mean=mean, coord_scale=scale)
