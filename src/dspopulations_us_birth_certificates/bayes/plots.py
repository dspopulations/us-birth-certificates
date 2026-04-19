"""Posterior plots for Bayesian cell models.

Uses shared plot style constants from ``dse_research_utils.plot.styles``
so figures match the conventions of sibling DSE research repos.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import dse_research_utils.plot.styles as plot_styles
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import arviz as az


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=plot_styles.DPI_FILE, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")


def plot_trend_by_dim(
    idata: az.InferenceData,
    cells: pd.DataFrame,
    *,
    dim: str,
    output_path: Path,
    hdi_prob: float = 0.94,
) -> None:
    """Posterior mean cell rate (y_cell / n_cell expected) aggregated over ``dim``.

    For each unique value of ``dim``, plot the posterior HDI of the
    cell-weighted expected rate alongside the observed ratio.
    """
    import arviz as az
    import matplotlib.pyplot as plt

    p = idata.posterior["p"]
    n_cell = cells["n_cell"].to_numpy(dtype=np.float64)
    y_cell = cells["y_cell"].to_numpy(dtype=np.float64)

    dim_values = cells[dim].to_numpy()
    unique = np.sort(np.unique(dim_values))

    # For each unique dim value, pool cells: rate = sum(p_i * n_i) / sum(n_i)
    expected_rate = np.zeros((p.sizes["chain"], p.sizes["draw"], len(unique)))
    observed_rate = np.zeros(len(unique))
    for i, v in enumerate(unique):
        mask = dim_values == v
        n = n_cell[mask]
        n_sum = n.sum()
        expected_rate[..., i] = (p.values[..., mask] * n[None, None, :]).sum(
            axis=-1
        ) / n_sum
        observed_rate[i] = y_cell[mask].sum() / n_sum

    mean = expected_rate.mean(axis=(0, 1))
    hdi = az.hdi(expected_rate, hdi_prob=hdi_prob)
    lo = hdi[..., 0]
    hi = hdi[..., 1]

    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_MD)
    ax.fill_between(
        unique,
        lo,
        hi,
        alpha=0.3,
        color=plot_styles.COLOUR_BLUE,
        label=f"{int(hdi_prob * 100)}% HDI",
    )
    ax.plot(unique, mean, lw=2, color=plot_styles.COLOUR_BLUE, label="Posterior mean")
    ax.plot(
        unique,
        observed_rate,
        "o",
        color=plot_styles.TEXT_COLOUR,
        label="Observed",
        markersize=4,
    )
    ax.set_xlabel(dim)
    ax.set_ylabel("DS rate per birth")
    ax.set_title(f"Posterior cell-weighted rate by {dim}")
    ax.legend()
    _save(fig, output_path)
    plt.close(fig)


def plot_ppc(
    idata: az.InferenceData,
    *,
    output_path: Path,
) -> None:
    """Posterior predictive check plot (observed vs replicated)."""
    import arviz as az
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_MD)
    az.plot_ppc(idata, num_pp_samples=100, ax=ax)
    _save(fig, output_path)
    plt.close(fig)


def plot_trace(
    idata: az.InferenceData,
    *,
    var_names: tuple[str, ...],
    output_path: Path,
) -> None:
    """Trace plot for a small set of named variables."""
    import arviz as az
    import matplotlib.pyplot as plt

    axes = az.plot_trace(idata, var_names=list(var_names))
    fig = axes.ravel()[0].figure
    width, _ = plot_styles.FIGSIZE_LG
    fig.set_size_inches(width, max(2, 1.5 * len(var_names)))
    _save(fig, output_path)
    plt.close(fig)
