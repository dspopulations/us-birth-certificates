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


def _save(fig, path: Path, *, data: pd.DataFrame | None = None) -> None:
    """Save a figure as PNG + SVG and (optionally) the underlying data as CSV.

    The CSV is written to ``path.with_suffix('.csv')`` and carries the
    arrays actually rendered on the plot. Keeping plot image and plot data
    at the same stem makes it trivial for readers to re-plot or validate
    the figure in a notebook.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=plot_styles.DPI_FILE, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    if data is not None:
        data.to_csv(path.with_suffix(".csv"), index=False)


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

    data = pd.DataFrame(
        {
            dim: unique,
            "posterior_mean": mean,
            "hdi_lo": lo,
            "hdi_hi": hi,
            "observed_rate": observed_rate,
            "hdi_prob": hdi_prob,
        }
    )
    _save(fig, output_path, data=data)
    plt.close(fig)


def plot_ppc(
    idata: az.InferenceData,
    *,
    output_path: Path,
    hdi_prob: float = 0.94,
) -> None:
    """Posterior predictive check plot (observed vs replicated).

    The CSV companion carries, for each cell, the observed ``y_obs`` value
    plus the posterior-predictive mean and HDI — a compact summary of the
    density comparison the plot shows.
    """
    import arviz as az
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_MD)
    az.plot_ppc(idata, num_pp_samples=100, ax=ax)

    data: pd.DataFrame | None = None
    if "posterior_predictive" in idata.groups() and "observed_data" in idata.groups():
        pp = idata.posterior_predictive  # type: ignore[attr-defined]
        obs = idata.observed_data  # type: ignore[attr-defined]
        if "y_obs" in pp.data_vars and "y_obs" in obs.data_vars:
            pp_vals = np.asarray(pp["y_obs"].values)  # (chain, draw, cell)
            observed = np.asarray(obs["y_obs"].values)  # (cell,)
            pp_mean = pp_vals.mean(axis=(0, 1))
            hdi = az.hdi(pp_vals, hdi_prob=hdi_prob)
            hdi_vals = np.asarray(hdi.values if hasattr(hdi, "values") else hdi)
            if hdi_vals.ndim == 3:
                hdi_vals = hdi_vals.reshape(-1, 2)
            data = pd.DataFrame(
                {
                    "cell_index": np.arange(len(observed)),
                    "observed": observed,
                    "pp_mean": pp_mean,
                    "pp_hdi_lo": hdi_vals[..., 0],
                    "pp_hdi_hi": hdi_vals[..., 1],
                    "hdi_prob": hdi_prob,
                }
            )

    _save(fig, output_path, data=data)
    plt.close(fig)


def plot_trace(
    idata: az.InferenceData,
    *,
    var_names: tuple[str, ...],
    output_path: Path,
) -> None:
    """Trace plot for a small set of named variables.

    The CSV companion is long-format ``(variable, chain, draw, value)`` for
    each scalar variable in ``var_names`` — enough to reconstruct the
    rendered trace lines in a notebook.
    """
    import arviz as az
    import matplotlib.pyplot as plt

    axes = az.plot_trace(idata, var_names=list(var_names))
    fig = axes.ravel()[0].figure
    width, _ = plot_styles.FIGSIZE_LG
    fig.set_size_inches(width, max(2, 1.5 * len(var_names)))

    frames: list[pd.DataFrame] = []
    posterior = idata.posterior  # type: ignore[attr-defined]
    for name in var_names:
        if name not in posterior.data_vars:
            continue
        arr = np.asarray(posterior[name].values)
        if arr.ndim < 2:
            continue
        # For scalar RVs arr.shape == (chain, draw); higher-dim RVs get one
        # row per trailing index so the CSV stays tidy.
        n_chain, n_draw = arr.shape[:2]
        trailing = int(np.prod(arr.shape[2:])) if arr.ndim > 2 else 1
        flat = arr.reshape(n_chain, n_draw, trailing)
        for k in range(trailing):
            chain_idx, draw_idx = np.meshgrid(
                np.arange(n_chain), np.arange(n_draw), indexing="ij"
            )
            frames.append(
                pd.DataFrame(
                    {
                        "variable": name if trailing == 1 else f"{name}[{k}]",
                        "chain": chain_idx.ravel(),
                        "draw": draw_idx.ravel(),
                        "value": flat[..., k].ravel(),
                    }
                )
            )
    data = pd.concat(frames, ignore_index=True) if frames else None
    _save(fig, output_path, data=data)
    plt.close(fig)


def plot_prior_draws(
    idata: az.InferenceData,
    cells: pd.DataFrame,
    *,
    coord_name: str,
    smooth_name: str,
    output_path: Path,
    n_draws: int = 50,
    hdi_prob: float = 0.94,
    seed: int = 0,
) -> None:
    """Prior draws of an HSGP smooth on its original coord axis.

    Plots ``n_draws`` random prior samples of ``smooth_name`` (e.g. ``f_t``)
    as thin lines, with a ``hdi_prob`` HDI band, against the unique values of
    ``coord_name`` taken from ``cells``. Useful for eyeballing whether the
    prior's smoothness and amplitude are plausible before fitting.

    Writes ``{output_path}.png`` and ``{output_path}.svg``. Returns silently
    if the smooth is absent from the ``prior`` group.
    """
    import arviz as az
    import matplotlib.pyplot as plt

    prior = idata.prior  # type: ignore[attr-defined]
    if smooth_name not in prior.data_vars:
        return

    f = np.asarray(prior[smooth_name].values)
    if f.ndim < 3:
        return
    n_chain, n_draw, _ = f.shape

    coord = cells[coord_name].to_numpy()
    unique, inv = np.unique(coord, return_inverse=True)
    first_idx = np.array(
        [int(np.flatnonzero(inv == k)[0]) for k in range(len(unique))],
        dtype=np.int64,
    )
    f_unique = f[..., first_idx]  # (chain, draw, n_unique)

    rng = np.random.default_rng(seed)
    flat = f_unique.reshape(-1, len(unique))
    n_samples = flat.shape[0]
    sel = rng.choice(n_samples, size=min(n_draws, n_samples), replace=False)
    picked = flat[sel]

    hdi = az.hdi(f_unique, hdi_prob=hdi_prob)
    # az.hdi returns either a DataArray or ndarray depending on input;
    # .values on a DataArray gives (n_unique, 2); ndarray is already shaped.
    hdi_vals = np.asarray(hdi.values if hasattr(hdi, "values") else hdi)
    if hdi_vals.ndim == 3:  # defensive: some arviz versions return (1, n, 2)
        hdi_vals = hdi_vals.reshape(-1, 2)
    lo = hdi_vals[..., 0]
    hi = hdi_vals[..., 1]

    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_MD)
    for row in picked:
        ax.plot(
            unique,
            row,
            lw=0.6,
            alpha=0.35,
            color=plot_styles.COLOUR_BLUE,
        )
    ax.fill_between(
        unique,
        lo,
        hi,
        alpha=0.2,
        color=plot_styles.COLOUR_ORANGE,
        label=f"{int(hdi_prob * 100)}% HDI",
    )
    ax.axhline(0.0, color=plot_styles.TEXT_COLOUR, lw=0.5, alpha=0.5)
    ax.set_xlabel(coord_name)
    ax.set_ylabel(f"{smooth_name} (logit scale)")
    ax.set_title(
        f"Prior draws of {smooth_name}({coord_name}) — {len(picked)} samples"
    )
    ax.legend()

    # Long-form CSV: one row per (coord, series) where series is either
    # "hdi_lo" / "hdi_hi" / "median" or "draw_{k}". Recovers the plot exactly.
    median = np.median(flat, axis=0)
    records: list[dict[str, object]] = []
    for i, cv in enumerate(unique):
        records.append({"coord": cv, "series": "hdi_lo", "value": float(lo[i])})
        records.append({"coord": cv, "series": "hdi_hi", "value": float(hi[i])})
        records.append({"coord": cv, "series": "median", "value": float(median[i])})
        for k, draw_vals in enumerate(picked):
            records.append(
                {"coord": cv, "series": f"draw_{k}", "value": float(draw_vals[i])}
            )
    data = pd.DataFrame(records).rename(columns={"coord": coord_name})
    data["hdi_prob"] = hdi_prob

    _save(fig, output_path, data=data)
    plt.close(fig)


def plot_prior_posterior_overlay(
    idata: az.InferenceData,
    *,
    var_names: tuple[str, ...],
    output_path: Path,
) -> None:
    """Overlay prior and posterior marginal densities for named scalars.

    Skips variables missing from either group. Returns silently if no
    overlay can be drawn (e.g. prior-only run or empty ``var_names``).
    """
    import arviz as az
    import matplotlib.pyplot as plt

    groups = idata.groups()
    if "prior" not in groups or "posterior" not in groups:
        return

    prior = idata.prior  # type: ignore[attr-defined]
    posterior = idata.posterior  # type: ignore[attr-defined]
    available = [
        name
        for name in var_names
        if name in prior.data_vars and name in posterior.data_vars
    ]
    if not available:
        return

    n = len(available)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 3.5, nrows * 2.5),
        squeeze=False,
    )

    for ax, name in zip(axes.flat, available, strict=False):
        prior_samples = np.asarray(prior[name].values).reshape(-1)
        post_samples = np.asarray(posterior[name].values).reshape(-1)
        try:
            az.plot_kde(
                prior_samples,
                ax=ax,
                label="prior",
                plot_kwargs={
                    "color": plot_styles.COLOUR_ORANGE,
                    "linestyle": "--",
                },
            )
            az.plot_kde(
                post_samples,
                ax=ax,
                label="posterior",
                plot_kwargs={"color": plot_styles.COLOUR_BLUE},
            )
        except Exception:  # noqa: BLE001 — fall back to histograms
            ax.hist(
                prior_samples,
                bins=30,
                density=True,
                alpha=0.4,
                color=plot_styles.COLOUR_ORANGE,
                label="prior",
            )
            ax.hist(
                post_samples,
                bins=30,
                density=True,
                alpha=0.4,
                color=plot_styles.COLOUR_BLUE,
                label="posterior",
            )
        ax.set_title(name)
        ax.legend(fontsize="small")

    for ax in axes.flat[n:]:
        ax.set_visible(False)

    fig.tight_layout()

    frames: list[pd.DataFrame] = []
    for name in available:
        prior_samples = np.asarray(prior[name].values).reshape(-1)
        post_samples = np.asarray(posterior[name].values).reshape(-1)
        frames.append(
            pd.DataFrame(
                {
                    "variable": name,
                    "group": "prior",
                    "sample_index": np.arange(len(prior_samples)),
                    "value": prior_samples,
                }
            )
        )
        frames.append(
            pd.DataFrame(
                {
                    "variable": name,
                    "group": "posterior",
                    "sample_index": np.arange(len(post_samples)),
                    "value": post_samples,
                }
            )
        )
    data = pd.concat(frames, ignore_index=True) if frames else None
    _save(fig, output_path, data=data)
    plt.close(fig)
