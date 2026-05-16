"""Shared rendering loop for selection-model diagnostics.

Both :mod:`scripts.render_selection_diagnostics` (post-hoc rendering
from a saved fit directory) and :mod:`scripts.fit_selection_model`
(inline rendering at the end of a fit) call into :func:`render_all` to
produce the identical six-figure PNG + SVG + CSV companion set.

Layout
------
``out_dir/``
    ``plots/<stem>.png`` + ``<stem>.svg``
    ``tables/<stem>.csv`` (where a diagnostic has a tidy table companion)

Convergence summarisation is **not** done here — ``az.summary`` is
expensive on per-cell deterministics and would double the wall-clock
time if both callers recomputed it. The fit CLI computes and writes
``summary.csv`` as a separate step; the post-hoc CLI reads it.

Each diagnostic is guarded: one failure is logged via ``cli_output`` and
the rest still render.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from dspopulations_us_birth_certificates import cli_output
from dspopulations_us_birth_certificates.selection import diagnostics

if TYPE_CHECKING:
    import xarray as xr
    from matplotlib.figure import Figure


DEFAULT_STRATA: tuple[str, ...] = ("year_idx", "race_idx", "age_idx")


@dataclass(frozen=True)
class RenderOptions:
    """Options for :func:`render_all`."""

    cchd_target: float = 0.225
    hdi_prob: float = 0.94
    strata: tuple[str, ...] = DEFAULT_STRATA


def _save_figure(
    fig: Figure,
    plots_dir: Path,
    stem: str,
    *,
    data: pd.DataFrame | None = None,
    dpi: float | None = None,
) -> None:
    """Write PNG + SVG; co-save a CSV companion where one is supplied."""
    import dse_research_utils.plot.styles as plot_styles
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        plots_dir / f"{stem}.png",
        dpi=dpi if dpi is not None else plot_styles.DPI_FILE,
        bbox_inches="tight",
    )
    fig.savefig(plots_dir / f"{stem}.svg", bbox_inches="tight")
    if data is not None:
        data.to_csv(plots_dir / f"{stem}.csv", index=False)
    plt.close(fig)


def _save_table(df: pd.DataFrame, tables_dir: Path, name: str) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables_dir / name, index=False)


def _guarded(
    label: str,
    func: Callable[[], tuple[Figure, pd.DataFrame | None]],
    plots_dir: Path,
    tables_dir: Path,
    stem: str,
) -> None:
    """Execute one diagnostic; log but do not re-raise on failure."""
    try:
        fig, data = func()
    except Exception as exc:  # noqa: BLE001 — one failure mustn't sink the batch
        cli_output.warning(f"{label}: {type(exc).__name__}: {exc}")
        return
    _save_figure(fig, plots_dir, stem, data=data)
    if data is not None:
        _save_table(data, tables_dir, f"{stem}.csv")
    cli_output.success(f"{label} -> {plots_dir / stem}.png")


def render_all(
    idata: xr.DataTree,
    cells: pd.DataFrame,
    out_dir: Path,
    *,
    options: RenderOptions,
) -> None:
    """Render the six diagnostic figures + their CSV companions."""
    plots_dir = out_dir / "plots"
    tables_dir = out_dir / "tables"

    _guarded(
        "Identifiability pair-plot",
        lambda: (
            diagnostics.identifiability_pairplot(idata),
            diagnostics.identifiability_table(idata),
        ),
        plots_dir,
        tables_dir,
        "identifiability",
    )

    _guarded(
        "Termination year trajectory",
        lambda: (
            diagnostics.eta_term_year_trajectory_plot(
                idata, hdi_prob=options.hdi_prob
            ),
            diagnostics.eta_term_year_trajectory_table(
                idata, hdi_prob=options.hdi_prob
            ),
        ),
        plots_dir,
        tables_dir,
        "eta_term_year_trajectory",
    )

    _guarded(
        "CCHD consistency",
        lambda: (
            diagnostics.cchd_consistency_check(
                idata, cells, published_cchd_prevalence=options.cchd_target
            ),
            diagnostics.cchd_consistency_summary(
                idata, cells, published_cchd_prevalence=options.cchd_target
            ),
        ),
        plots_dir,
        tables_dir,
        "cchd_consistency",
    )

    _guarded(
        "Age curve check",
        lambda: (
            diagnostics.age_curve_check(idata, cells, hdi_prob=options.hdi_prob),
            diagnostics.age_curve_table(idata, hdi_prob=options.hdi_prob),
        ),
        plots_dir,
        tables_dir,
        "age_curve",
    )

    def _decomposition() -> tuple[Figure, pd.DataFrame | None]:
        fig = diagnostics.decomposition_by_race(idata, cells)
        data = getattr(fig, "_selection_data", None)
        return fig, data

    _guarded(
        "Decomposition by race",
        _decomposition,
        plots_dir,
        tables_dir,
        "decomposition_by_race",
    )

    for stratum in options.strata:
        _guarded(
            f"PPC by {stratum}",
            lambda stratum=stratum: (
                diagnostics.posterior_predictive_by_stratum(
                    idata, cells, stratum_col=stratum, hdi_prob=options.hdi_prob
                ),
                None,
            ),
            plots_dir,
            tables_dir,
            f"ppc_{stratum}",
        )


def expected_stems(strata: Iterable[str] = DEFAULT_STRATA) -> tuple[str, ...]:
    """Stem names of the figures :func:`render_all` writes (tests rely on this)."""
    base = (
        "identifiability",
        "eta_term_year_trajectory",
        "cchd_consistency",
        "age_curve",
        "decomposition_by_race",
    )
    return base + tuple(f"ppc_{s}" for s in strata)


__all__ = [
    "DEFAULT_STRATA",
    "RenderOptions",
    "expected_stems",
    "render_all",
]
