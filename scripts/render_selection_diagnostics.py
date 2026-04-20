"""Render posterior diagnostics for a fitted selection-model run.

Given a fit directory (containing ``idata.nc`` + ``cells.parquet``) or
explicit ``--idata`` / ``--cells`` paths, this script walks the six
diagnostics in :mod:`dspopulations_us_birth_certificates.selection.diagnostics`,
saving each as PNG + SVG + (where applicable) a CSV companion carrying
the underlying numeric series.

The output layout mirrors the ``bayes`` pipeline — figures under
``<out-dir>/plots/``, tables under ``<out-dir>/tables/`` — so the
Quarto template at ``docs/models/<model-id>/index.qmd`` can locate
artefacts without per-model wiring.

Examples
--------
    python scripts/render_selection_diagnostics.py \\
        --fit-dir output/selection/variantC/20260420-094500

    python scripts/render_selection_diagnostics.py \\
        --idata output/selection/variantC/20260420-094500/idata.nc \\
        --cells output/selection/variantC/20260420-094500/cells.parquet \\
        --out-dir docs/figures/variantC
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import dse_research_utils.environment.setup as setup
import dse_research_utils.plot.styles as plot_styles
import pandas as pd

from dspopulations_us_birth_certificates import cli_output
from dspopulations_us_birth_certificates.selection import diagnostics


@dataclass
class RenderCliConfig:
    idata_path: Path
    cells_path: Path
    out_dir: Path
    post_dobbs_year_start: int
    cchd_target: float
    hdi_prob: float
    strata: tuple[str, ...]


def _parse_args(argv: list[str] | None) -> RenderCliConfig:
    p = argparse.ArgumentParser(
        description=(
            "Render posterior diagnostics for a selection-model fit."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--fit-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing idata.nc and cells.parquet. If set, "
            "--idata / --cells / --out-dir default from it."
        ),
    )
    p.add_argument("--idata", type=Path, default=None)
    p.add_argument("--cells", type=Path, default=None)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to write plots/ and tables/ subdirectories into.",
    )
    p.add_argument(
        "--post-dobbs-year-start",
        type=int,
        default=None,
        help=(
            "year_idx at which the post-Dobbs sigma kicks in. Read from "
            "cells.attrs when omitted."
        ),
    )
    p.add_argument(
        "--cchd-target",
        type=float,
        default=0.225,
        help="EUROCAT-published CCHD prevalence for the consistency plot.",
    )
    p.add_argument(
        "--hdi-prob",
        type=float,
        default=0.94,
        help="Credible-interval width for forest plots and PPC bars.",
    )
    p.add_argument(
        "--strata",
        nargs="+",
        default=["year_idx", "race_idx", "age_idx"],
        help="Cell columns to use as strata for the PPC panel plots.",
    )
    ns = p.parse_args(argv)

    fit_dir = ns.fit_dir
    idata = ns.idata or (fit_dir / "idata.nc" if fit_dir else None)
    cells = ns.cells or (fit_dir / "cells.parquet" if fit_dir else None)
    out_dir = ns.out_dir or fit_dir

    if idata is None or cells is None or out_dir is None:
        raise SystemExit(
            "Specify either --fit-dir or (--idata AND --cells AND --out-dir)."
        )
    if not idata.exists():
        raise SystemExit(f"idata not found: {idata}")
    if not cells.exists():
        raise SystemExit(f"cells not found: {cells}")

    return RenderCliConfig(
        idata_path=idata,
        cells_path=cells,
        out_dir=out_dir,
        post_dobbs_year_start=(
            ns.post_dobbs_year_start
            if ns.post_dobbs_year_start is not None
            else -1
        ),
        cchd_target=ns.cchd_target,
        hdi_prob=ns.hdi_prob,
        strata=tuple(ns.strata),
    )


def _save_figure(fig, plots_dir: Path, stem: str, *, data=None) -> None:
    """Save PNG + SVG (+ optional CSV) — mirrors ``bayes.plots._save``."""
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        plots_dir / f"{stem}.png",
        dpi=plot_styles.DPI_FILE,
        bbox_inches="tight",
    )
    fig.savefig(plots_dir / f"{stem}.svg", bbox_inches="tight")
    if data is not None:
        data.to_csv(plots_dir / f"{stem}.csv", index=False)
    plt.close(fig)


def _save_table(df: pd.DataFrame, tables_dir: Path, name: str) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables_dir / name, index=False)


def _resolve_dobbs(cells: pd.DataFrame, cli_value: int) -> int:
    """Prefer CLI override, else cells.attrs, else default 6 (2022 - 2016)."""
    if cli_value >= 0:
        return cli_value
    attr = cells.attrs.get("post_dobbs_year_start")
    if attr is not None:
        return int(attr)
    cli_output.warning(
        "post_dobbs_year_start not supplied and not on cells.attrs; "
        "falling back to 6 (2022 given year_start=2016)."
    )
    return 6


def _run_diagnostic(
    label: str,
    func: Callable[[], tuple[object, pd.DataFrame | None]],
    plots_dir: Path,
    tables_dir: Path,
    stem: str,
) -> None:
    """Execute one diagnostic and log success / failure without aborting."""
    try:
        fig, data = func()
    except Exception as exc:  # noqa: BLE001 — one failure mustn't sink the run
        cli_output.warning(f"{label}: {type(exc).__name__}: {exc}")
        return
    _save_figure(fig, plots_dir, stem, data=data)
    if data is not None:
        _save_table(data, tables_dir, f"{stem}.csv")
    cli_output.success(f"{label} -> {plots_dir / stem}.png")


def main(argv: list[str] | None = None) -> int:
    cli = _parse_args(argv)
    setup.init_script()

    cli_output.banner(
        "render_selection_diagnostics",
        f"{cli.idata_path.name} + {cli.cells_path.name}",
    )

    cli_output.section("Inputs")
    import arviz as az

    idata = az.from_netcdf(str(cli.idata_path))
    cells = pd.read_parquet(cli.cells_path)
    dobbs = _resolve_dobbs(cells, cli.post_dobbs_year_start)
    cli_output.print_kv(
        "Paths & settings",
        [
            ("idata", cli.idata_path),
            ("cells", cli.cells_path),
            ("out_dir", cli.out_dir),
            ("n_cells", len(cells)),
            ("post_dobbs_year_start", dobbs),
            ("cchd_target", cli.cchd_target),
            ("hdi_prob", cli.hdi_prob),
            ("strata", list(cli.strata)),
        ],
    )

    plots_dir = cli.out_dir / "plots"
    tables_dir = cli.out_dir / "tables"

    cli_output.section("Diagnostics")

    _run_diagnostic(
        "Identifiability pair-plot",
        lambda: (
            diagnostics.identifiability_pairplot(idata),
            diagnostics.identifiability_table(idata),
        ),
        plots_dir,
        tables_dir,
        "identifiability",
    )

    _run_diagnostic(
        "Dobbs year trajectory",
        lambda: (
            diagnostics.dobbs_year_trajectory_plot(
                idata,
                post_dobbs_year_start=dobbs,
                hdi_prob=cli.hdi_prob,
            ),
            diagnostics.dobbs_year_trajectory_table(
                idata,
                post_dobbs_year_start=dobbs,
                hdi_prob=cli.hdi_prob,
            ),
        ),
        plots_dir,
        tables_dir,
        "dobbs_year_trajectory",
    )

    _run_diagnostic(
        "CCHD consistency",
        lambda: (
            diagnostics.cchd_consistency_check(
                idata, cells, published_cchd_prevalence=cli.cchd_target
            ),
            diagnostics.cchd_consistency_summary(
                idata, cells, published_cchd_prevalence=cli.cchd_target
            ),
        ),
        plots_dir,
        tables_dir,
        "cchd_consistency",
    )

    _run_diagnostic(
        "Age curve check",
        lambda: (
            diagnostics.age_curve_check(idata, cells, hdi_prob=cli.hdi_prob),
            diagnostics.age_curve_table(idata, hdi_prob=cli.hdi_prob),
        ),
        plots_dir,
        tables_dir,
        "age_curve",
    )

    def _decomposition() -> tuple[object, pd.DataFrame]:
        fig = diagnostics.decomposition_by_race(idata, cells)
        data = getattr(fig, "_selection_data", None)
        return fig, data

    _run_diagnostic(
        "Decomposition by race",
        _decomposition,
        plots_dir,
        tables_dir,
        "decomposition_by_race",
    )

    for stratum in cli.strata:
        _run_diagnostic(
            f"PPC by {stratum}",
            lambda stratum=stratum: (
                diagnostics.posterior_predictive_by_stratum(
                    idata, cells, stratum_col=stratum, hdi_prob=cli.hdi_prob
                ),
                None,
            ),
            plots_dir,
            tables_dir,
            f"ppc_{stratum}",
        )

    cli_output.section("Convergence summary")
    try:
        summary = diagnostics.summary_table(idata)
        health = diagnostics.convergence_health(summary)
        _save_table(summary, tables_dir, "convergence_summary.csv")
        cli_output.info(
            f"max Rhat=[bold]{health['max_rhat']:.4f}[/bold] "
            f"(target <{health['rhat_threshold']}), "
            f"min ESS=[bold]{health['min_ess']:.0f}[/bold] "
            f"(target >={health['ess_threshold']:.0f})"
        )
        if health["all_ok"]:
            cli_output.success("Convergence checks passed.")
        else:
            cli_output.warning(
                "Convergence flags — inspect tables/convergence_summary.csv."
            )
    except Exception as exc:  # noqa: BLE001
        cli_output.warning(
            f"Convergence summary failed: {type(exc).__name__}: {exc}"
        )

    cli_output.section("Done")
    cli_output.info(f"plots -> [blue]{plots_dir}[/blue]")
    cli_output.info(f"tables -> [blue]{tables_dir}[/blue]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
