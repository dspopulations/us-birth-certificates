"""Render posterior diagnostics for a fitted selection-model run.

Given a fit directory (containing ``idata.nc`` + ``cells.parquet``) or
explicit ``--idata`` / ``--cells`` paths, this script calls
:func:`dspopulations_us_birth_certificates.selection.render.render_all`
to produce the diagnostic figures plus their CSV companions.

The shared rendering loop lives in ``selection.render`` so the fit CLI
can call the same code path inline after NUTS.

Examples
--------
    python scripts/render_selection_diagnostics.py \\
        --fit-dir output/selection/C/full/20260420-094500

    python scripts/render_selection_diagnostics.py \\
        --idata output/selection/C/full/20260420-094500/idata.nc \\
        --cells output/selection/C/full/20260420-094500/cells.parquet \\
        --out-dir docs/figures/variantC
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import dse_research_utils.environment.setup as setup
import pandas as pd

from dspopulations_us_birth_certificates import cli_output
from dspopulations_us_birth_certificates.selection import diagnostics
from dspopulations_us_birth_certificates.selection.render import (
    DEFAULT_STRATA,
    RenderOptions,
    render_all,
)


@dataclass
class RenderCliConfig:
    idata_path: Path
    cells_path: Path
    out_dir: Path
    config_path: Path | None
    cchd_target: float
    hdi_prob: float
    strata: tuple[str, ...]


def _parse_args(argv: list[str] | None) -> RenderCliConfig:
    p = argparse.ArgumentParser(
        description="Render posterior diagnostics for a selection-model fit.",
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
        default=list(DEFAULT_STRATA),
        help="Cell columns to use as strata for the PPC panel plots.",
    )
    ns = p.parse_args(argv)

    fit_dir = ns.fit_dir
    idata = ns.idata or (fit_dir / "idata.nc" if fit_dir else None)
    cells = ns.cells or (fit_dir / "cells.parquet" if fit_dir else None)
    out_dir = ns.out_dir or fit_dir
    config_path = fit_dir / "config.json" if fit_dir else None
    if config_path is None and idata is not None:
        sibling_config = idata.parent / "config.json"
        config_path = sibling_config if sibling_config.is_file() else None
    if config_path is not None and not config_path.is_file():
        config_path = None

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
        config_path=config_path,
        cchd_target=ns.cchd_target,
        hdi_prob=ns.hdi_prob,
        strata=tuple(ns.strata),
    )


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
    config = (
        json.loads(cli.config_path.read_text(encoding="utf-8"))
        if cli.config_path is not None
        else {}
    )
    year_range = config.get("year_range")
    parsed_year_range = (
        (int(year_range[0]), int(year_range[1]))
        if isinstance(year_range, (list, tuple)) and len(year_range) == 2
        else None
    )
    cli_output.print_kv(
        "Paths & settings",
        [
            ("idata", cli.idata_path),
            ("cells", cli.cells_path),
            ("out_dir", cli.out_dir),
            ("config", cli.config_path or "(none)"),
            ("n_cells", len(cells)),
            ("cchd_target", cli.cchd_target),
            ("hdi_prob", cli.hdi_prob),
            ("strata", list(cli.strata)),
        ],
    )

    cli_output.section("Diagnostics")
    render_all(
        idata,
        cells,
        cli.out_dir,
        options=RenderOptions(
            cchd_target=cli.cchd_target,
            hdi_prob=cli.hdi_prob,
            strata=cli.strata,
            priors_config=config.get("priors"),
            year_range=parsed_year_range,
        ),
    )

    # Read summary.csv if the fit CLI saved one; otherwise compute afresh.
    # az.summary on per-cell deterministics is slow, so we prefer the cache.
    cli_output.section("Convergence")
    summary_path = cli.out_dir / "summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path, index_col=0)
        cli_output.info(f"Loaded cached summary from {summary_path}")
    else:
        cli_output.info("Computing posterior summary (no cached summary.csv)...")
        summary = diagnostics.summary_table(idata)
        summary.to_csv(summary_path)
    health = diagnostics.convergence_health(summary)
    cli_output.info(
        f"max Rhat=[bold]{health['max_rhat']:.4f}[/bold] "
        f"(target <{health['rhat_threshold']}), "
        f"min ESS=[bold]{health['min_ess']:.0f}[/bold] "
        f"(target >={health['ess_threshold']:.0f})"
    )
    if health["all_ok"]:
        cli_output.success("Convergence checks passed.")
    else:
        cli_output.warning("Convergence flags — inspect summary.csv.")

    cli_output.section("Done")
    cli_output.info(f"plots -> [blue]{cli.out_dir / 'plots'}[/blue]")
    cli_output.info(f"tables -> [blue]{cli.out_dir / 'tables'}[/blue]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
