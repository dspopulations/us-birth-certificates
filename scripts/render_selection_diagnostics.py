"""Render posterior diagnostics for a fitted selection-model run.

Given a fit directory (containing ``idata.nc`` + ``cells.parquet``) or
explicit ``--idata`` / ``--cells`` paths, this script calls
:func:`dspopulations_us_birth_certificates.selection.render.render_all`
to produce the six diagnostic figures plus their CSV companions.

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
import sys
from dataclasses import dataclass
from pathlib import Path

import dse_research_utils.environment.setup as setup
import pandas as pd

from dspopulations_us_birth_certificates import cli_output
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
    post_dobbs_year_start: int
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
        default=list(DEFAULT_STRATA),
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
            ns.post_dobbs_year_start if ns.post_dobbs_year_start is not None else -1
        ),
        cchd_target=ns.cchd_target,
        hdi_prob=ns.hdi_prob,
        strata=tuple(ns.strata),
    )


def _resolve_dobbs(cells: pd.DataFrame, cli_value: int) -> int:
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

    cli_output.section("Diagnostics")
    render_all(
        idata,
        cells,
        cli.out_dir,
        options=RenderOptions(
            post_dobbs_year_start=dobbs,
            cchd_target=cli.cchd_target,
            hdi_prob=cli.hdi_prob,
            strata=cli.strata,
        ),
    )

    cli_output.section("Done")
    cli_output.info(f"plots -> [blue]{cli.out_dir / 'plots'}[/blue]")
    cli_output.info(f"tables -> [blue]{cli.out_dir / 'tables'}[/blue]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
