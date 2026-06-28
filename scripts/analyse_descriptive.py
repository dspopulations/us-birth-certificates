"""Generate the descriptive recorded-DS report artefacts and (optionally) render it.

Produces every figure/table for ``docs/analysis/descriptive.qmd`` from read-only
DuckDB queries against ``data/us_births.db``, into a timestamped run directory,
then copies the Quarto template alongside them and (with ``--render``) renders it.
Mirrors ``scripts/analyse_predicted.py``.

Example
-------
    python scripts/analyse_descriptive.py --render
"""

from __future__ import annotations

import os

# Headless rendering + the MKL/OpenMP single-thread workaround this Windows box
# needs to stop numpy/matplotlib from hard-crashing on import. Must precede any
# numpy/matplotlib import (i.e. the dse_research_utils / descriptive_analyses
# imports below).
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import argparse
import sys
from datetime import datetime
from pathlib import Path

import dse_research_utils.environment.setup as setup

from dspopulations_us_birth_certificates import cli_output
from dspopulations_us_birth_certificates.descriptive_analyses import build_all
from dspopulations_us_birth_certificates.predicted_analyses import (
    copy_analysis_template,
    render_quarto,
    save_config,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build the descriptive report on recorded Down syndrome live "
            "births (1989-2024) and optionally render the Quarto template."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--duckdb-path",
        type=Path,
        default=Path("data/us_births.db"),
        help="Path to the DuckDB database.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: output/analyse_descriptive/<timestamp>).",
    )
    p.add_argument(
        "--render",
        action="store_true",
        help=(
            "Invoke `quarto render` on the per-run index.qmd after generating "
            "artefacts. Requires `quarto` on PATH."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    ns = parse_args(argv)
    setup.init_script()

    output_dir = ns.output_dir or (
        Path("output/analyse_descriptive") / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cli_output.section("analyse_descriptive")
    cli_output.info(f"DuckDB: [blue]{ns.duckdb_path}[/blue]")
    cli_output.info(f"Output: [blue]{output_dir}[/blue]")

    cli_output.section("Sections A–E")
    summary = build_all(ns.duckdb_path, output_dir)
    summary["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cli_output.info(
        f"recorded = {summary.get('total_recorded', 0):,}, "
        f"births = {summary.get('total_births', 0):,}, "
        f"recording rate = {summary.get('overall_recording_rate', float('nan')):.1%}"
    )
    save_config(output_dir, summary)
    cli_output.success(f"Artefacts + config -> {output_dir}")

    cli_output.section("Report template")
    qmd_path = copy_analysis_template(output_dir, template_name="descriptive")
    if qmd_path is not None:
        cli_output.success(f"index.qmd -> {qmd_path}")
    else:
        cli_output.warning(
            "No Quarto template at docs/analysis/descriptive.qmd — "
            "skipping template copy."
        )

    if ns.render and qmd_path is not None:
        cli_output.section("Render")
        try:
            render_quarto(qmd_path)
            cli_output.success(f"Rendered {qmd_path.with_suffix('.html')}")
        except FileNotFoundError:
            cli_output.warning(
                "`quarto` not found on PATH — install Quarto and retry, "
                f"or render manually: quarto render {qmd_path}"
            )
        except Exception as exc:  # noqa: BLE001 — rendering is optional
            cli_output.warning(f"Quarto render raised {type(exc).__name__}: {exc}")

    cli_output.section("Done")
    cli_output.success(f"Artefacts in {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
