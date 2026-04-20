"""Compare two ``analyse_predicted`` runs side-by-side.

Takes two previously-produced ``output/analyse_predicted*/<timestamp>``
directories (one per model variant), stages their summary CSVs under a
new output directory with ``_left``/``_right`` suffixes, copies the
Quarto template at ``docs/analysis/compare_predicted.qmd``, and
optionally renders. The template produces one Δ-annotated table per
registered grouping so shifts in the predicted-missing demographic
profile across variants are visible at a glance.

Typical usage — usbc10 vs usbc11:

    python scripts/compare_predicted.py \\
        --left output/analyse_predicted_usbc10/<ts> \\
        --right output/analyse_predicted_usbc11/<ts> \\
        --left-label usbc10 \\
        --right-label usbc11 \\
        --render
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import dse_research_utils.environment.setup as setup

from dspopulations_us_birth_certificates import cli_output
from dspopulations_us_birth_certificates.predicted_analyses import (
    copy_analysis_template,
    render_quarto,
    stage_compare_artefacts,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare two `analyse_predicted` runs side-by-side into a single "
            "Δ-annotated report."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--left",
        type=Path,
        required=True,
        help="Path to the left-hand analyse_predicted run directory.",
    )
    p.add_argument(
        "--right",
        type=Path,
        required=True,
        help="Path to the right-hand analyse_predicted run directory.",
    )
    p.add_argument(
        "--left-label",
        default="left",
        help="Short label for the left-hand run (e.g. 'usbc10').",
    )
    p.add_argument(
        "--right-label",
        default="right",
        help="Short label for the right-hand run (e.g. 'usbc11').",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory "
            "(default: output/compare_predicted/<timestamp>)."
        ),
    )
    p.add_argument(
        "--render",
        action="store_true",
        help="Invoke `quarto render` after staging.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    ns = parse_args(argv)
    setup.init_script()

    output_dir = ns.output_dir or (
        Path("output/compare_predicted")
        / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cli_output.section("compare_predicted")
    cli_output.info(
        f"Left  [{ns.left_label}]: [blue]{ns.left}[/blue]"
    )
    cli_output.info(
        f"Right [{ns.right_label}]: [blue]{ns.right}[/blue]"
    )
    cli_output.info(f"Output: [blue]{output_dir}[/blue]")

    cli_output.section("Stage artefacts")
    compare_config = stage_compare_artefacts(
        left_dir=ns.left,
        right_dir=ns.right,
        output_dir=output_dir,
        left_label=ns.left_label,
        right_label=ns.right_label,
    )
    cli_output.success(
        f"compare_config.json + {len(compare_config['left_config'].get('groupings', [])) or '?'} "
        f"per-variable summary CSV pairs staged -> {output_dir}"
    )

    qmd_path = copy_analysis_template(
        output_dir, template_name="compare_predicted"
    )
    if qmd_path is not None:
        cli_output.success(f"index.qmd -> {qmd_path}")
    else:
        cli_output.warning(
            "No Quarto template at docs/analysis/compare_predicted.qmd — "
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
            cli_output.warning(
                f"Quarto render raised {type(exc).__name__}: {exc}"
            )

    cli_output.section("Done")
    cli_output.success(f"Artefacts in {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
