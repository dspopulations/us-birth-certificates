"""Compare recorded, predicted-missing, and recorded+predicted DS births.

Iterates over the category groupings registered in
``predicted_analyses.CATEGORY_GROUPINGS`` (currently ``mage_c``,
``mracehisp``, and ``meduc``) and for each one writes a stacked-
proportion plot plus a wide summary CSV. All artefacts land in a
timestamped run directory so the Quarto template at
``docs/analysis/predicted.qmd`` can render against it.

The prediction flag ``ds_pred_missing`` must already be present on the
``us_births`` table — that column is written by ``scripts/fit_model.py``
(``write_predictions_to_duckdb``). If it is missing, run fit_model with
``--write-predictions`` first.

Example
-------
    python scripts/analyse_predicted.py
    python scripts/analyse_predicted.py --years 2016-2024 --render
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import dse_research_utils.environment.setup as setup
import matplotlib.pyplot as plt

from dspopulations_us_birth_certificates import cli_output
from dspopulations_us_birth_certificates.predicted_analyses import (
    CATEGORY_GROUPINGS,
    CategoryGrouping,
    copy_analysis_template,
    load_category_counts,
    plot_stacked_proportions,
    render_quarto,
    save_category_summary,
    save_config,
)


def _parse_year_range(raw: str | None) -> tuple[int | None, int | None]:
    if raw is None:
        return None, None
    try:
        lo, hi = raw.split("-", 1)
        return int(lo), int(hi)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--years expects 'YYYY-YYYY', got {raw!r}"
        ) from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare maternal/child characteristics across recorded, "
            "predicted-missing, and recorded+predicted DS births."
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
        "--years",
        default="2016-2024",
        help=(
            "Year range as 'YYYY-YYYY'. Defaults to the model training "
            "window (2016-2024) so the report's row scope matches the "
            "scope the gradient-boosting models were fit against. Pass an "
            "explicit range to override."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: output/analyse_predicted/<timestamp>).",
    )
    p.add_argument(
        "--all-years",
        action="store_true",
        help=(
            "Count across all years, not just those with prediction coverage. "
            "Default is to restrict to rows where p_ds_lb_pred_01 is non-null "
            "so `recorded` and `predicted` describe the same underlying years."
        ),
    )
    p.add_argument(
        "--render",
        action="store_true",
        help=(
            "Invoke `quarto render` on the per-run index.qmd after generating "
            "artefacts. Without this flag the template is copied but not "
            "rendered. Requires `quarto` on PATH."
        ),
    )
    return p.parse_args(argv)


def _run_grouping(
    grouping: CategoryGrouping,
    *,
    duckdb_path: Path,
    start_year: int | None,
    end_year: int | None,
    restrict: bool,
    output_dir: Path,
    year_suffix: str,
) -> None:
    cli_output.section(f"{grouping.variable}: load + plot")
    counts = load_category_counts(
        grouping,
        db_path=duckdb_path,
        start_year=start_year,
        end_year=end_year,
        restrict_to_prediction_coverage=restrict,
    )
    cli_output.info(
        f"recorded = {int(counts['recorded'].sum()):,}, "
        f"predicted = {int(counts['predicted'].sum()):,}, "
        f"rprime   = {int(counts['rprime'].sum()):,}"
    )

    title = f"{grouping.title}: recorded vs. predicted DS births{year_suffix}"
    fig = plot_stacked_proportions(
        counts,
        title=title,
        legend_title=grouping.legend_title,
        colormap=grouping.colormap,
        save=True,
        output_dir=str(output_dir),
        file_name=f"{grouping.variable}_recorded_vs_predicted",
    )
    plt.close(fig)
    save_category_summary(counts, output_dir, variable=grouping.variable)
    cli_output.success(
        f"{grouping.variable}: plot + summary -> {output_dir}"
    )


def main(argv: list[str] | None = None) -> int:
    ns = parse_args(argv)
    setup.init_script()

    start_year, end_year = _parse_year_range(ns.years)
    output_dir = ns.output_dir or (
        Path("output/analyse_predicted") / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cli_output.section("analyse_predicted")
    cli_output.info(f"DuckDB: [blue]{ns.duckdb_path}[/blue]")
    cli_output.info(f"Output: [blue]{output_dir}[/blue]")
    if start_year is not None or end_year is not None:
        cli_output.info(
            f"Years : {start_year if start_year is not None else '*'}"
            f"–{end_year if end_year is not None else '*'}"
        )

    save_config(
        output_dir,
        {
            "duckdb_path": str(ns.duckdb_path),
            "start_year": start_year,
            "end_year": end_year,
            "restrict_to_prediction_coverage": not ns.all_years,
            "groupings": list(CATEGORY_GROUPINGS.keys()),
        },
    )

    year_suffix = (
        f"  ({start_year}–{end_year})"
        if start_year is not None and end_year is not None
        else ""
    )
    for grouping in CATEGORY_GROUPINGS.values():
        _run_grouping(
            grouping,
            duckdb_path=ns.duckdb_path,
            start_year=start_year,
            end_year=end_year,
            restrict=not ns.all_years,
            output_dir=output_dir,
            year_suffix=year_suffix,
        )

    cli_output.section("Report template")
    qmd_path = copy_analysis_template(output_dir, template_name="predicted")
    if qmd_path is not None:
        cli_output.success(f"index.qmd -> {qmd_path}")
    else:
        cli_output.info(
            "No Quarto template at docs/analysis/predicted.qmd — "
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
