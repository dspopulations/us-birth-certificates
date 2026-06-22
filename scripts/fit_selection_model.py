"""Fit the three-stage Bayesian selection model.

Thin CLI over ``dspopulations_us_birth_certificates.selection``. Given a
variant (A/B/C), spec (theta_only / theta_s / single_eta / full), and
run profile, this script:

    1. Aggregates ``us_births`` into selection cells via
       ``selection.prepare_cells``.
    2. Builds the PyMC model via ``selection.build_model``.
    3. Runs prior-predictive + NUTS + posterior-predictive sampling
       via ``selection.sample``.
    4. Writes ``idata.nc``, ``cells.parquet``, ``config.json``,
       ``run_config.json``, ``summary.csv``, diagnostic plots/tables,
       and copies the Quarto template at
       ``docs/models/selection/index.qmd`` into the run directory.

Profiles
--------
- ``dev``      — 1000 tune + 1000 draws × 2 chains, target_accept=0.9,
                 nutpie. Enough posterior mass to clear ESS gates on
                 the named RVs at full spec; a few minutes for
                 theta_only, ~30 min for full.
- ``reporting``— 1500 tune + 1500 draws × 4 chains, target_accept=0.95,
                 nutpie. The publication-quality preset.

Examples
--------
    python scripts/fit_selection_model.py --variant C --spec theta_only \\
        --profile dev

    python scripts/fit_selection_model.py --variant C --spec full \\
        --profile reporting --render
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import dse_research_utils.environment.setup as setup
import dse_research_utils.metadata.packages as package_metadata
import duckdb
import numpy as np

from dspopulations_us_birth_certificates import PACKAGE_LIST, cli_output
from dspopulations_us_birth_certificates.selection import (
    MODEL_ID,
    SPECS,
    VARIANTS,
    FitContext,
    SelectionModelConfig,
    build_model,
    copy_docs_template,
    diagnostics,
    prepare_cells,
    preset_names,
    render_quarto,
    sample,
    save_artefacts,
    save_summary,
    selection_run_config,
    summarise_cells,
)
from dspopulations_us_birth_certificates.selection.render import (
    DEFAULT_STRATA,
    RenderOptions,
    render_all,
)


@dataclass
class FitSelectionCliConfig:
    variant: str
    spec: str
    profile: str
    start_year: int
    end_year: int
    random_seed: int
    duckdb_path: Path
    output_dir: Path
    prior_only: bool
    render: bool
    overrides: dict[str, Any] = field(default_factory=dict)


def _parse_years(raw: str, fallback: tuple[int, int]) -> tuple[int, int]:
    if raw is None:
        return fallback
    try:
        lo, hi = raw.split("-", 1)
        return int(lo), int(hi)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--years expects 'YYYY-YYYY', got {raw!r}"
        ) from exc


def parse_args(argv: list[str] | None = None) -> FitSelectionCliConfig:
    p = argparse.ArgumentParser(
        description="Fit the three-stage Bayesian selection model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--variant",
        required=True,
        choices=sorted(VARIANTS),
        help="Prior-sensitivity variant (A/B/C).",
    )
    p.add_argument(
        "--spec",
        default="full",
        choices=list(SPECS),
        help="Staged spec to fit.",
    )
    p.add_argument(
        "--profile",
        default="dev",
        choices=list(preset_names()),
        help="Run-config preset.",
    )
    p.add_argument(
        "--years",
        default=None,
        help="Year range as 'YYYY-YYYY'. Defaults to 2016-2024.",
    )
    p.add_argument("--random-seed", type=int, default=47)
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
        help="Output directory (default: output/selection/<variant>/<spec>/<ts>).",
    )
    p.add_argument(
        "--prior-only",
        action="store_true",
        help="Run prior-predictive only; skip NUTS + posterior predictive.",
    )
    p.add_argument("--draws", type=int, default=None)
    p.add_argument("--tune", type=int, default=None)
    p.add_argument("--chains", type=int, default=None)
    p.add_argument("--target-accept", type=float, default=None)
    p.add_argument("--prior-predictive-samples", type=int, default=None)
    p.add_argument(
        "--render",
        action="store_true",
        help=(
            "After the fit, render the Quarto template at "
            "docs/models/selection/index.qmd against the run's output "
            "directory. Requires ``quarto`` on PATH."
        ),
    )

    ns = p.parse_args(argv)
    start_year, end_year = _parse_years(ns.years, (2016, 2024))
    out_dir = ns.output_dir or (
        Path("output/selection")
        / ns.variant
        / ns.spec
        / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    overrides = {
        k: v
        for k, v in {
            "draws": ns.draws,
            "tune": ns.tune,
            "chains": ns.chains,
            "target_accept": ns.target_accept,
            "prior_predictive_samples": ns.prior_predictive_samples,
        }.items()
        if v is not None
    }
    return FitSelectionCliConfig(
        variant=ns.variant,
        spec=ns.spec,
        profile=ns.profile,
        start_year=start_year,
        end_year=end_year,
        random_seed=ns.random_seed,
        duckdb_path=ns.duckdb_path,
        output_dir=out_dir,
        prior_only=ns.prior_only,
        render=ns.render,
        overrides=overrides,
    )


def _build_run_config(cli: FitSelectionCliConfig):
    base = selection_run_config(cli.profile, random_seed=cli.random_seed)
    if not cli.overrides:
        return base
    return replace(base, **cli.overrides)


def main(argv: list[str] | None = None) -> int:
    cli = parse_args(argv)
    setup.init_script()
    np.random.seed(cli.random_seed)
    cli.output_dir.mkdir(parents=True, exist_ok=True)

    cli_output.banner(
        "fit_selection_model",
        f"variant={cli.variant}  spec={cli.spec}  profile={cli.profile}",
    )

    cli_output.section("Environment")
    package_metadata.report_package_versions(list(PACKAGE_LIST))

    cli_output.section("Run configuration")
    run_config = _build_run_config(cli)
    cli_output.info(
        f"years=[bold]{cli.start_year}-{cli.end_year}[/bold], "
        f"draws=[bold]{run_config.draws}[/bold], "
        f"tune=[bold]{run_config.tune}[/bold], "
        f"chains=[bold]{run_config.chains}[/bold], "
        f"target_accept=[bold]{run_config.target_accept}[/bold], "
        f"sampler=[bold]{run_config.nuts_sampler}[/bold]"
    )
    cli_output.info(f"output_dir=[blue]{cli.output_dir}[/blue]")

    cli_output.section("Load cells")
    # Variant D is the comparative R' track: R_cell counts down_ind OR the C-only,
    # demographically-blind GB predicted-missing flag (USBC11_M1_CN), and recording
    # is pinned to ~1 (see variant_D_rprime). R' = recorded + predicted-missing;
    # "predicted" is the missing flag, not a C+P training label.
    missing_flag = "ds_pred_missing_14" if cli.variant == "D" else None
    con = duckdb.connect(str(cli.duckdb_path), read_only=True)
    try:
        cells = prepare_cells(
            con,
            year_range=(cli.start_year, cli.end_year),
            missing_flag_column=missing_flag,
        )
    finally:
        con.close()
    summary = summarise_cells(cells)
    cli_output.print_kv(
        "Cells",
        [
            ("n_cells", summary["n_cells"]),
            ("n_total (livebirths)", f"{summary['n_total']:,}"),
            ("r_total (recorded DS)", f"{summary['r_total']:,}"),
            ("recorded_rate", f"{summary['recorded_rate']:.2e}"),
            ("year_range", summary.get("year_range")),
        ],
    )

    cli_output.section("Build model")
    priors = VARIANTS[cli.variant]()
    n_year = cells.attrs["n_year"]
    model = build_model(cells, priors, spec=cli.spec, n_year=n_year)

    cli_output.section("Sample")
    idata = sample(model, config=run_config, prior_only=cli.prior_only)

    cli_output.section("Save artefacts")
    model_config = SelectionModelConfig.from_priors(
        variant=cli.variant,
        spec=cli.spec,
        year_range=(cli.start_year, cli.end_year),
        priors_obj=priors,
        notes=f"profile={cli.profile}",
    )
    context = FitContext(
        config=model_config,
        run_config=run_config,
        output_dir=cli.output_dir,
        cells=cells,
        model=model,
        idata=idata,
    )
    save_artefacts(context, cli.output_dir)
    cli_output.success(
        f"idata.nc, cells.parquet, config.json -> {cli.output_dir}"
    )

    qmd_path = copy_docs_template(MODEL_ID, cli.output_dir)
    if qmd_path is not None:
        cli_output.success(f"index.qmd -> {qmd_path}")
    else:
        cli_output.info(
            f"No Quarto template at docs/models/{MODEL_ID}/index.qmd "
            "— skipping template copy."
        )

    if cli.prior_only:
        cli_output.info("Prior-only run — skipping posterior summary / diagnostics.")
        return 0

    cli_output.section("Posterior summary")
    az_summary = diagnostics.summary_table(idata)
    save_summary(az_summary, cli.output_dir)
    health = diagnostics.convergence_health(az_summary)
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
            "Convergence checks did not all pass — inspect summary.csv. "
            "Consider --profile reporting or --target-accept 0.98."
        )

    cli_output.section("Diagnostics")
    render_all(
        idata,
        cells,
        cli.output_dir,
        options=RenderOptions(strata=DEFAULT_STRATA),
    )

    if cli.render and qmd_path is not None:
        cli_output.section("Render")
        try:
            render_quarto(qmd_path)
            cli_output.success(f"Rendered {qmd_path.with_suffix('.html')}")
        except FileNotFoundError:
            cli_output.warning(
                "`quarto` not on PATH — render manually: "
                f"quarto render {qmd_path}"
            )
        except Exception as exc:  # noqa: BLE001
            cli_output.warning(
                f"Quarto render raised {type(exc).__name__}: {exc}"
            )

    cli_output.section("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
