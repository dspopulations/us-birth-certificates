"""Fit a Bayesian regression model for Down syndrome live-birth trends.

Thin CLI over ``dspopulations_us_birth_certificates.bayes``. Given a
model id from the registry, an outcome construction, a year range, and a
profile, this script:

    1. Aggregates ``us_births`` into cells via ``bayes.load_cells``.
    2. Builds the PyMC model via ``MODELS[model_id].build(cells)``.
    3. Runs prior-predictive + NUTS + posterior-predictive sampling.
    4. Writes ``idata.nc``, ``cells.parquet``, ``config.json``,
       ``run_config.json``, ``summary.csv``, and core posterior plots.

Configuration profiles
----------------------
Pick a profile with ``--profile {dev,reporting}``. Presets come from
``BayesRunConfig.from_name()`` so CLI and library agree on meaning.

- ``dev``: 500 tune + 500 draw, 2 chains, nutpie. Minutes on a laptop.
- ``reporting``: 2000 tune + 2000 draw, 4 chains, target_accept=0.9.

Examples
--------
    python scripts/fit_bayes_model.py --model-id m1-year-age \\
        --outcome recorded --profile dev

    python scripts/fit_bayes_model.py --model-id m1-year-age \\
        --outcome recorded_plus_predicted --profile reporting
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import dse_research_utils.environment.setup as setup
import dse_research_utils.metadata.packages as package_metadata
import numpy as np

from dspopulations_us_birth_certificates import PACKAGE_LIST, cli_output
from dspopulations_us_birth_certificates.bayes import (
    MODELS,
    BayesFitContext,
    BayesRunConfig,
    diagnostics,
    load_cells,
    plots,
    sample,
    save_artefacts,
    save_summary,
)


@dataclass
class BayesFitCliConfig:
    model_id: str
    outcome: str
    profile: str
    start_year: int
    end_year: int
    random_seed: int
    duckdb_path: Path
    output_dir: Path
    prior_only: bool
    drop_na_dims: bool
    overrides: dict = field(default_factory=dict)


def _parse_year_range(raw: str | None, fallback: tuple[int, int]) -> tuple[int, int]:
    if raw is None:
        return fallback
    try:
        lo, hi = raw.split("-", 1)
        return int(lo), int(hi)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--years expects 'YYYY-YYYY', got {raw!r}"
        ) from exc


def parse_args(argv: list[str] | None = None) -> BayesFitCliConfig:
    p = argparse.ArgumentParser(
        description="Fit a Bayesian regression model for DS live-birth trends.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model-id",
        required=True,
        choices=sorted(MODELS.keys()),
        help="Named Bayesian model variant from bayes.MODELS.",
    )
    p.add_argument(
        "--outcome",
        required=True,
        choices=("recorded", "recorded_plus_predicted"),
        help="Which positive-indicator construction to use.",
    )
    p.add_argument(
        "--profile",
        default="dev",
        choices=list(BayesRunConfig.preset_names()),
        help="Run-config preset.",
    )
    p.add_argument(
        "--years",
        default=None,
        help="Year range as 'YYYY-YYYY'. Defaults to the model's year_range.",
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
        help="Output directory (default: output/bayes/<model>/<outcome>/<ts>).",
    )
    p.add_argument(
        "--prior-only",
        action="store_true",
        help="Run prior-predictive only; skip NUTS + posterior predictive.",
    )
    p.add_argument(
        "--keep-na-dims",
        action="store_true",
        help="Keep cells with NULL dim values (default: drop).",
    )
    p.add_argument(
        "--draws",
        type=int,
        default=None,
        help="Override posterior draws from the profile.",
    )
    p.add_argument(
        "--tune",
        type=int,
        default=None,
        help="Override tuning draws from the profile.",
    )
    p.add_argument(
        "--chains",
        type=int,
        default=None,
        help="Override chain count from the profile.",
    )

    ns = p.parse_args(argv)
    definition = MODELS[ns.model_id]
    start_year, end_year = _parse_year_range(ns.years, definition.year_range)

    out_dir = ns.output_dir or (
        Path("output/bayes")
        / ns.model_id
        / ns.outcome
        / datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    overrides = {
        k: v
        for k, v in {
            "draws": ns.draws,
            "tune": ns.tune,
            "chains": ns.chains,
        }.items()
        if v is not None
    }

    return BayesFitCliConfig(
        model_id=ns.model_id,
        outcome=ns.outcome,
        profile=ns.profile,
        start_year=start_year,
        end_year=end_year,
        random_seed=ns.random_seed,
        duckdb_path=ns.duckdb_path,
        output_dir=out_dir,
        prior_only=ns.prior_only,
        drop_na_dims=not ns.keep_na_dims,
        overrides=overrides,
    )


def _build_run_config(cli: BayesFitCliConfig) -> BayesRunConfig:
    base = BayesRunConfig.from_name(cli.profile, random_seed=cli.random_seed)
    if not cli.overrides:
        return base
    from dataclasses import replace

    return replace(base, **cli.overrides)


def main(argv: list[str] | None = None) -> int:
    cli = parse_args(argv)
    setup.init_script()
    np.random.seed(cli.random_seed)
    cli.output_dir.mkdir(parents=True, exist_ok=True)

    cli_output.banner(
        "fit_bayes_model",
        f"model={cli.model_id}  outcome={cli.outcome}  profile={cli.profile}",
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
        f"sampler=[bold]{run_config.nuts_sampler}[/bold]"
    )
    cli_output.info(f"output_dir=[blue]{cli.output_dir}[/blue]")

    cli_output.section("Load cells")
    definition = MODELS[cli.model_id]
    cells = load_cells(
        outcome=cli.outcome,
        dims=definition.dims,
        year_range=(cli.start_year, cli.end_year),
        db_path=cli.duckdb_path,
        drop_na_dims=cli.drop_na_dims,
    )
    cli_output.info(
        f"cells=[bold]{len(cells)}[/bold], "
        f"n_total=[bold]{int(cells['n_cell'].sum()):,}[/bold], "
        f"y_total=[bold]{int(cells['y_cell'].sum()):,}[/bold]"
    )

    cli_output.section("Build model")
    model = definition.build(cells)

    cli_output.section("Sample")
    idata = sample(model, config=run_config, prior_only=cli.prior_only)

    context = BayesFitContext(
        config=definition.to_config(outcome=cli.outcome),
        run_config=run_config,
        output_dir=cli.output_dir,
        cells=cells,
        model=model,
        idata=idata,
    )

    cli_output.section("Save artefacts")
    save_artefacts(context, cli.output_dir)
    cli_output.success(f"idata.nc, cells.parquet, config.json -> {cli.output_dir}")

    if cli.prior_only:
        cli_output.info("Prior-only run - skipping summary / plots.")
        return 0

    cli_output.section("Diagnostics")
    summary = diagnostics.summary_table(idata)
    save_summary(summary, cli.output_dir)
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
        cli_output.warning(
            "Convergence checks did not all pass - inspect summary.csv and "
            "consider re-running with --profile reporting or more draws."
        )

    cli_output.section("Plots")
    plots_dir = cli.output_dir / "plots"
    key_vars = tuple(
        v
        for v in ("alpha", "ls_year", "eta_year", "ls_age", "eta_age")
        if v in idata.posterior.data_vars
    )
    try:
        for dim in definition.dims:
            plots.plot_trend_by_dim(
                idata, cells, dim=dim, output_path=plots_dir / f"trend_{dim}"
            )
        plots.plot_ppc(idata, output_path=plots_dir / "ppc")
        if key_vars:
            plots.plot_trace(
                idata, var_names=key_vars, output_path=plots_dir / "trace_key_rvs"
            )
        cli_output.success(f"Plots -> {plots_dir}")
    except Exception as exc:  # noqa: BLE001 — diagnostics shouldn't fail the run
        cli_output.warning(f"Plot generation raised {type(exc).__name__}: {exc}")

    cli_output.section("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
