"""Fit the core age-reduction-recording Bayesian model."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import dse_research_utils.environment.setup as setup
import dse_research_utils.metadata.packages as package_metadata
import duckdb
import numpy as np

from dspopulations_us_birth_certificates import PACKAGE_LIST, cli_output
from dspopulations_us_birth_certificates.selection import (
    FitContext,
    copy_docs_template,
    diagnostics,
    render_report,
    sample,
    save_artefacts,
    save_summary,
    selection_run_config,
)
from dspopulations_us_birth_certificates.selection.core_models import (
    core_model_names,
    get_core_model_definition,
)
from dspopulations_us_birth_certificates.selection.core_reduction import (
    CORE_REDUCTION_MODEL_ID,
    DEFAULT_EXTRAPOLATED_REDUCTION_START,
    DEFAULT_REDUCTION_CSV,
    CoreReductionModelConfig,
    CoreReductionPriors,
    build_core_reduction_model,
    core_year_summary,
    prepare_core_age_year_cells,
)
from dspopulations_us_birth_certificates.selection.core_reporting import (
    render_core_all,
)


def _parse_years(raw: str | None, fallback: tuple[int, int]) -> tuple[int, int]:
    if raw is None:
        return fallback
    try:
        lo, hi = raw.split("-", 1)
        return int(lo), int(hi)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--years expects 'YYYY-YYYY', got {raw!r}"
        ) from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fit the core age-reduction-recording Bayesian model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "model",
        nargs="?",
        default="DSP001",
        help=f"Core model ID. Valid models: {', '.join(core_model_names())}.",
    )
    p.add_argument("--profile", default="dev", choices=("dev", "reporting"))
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
        "--reduction-csv",
        type=Path,
        default=DEFAULT_REDUCTION_CSV,
        help="CSV containing year,reduction prior means.",
    )
    p.add_argument(
        "--extrapolated-start",
        type=int,
        default=DEFAULT_EXTRAPOLATED_REDUCTION_START,
        help="First year treated as extrapolated for wider reduction uncertainty.",
    )
    p.add_argument("--observed-reduction-sigma", type=float, default=0.20)
    p.add_argument("--extrapolated-reduction-sigma", type=float, default=0.45)
    p.add_argument("--recording-s-mean", type=float, default=0.5)
    p.add_argument("--recording-s-sigma", type=float, default=1.0)
    p.add_argument(
        "--recording-s-year-sigma",
        type=float,
        default=0.35,
        help=(
            "Logit-scale prior SD for centred year offsets in s_year models. "
            "Ignored by constant-s models."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory (default: output/selection_core_reduction/"
            "<DSPnnn>/<timestamp>)."
        ),
    )
    p.add_argument("--prior-only", action="store_true")
    p.add_argument(
        "--render",
        action="store_true",
        help="Copy and render the checked-in Quarto report template.",
    )
    p.add_argument("--draws", type=int, default=None)
    p.add_argument("--tune", type=int, default=None)
    p.add_argument("--chains", type=int, default=None)
    p.add_argument("--target-accept", type=float, default=None)
    p.add_argument("--prior-predictive-samples", type=int, default=None)
    p.add_argument(
        "--nuts-sampler",
        default=None,
        help="Override sampler backend in the selected profile, e.g. nutpie or pymc.",
    )
    ns = p.parse_args(argv)
    try:
        ns.model_definition = get_core_model_definition(ns.model)
    except ValueError as exc:
        p.error(str(exc))
    ns.year_range = _parse_years(ns.years, (2016, 2024))
    ns.output_dir = ns.output_dir or (
        Path("output")
        / CORE_REDUCTION_MODEL_ID
        / ns.model_definition.model_id
        / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    ns.overrides = {
        k: v
        for k, v in {
            "draws": ns.draws,
            "tune": ns.tune,
            "chains": ns.chains,
            "target_accept": ns.target_accept,
            "prior_predictive_samples": ns.prior_predictive_samples,
            "nuts_sampler": ns.nuts_sampler,
        }.items()
        if v is not None
    }
    return ns


def _build_run_config(ns: argparse.Namespace):
    base = selection_run_config(ns.profile, random_seed=ns.random_seed)
    if not ns.overrides:
        return base
    return replace(base, **ns.overrides)


def main(argv: list[str] | None = None) -> int:
    ns = parse_args(argv)
    setup.init_script()
    np.random.seed(ns.random_seed)
    ns.output_dir.mkdir(parents=True, exist_ok=True)

    cli_output.banner(
        "fit_core_reduction_model",
        (
            f"model={ns.model_definition.model_id}  "
            f"years={ns.year_range[0]}-{ns.year_range[1]}  profile={ns.profile}"
        ),
    )
    cli_output.info(ns.model_definition.title)

    cli_output.section("Environment")
    package_metadata.report_package_versions(list(PACKAGE_LIST))

    cli_output.section("Run configuration")
    run_config = _build_run_config(ns)
    cli_output.info(
        f"draws=[bold]{run_config.draws}[/bold], "
        f"tune=[bold]{run_config.tune}[/bold], "
        f"chains=[bold]{run_config.chains}[/bold], "
        f"target_accept=[bold]{run_config.target_accept}[/bold], "
        f"sampler=[bold]{run_config.nuts_sampler}[/bold]"
    )
    cli_output.info(f"output_dir=[blue]{ns.output_dir}[/blue]")

    cli_output.section("Load age-year cells")
    con = duckdb.connect(str(ns.duckdb_path), read_only=True)
    try:
        cells = prepare_core_age_year_cells(con, year_range=ns.year_range)
    finally:
        con.close()
    n_total = int(cells["N_cell"].sum())
    r_total = int(cells["R_cell"].sum())
    cli_output.print_kv(
        "Cells",
        [
            ("n_cells", len(cells)),
            ("n_total (livebirths)", f"{n_total:,}"),
            ("r_total (recorded DS)", f"{r_total:,}"),
            ("recorded_rate", f"{r_total / n_total:.2e}"),
            ("year_range", cells.attrs.get("year_range")),
        ],
    )

    cli_output.section("Build model")
    priors = CoreReductionPriors.from_reduction_csv(
        year_range=ns.year_range,
        path=ns.reduction_csv,
        observed_logit_sigma=ns.observed_reduction_sigma,
        extrapolated_logit_sigma=ns.extrapolated_reduction_sigma,
        extrapolated_start=ns.extrapolated_start,
        recording_s_mean=ns.recording_s_mean,
        recording_s_sigma=ns.recording_s_sigma,
        recording_s_year_sigma=ns.recording_s_year_sigma,
    )
    cli_output.info(
        "reduction prior: "
        f"observed sigma={ns.observed_reduction_sigma}, "
        f"extrapolated sigma={ns.extrapolated_reduction_sigma} "
        f"from {ns.extrapolated_start}"
    )
    if ns.model_definition.recording_model == "year":
        cli_output.info(
            "recording sensitivity: partially pooled s_year offsets "
            f"(sigma={ns.recording_s_year_sigma})"
        )
    else:
        cli_output.info("recording sensitivity: constant s")
    model = build_core_reduction_model(
        cells,
        priors,
        n_year=cells.attrs["n_year"],
        recording_model=ns.model_definition.recording_model,
    )

    cli_output.section("Sample")
    idata = sample(model, config=run_config, prior_only=ns.prior_only)

    cli_output.section("Save artefacts")
    model_config = CoreReductionModelConfig.from_priors(
        year_range=ns.year_range,
        priors_obj=priors,
        model_definition=ns.model_definition,
        notes=f"profile={ns.profile}",
    )
    context = FitContext(
        config=model_config,
        run_config=run_config,
        output_dir=ns.output_dir,
        cells=cells,
        model=model,
        idata=idata,
    )
    save_artefacts(context, ns.output_dir)
    cli_output.success(f"idata.nc, cells.parquet, config.json -> {ns.output_dir}")
    qmd_path = copy_docs_template(ns.model_definition.template_id, ns.output_dir)
    if qmd_path is not None:
        cli_output.success(f"index.qmd -> {qmd_path}")
    else:
        cli_output.warning(
            "No Quarto template at "
            f"docs/models/{ns.model_definition.template_id}/index.qmd."
        )

    if ns.prior_only:
        cli_output.info("Prior-only run - skipping posterior summary / report render.")
        return 0

    cli_output.section("Posterior summary")
    summary_vars = [
        "rho_year",
        "eta_year",
        "recording_s",
        "recording_s_year",
        "true_count_year",
        "true_count_total",
    ]
    if "recording_s_year_offset" in idata.posterior:
        summary_vars.insert(4, "recording_s_year_offset")
    summary = diagnostics.summary_table(
        idata,
        var_names=tuple(summary_vars),
    )
    save_summary(summary, ns.output_dir)
    year_summary = core_year_summary(idata, cells)
    year_summary.to_csv(ns.output_dir / "year_summary.csv", index=False)
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
        cli_output.warning("Convergence checks did not all pass - inspect summary.csv.")

    cli_output.section("Report outputs")
    render_core_all(
        idata,
        cells,
        ns.output_dir,
        priors_config=priors.to_dict(),
        year_range=ns.year_range,
        recording_model=ns.model_definition.recording_model,
    )
    cli_output.success(f"plots/ and tables/ -> {ns.output_dir}")
    render_report(qmd_path, do_render=ns.render)

    cli_output.section("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
