"""Fit isolated prior stress tests using the same recorded-count likelihood."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd
from dse_research_utils.environment.setup import init_script

from dspopulations_us_birth_certificates.file_io import (
    write_atomically,
    write_text_atomically,
)
from dspopulations_us_birth_certificates.paths import resolve_output_root
from dspopulations_us_birth_certificates.selection import (
    FitContext,
    SelectionModelConfig,
    build_model,
    copy_docs_template,
    diagnostics,
    prepare_cells,
    sample,
    save_artefacts,
    save_summary,
    selection_run_config,
)
from dspopulations_us_birth_certificates.selection.fit_validation import (
    validate_fit,
    write_validation,
)
from dspopulations_us_birth_certificates.selection.render import (
    RenderOptions,
    render_all,
)
from dspopulations_us_birth_certificates.selection.sensitivity import (
    SCENARIOS,
    add_headline_quantities,
    annual_summary,
    combine_likelihood_cells,
    predictive_summary,
    scenario_priors,
    total_summary,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS)
    )
    parser.add_argument("--duckdb-path", type=Path, default=Path("data/us_births.db"))
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--draws", type=int, default=1500)
    parser.add_argument("--tune", type=int, default=1500)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.95)
    parser.add_argument("--random-seed", type=int, default=67)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    if not 2016 <= args.start_year <= args.end_year <= 2024:
        parser.error("Use a year range within 2016-2024")
    if (
        args.chains < 2
        or min(args.draws, args.tune) < 1
        or not 0 < args.target_accept < 1
    ):
        parser.error(
            "Use at least two chains, positive draws/tuning and target acceptance in (0, 1)"
        )
    if len(set(args.scenarios)) != len(args.scenarios):
        parser.error("Do not repeat scenarios")
    args.output_dir = args.output_dir or resolve_output_root(
        args.output_root
    ) / "selection_sensitivity" / datetime.now().strftime("%Y%m%d-%H%M%S")
    return args


def main(argv=None):
    args = parse_args(argv)
    init_script()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    print(f"Sensitivity outputs: {output}", flush=True)
    with duckdb.connect(str(args.duckdb_path), read_only=True) as connection:
        original = prepare_cells(
            connection, year_range=(args.start_year, args.end_year)
        )
    cells = combine_likelihood_cells(original)
    original.to_parquet(output / "original_cells.parquet", index=False)
    print(
        f"Combined {len(original)} original strata into {len(cells)} likelihood cells",
        flush=True,
    )
    settings = replace(
        selection_run_config("reporting", random_seed=args.random_seed),
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        target_accept=args.target_accept,
        prior_predictive_samples=100,
    )
    results = []
    statuses = {}
    write_text_atomically(output / "status.json", json.dumps({"status": "running"}))
    for scenario in args.scenarios:
        run = output / scenario
        run.mkdir()
        write_validation(run, {"status": "sampling", "scope": "not_validated"})
        priors = scenario_priors(scenario)
        config = SelectionModelConfig.from_priors(
            variant="C",
            spec="full",
            year_range=(args.start_year, args.end_year),
            priors_obj=priors,
            notes=f"Isolated stress test: {scenario}. Identical recorded-count target. Clinical strata combined; no clinical-stratum predictive check. No additional prevalence-margin observation.",
        )
        print(f"Starting {scenario}", flush=True)
        model = build_model(
            cells,
            priors,
            spec="full",
            n_year=args.end_year - args.start_year + 1,
            start_year=args.start_year,
        )
        add_headline_quantities(model, cells)
        idata = sample(model, config=settings)
        summary, validation = validate_fit(idata, model)
        statuses[scenario] = validation["status"]
        write_validation(run, validation)
        context = FitContext(
            config=config,
            run_config=settings,
            output_dir=run,
            cells=cells,
            model=model,
            idata=idata,
            metrics={"validation_status": validation["status"]},
        )
        save_artefacts(context, run)
        save_summary(summary, run)
        anchor = diagnostics.s_anchor_shrinkage_table(
            idata, priors_config=config.priors, year_range=config.year_range
        )
        write_atomically(
            run / "recording_anchor.csv",
            lambda path, anchor=anchor: anchor.to_csv(path, index=False),
        )
        table = annual_summary(idata, cells, scenario, args.start_year)
        table["validation_status"] = validation["status"]
        write_atomically(
            run / "annual_summary.csv",
            lambda path, table=table: table.to_csv(path, index=False),
        )
        results.append(table)
        for name, data in (
            ("predictive_checks", predictive_summary(idata, cells, scenario)),
            (
                "totals",
                total_summary(idata, cells, scenario, priors.false_positive_rate),
            ),
        ):
            write_atomically(
                run / f"{name}.csv",
                lambda path, data=data: data.to_csv(path, index=False),
            )
        render_all(
            idata,
            cells,
            run,
            options=RenderOptions(
                priors_config=config.priors, year_range=config.year_range
            ),
        )
        copy_docs_template("selection", run)
        write_text_atomically(
            run / "scenario.json",
            json.dumps(
                {
                    "scenario": scenario,
                    "validation_status": validation["status"],
                    "clinical_checks": "not_run_after_combining_strata",
                },
                indent=2,
            ),
        )
        pd_all = pd.concat(results, ignore_index=True)
        write_atomically(
            output / "annual_comparison.csv",
            lambda path, pd_all=pd_all: pd_all.to_csv(path, index=False),
        )
        print(f"Finished {scenario}: validation={validation['status']}", flush=True)
        del idata, model, context
    passed = all(status == "passed" for status in statuses.values())
    write_text_atomically(
        output / "status.json",
        json.dumps(
            {"status": "passed" if passed else "failed", "scenarios": statuses},
            indent=2,
        ),
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
