"""Run reproducible, joint simulation-calibration experiments for DSP models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from dse_research_utils.environment.setup import init_script

from dspopulations_us_birth_certificates.selection.config import FitContext, RunConfig
from dspopulations_us_birth_certificates.selection.core_simulation import (
    calibration_table,
    simulate_core,
    simulation_design,
)
from dspopulations_us_birth_certificates.selection.fit_validation import (
    validate_fit,
    write_validation,
)
from dspopulations_us_birth_certificates.selection.io import save_artefacts
from dspopulations_us_birth_certificates.selection.sampling import sample


def main(argv=None) -> int:
    init_script()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", default=["DSP003", "DSP008", "DSP009", "DSP010"]
    )
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.replicates, args.draws, args.tune) < 1:
        parser.error("replicates, draws and tune must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("use an empty output directory to avoid mixing calibration runs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results, statuses = [], []
    for model_id in args.models:
        cells, specification = simulation_design(model_id)
        for replicate in range(args.replicates):
            seed = args.seed + replicate * 2
            simulated, spec, truth = simulate_core(cells, specification, seed=seed)
            model = spec.build(simulated)
            run = RunConfig(
                "reporting",
                draws=args.draws,
                tune=args.tune,
                chains=4,
                target_accept=0.98,
                prior_predictive_samples=100,
                posterior_predictive=True,
                nuts_sampler="nutpie",
                random_seed=seed + 1,
            )
            idata = sample(model, config=run)
            summary, health = validate_fit(idata, model)
            directory = args.output_dir / model_id / str(seed)
            write_validation(directory, health)
            save_artefacts(
                FitContext(
                    spec.to_config(),
                    run,
                    cells=simulated,
                    model=model,
                    idata=idata,
                    metrics={"validation_status": health["status"]},
                ),
                directory,
            )
            summary.to_csv(directory / "summary.csv")
            table = calibration_table(idata, truth, seed=seed, model_id=model_id)
            table["numerical_validation"] = health["status"]
            table.to_csv(directory / "calibration.csv", index=False)
            results.append(table)
            statuses.append(
                {"model": model_id, "seed": seed, "status": health["status"]}
            )
            pd.concat(results, ignore_index=True).to_csv(
                args.output_dir / "calibration.csv", index=False
            )
            (args.output_dir / "experiment.json").write_text(
                json.dumps(
                    {
                        "models": args.models,
                        "replicates_per_model": args.replicates,
                        "seed": args.seed,
                        "completed": statuses,
                        "interpretation": "regression_pilot"
                        if args.replicates < 100
                        else "inspect_rank_calibration_and_monte_carlo_error",
                        "caution": "Do not count correlated cells or years as independent simulation repetitions.",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
    return 0 if all(row["status"] == "passed" for row in statuses) else 2


if __name__ == "__main__":
    raise SystemExit(main())
