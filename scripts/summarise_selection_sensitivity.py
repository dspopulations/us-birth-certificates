"""Export aggregate evidence from saved three-stage sensitivity fits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import xarray as xr
from dse_research_utils.environment.setup import init_script
from dse_research_utils.metadata.provenance import sha256_file

from dspopulations_us_birth_certificates.file_io import (
    write_atomically,
    write_text_atomically,
)
from dspopulations_us_birth_certificates.selection.sensitivity import (
    SCENARIOS,
    annual_summary,
    predictive_summary,
    total_summary,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_dir", type=Path)
    parser.add_argument(
        "--replacement-study-dir",
        type=Path,
        nargs="+",
        default=[],
        help="Use completed reruns from this directory when present; retain original validation in provenance",
    )
    parser.add_argument(
        "--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS)
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    init_script()
    output = args.output_dir or args.study_dir / "summary"
    output.mkdir(parents=True, exist_ok=False)
    tables = {
        name: []
        for name in (
            "annual",
            "totals",
            "predictive_checks",
            "recording_anchor",
            "validation",
        )
    }
    provenance = {}
    reference_cells = None
    reference_years = None
    for scenario in args.scenarios:
        run = args.study_dir / scenario
        initial_validation = json.loads((run / "validation.json").read_text())
        reruns = []
        for replacement in args.replacement_study_dir:
            if (replacement / scenario).is_dir():
                run = replacement / scenario
                reruns.append(
                    {
                        "directory": str(run),
                        "validation": json.loads((run / "validation.json").read_text()),
                        "run_config": json.loads((run / "run_config.json").read_text()),
                    }
                )
        if not (run / "scenario.json").is_file():
            raise ValueError(f"{scenario} has not finished saving its artefacts")
        config = json.loads((run / "config.json").read_text())
        settings = json.loads((run / "run_config.json").read_text())
        validation = json.loads((run / "validation.json").read_text())
        if validation["status"] not in ("passed", "failed"):
            raise ValueError(f"{scenario} has not completed numerical validation")
        cells = pd.read_parquet(run / "cells.parquet")
        if reference_cells is None:
            reference_cells, reference_years = cells, config["year_range"]
        elif (
            not cells.equals(reference_cells) or config["year_range"] != reference_years
        ):
            raise ValueError(f"{scenario} uses different likelihood cells or years")
        # Open only the groups needed for aggregate summaries, without loading
        # prior draws or cell-level posterior probabilities into memory.
        with xr.open_datatree(run / "idata.nc") as idata:
            tables["annual"].append(
                annual_summary(idata, cells, scenario, config["year_range"][0])
            )
            tables["totals"].append(
                total_summary(
                    idata, cells, scenario, config["priors"]["false_positive_rate"]
                )
            )
            tables["predictive_checks"].append(
                predictive_summary(idata, cells, scenario)
            )
        anchor = pd.read_csv(run / "recording_anchor.csv")
        anchor.insert(0, "scenario", scenario)
        tables["recording_anchor"].append(anchor)
        tables["validation"].append(
            pd.DataFrame(
                [
                    {
                        "scenario": scenario,
                        "status": validation["status"],
                        "initial_status": initial_validation["status"],
                        "initial_max_rhat": initial_validation.get(
                            "convergence", {}
                        ).get("max_rhat"),
                        "max_rhat": validation["convergence"]["max_rhat"],
                        "min_ess": validation["convergence"]["min_ess"],
                        "divergences": validation["divergences"],
                        "min_bfmi": min(validation["bfmi"], default=None),
                        "failures": "; ".join(validation["failures"]),
                    }
                ]
            )
        )
        provenance[scenario] = {
            "initial_validation": initial_validation,
            "reruns": reruns,
            "config": config,
            "run_config": settings,
            "validation": validation,
            "fit_manifest": json.loads((run / "manifest.json").read_text()),
        }
        print(f"Summarised {scenario}: {validation['status']}", flush=True)
    for name, frames in tables.items():
        table = pd.concat(frames, ignore_index=True)
        if name != "validation":
            table["validation_status"] = table.scenario.map(
                {
                    key: value["validation"]["status"]
                    for key, value in provenance.items()
                }
            )
        write_atomically(
            output / f"{name}.csv",
            lambda path, table=table: table.to_csv(path, index=False),
        )
    write_text_atomically(
        output / "provenance.json",
        json.dumps(
            {
                "scenarios": provenance,
                "summary_sha256": {
                    path.name: sha256_file(path)
                    for path in sorted(output.glob("*.csv"))
                },
            },
            indent=2,
            allow_nan=False,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
