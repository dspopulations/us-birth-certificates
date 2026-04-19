"""Tune a registered ModelDefinition with Optuna.

Separates hyperparameter search from fitting so the expensive Optuna
loop only runs when the feature set or search space changes. Writes:

- output/tuning/<model_id>/best_params.json
- output/tuning/<model_id>/trials.csv
- output/tuning/<model_id>/study.pkl

Copy the best_params.json contents into the model's ``params`` class
attribute (reviewable commit); ``fit_model.py --no-optimize --model-id
<id>`` then uses those params.

Examples
--------
    python scripts/tune_model.py usbc10_m1 --profile dev
    python scripts/tune_model.py usbc10_m2 --n-trials 500 --timeout 3600
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from dspopulations_us_birth_certificates import cli_output, repl_utils, tuning
from dspopulations_us_birth_certificates.models import MODELS, RunConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "model_id",
        choices=sorted(MODELS.keys()),
        help="Model definition to tune.",
    )
    p.add_argument(
        "--profile",
        choices=list(RunConfig.preset_names()),
        default="reporting",
        help="RunConfig preset (default: reporting).",
    )
    p.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Override the profile's n_trials (optional).",
    )
    p.add_argument(
        "--num-boost-round",
        type=int,
        default=None,
        help="Override the profile's num_boost_round.",
    )
    p.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=None,
        help="Override the profile's early_stopping_rounds.",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Wall-time budget for the study, in seconds.",
    )
    p.add_argument("--random-seed", type=int, default=47)
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/tuning"),
        help="Root directory for tuning artefacts.",
    )
    p.add_argument("--duckdb-path", type=Path, default=Path("data/us_births.db"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    ns = parse_args(argv)

    out_dir = ns.output_root / ns.model_id
    cli_output.print_run_header(
        command="tune_model",
        profile=ns.profile,
        output_dir=out_dir,
        model_id=ns.model_id,
    )

    cli_output.section("Environment")
    repl_utils.print_environment_info()

    definition = MODELS[ns.model_id]
    base = RunConfig.from_name(ns.profile, random_seed=ns.random_seed)
    run_config = replace(
        base,
        num_boost_round=ns.num_boost_round or base.num_boost_round,
        early_stopping_rounds=(
            ns.early_stopping_rounds or base.early_stopping_rounds
        ),
    )

    cli_output.print_run_config(run_config)

    tuning.run_optuna_study(
        definition,
        run_config,
        n_trials=ns.n_trials,
        timeout=ns.timeout,
        output_root=ns.output_root,
        db_path=str(ns.duckdb_path),
    )

    cli_output.section("Done")
    cli_output.success(f"Artefacts written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
