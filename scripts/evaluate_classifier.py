"""Evaluate fresh classifiers with nested cross-validation of the recorded label.

Uses all original USBC10 candidate features, with no historical feature pruning
or saved tuned parameters. Results describe this training procedure, rather
than retrospectively validating the published M1 models or true DS diagnoses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from dse_research_utils.environment.setup import init_script
from dse_research_utils.metadata.provenance import (
    git_snapshot,
    package_versions,
    sha256_file,
)

from dspopulations_us_birth_certificates.data_utils import load_predictors_data
from dspopulations_us_birth_certificates.evaluation import (
    NestedCVConfig,
    nested_cross_validate,
)
from dspopulations_us_birth_certificates.file_io import (
    write_atomically,
    write_text_atomically,
)
from dspopulations_us_birth_certificates.models import MODELS
from dspopulations_us_birth_certificates.paths import resolve_output_root


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label",
        choices=("confirmed-pending", "confirmed"),
        default="confirmed-pending",
    )
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--duckdb-path", type=Path, default=Path("data/us_births.db"))
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--num-boost-round", type=int, default=2000)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=67)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    if not 2016 <= args.start_year <= args.end_year <= 2025:
        parser.error("Use a year range within 2016-2025")
    args.output_dir = args.output_dir or (
        resolve_output_root(args.output_root)
        / "nested_classifier"
        / args.label
        / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    return args


def _write_json(path, content):
    write_text_atomically(
        path, json.dumps(content, indent=2, allow_nan=False), encoding="utf-8"
    )


def main(argv=None):
    args = parse_args(argv)
    init_script()
    config = NestedCVConfig(
        **{key: getattr(args, key) for key in NestedCVConfig.__dataclass_fields__}
    )
    definition = MODELS["usbc10_m0_cn" if args.label == "confirmed" else "usbc10_m0"]
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    print(f"Nested evaluation outputs: {output}", flush=True)
    _write_json(output / "status.json", {"status": "running"})
    snapshot = git_snapshot(Path.cwd())
    _write_json(
        output / "run_config.json",
        {
            **asdict(config),
            "label_definition": args.label,
            "year_range": [args.start_year, args.end_year],
            "output_dir": str(output.resolve()),
            "database": str(args.duckdb_path.resolve()),
            "candidate_definition": definition.model_id,
            "categorical_features": list(definition.categorical_features),
            "numeric_features": list(definition.numeric_features),
            "saved_parameters_used": False,
            "historical_pruning_used": False,
            "include_unknown_as_negative": definition.include_unknown,
            "code_revision": snapshot.commit,
            "source_sha256": {
                str(path): sha256_file(path)
                for root in (Path("src"), Path("scripts"))
                for path in sorted(root.rglob("*.py"))
            },
            "lockfile_sha256": sha256_file("uv.lock"),
            "packages": package_versions(
                ("lightgbm", "optuna", "scikit-learn", "pandas", "numpy")
            ),
            "interpretation": "Nested evaluation of a fresh original-feature training procedure; not validation of prior M1 fits or true disease status. Random folds estimate within-period performance, not future-year transport.",
        },
    )
    data = (
        load_predictors_data(
            from_year=args.start_year,
            to_year=args.end_year,
            confirmed_only=definition.confirmed_only,
            include_unknown=definition.include_unknown,
            db_path=str(args.duckdb_path),
        )
        .sort_values("id")
        .reset_index(drop=True)
    )
    if not data["id"].is_unique:
        raise ValueError("Duplicate birth IDs would leak records across folds")
    ids_hash = hashlib.sha256(data["id"].to_numpy().tobytes()).hexdigest()
    features = list(definition.categorical_features) + list(definition.numeric_features)
    X = data[features].copy()
    y = data[definition.target_var].fillna(0).astype("int32")
    _write_json(
        output / "cohort.json",
        {
            "n": len(data),
            "n_positive": int(y.sum()),
            "ordered_id_sha256": ids_hash,
            "years": {str(k): int(v) for k, v in data.groupby("year").size().items()},
        },
    )
    del data

    def save_fold(result, study):
        fold_dir = output / f"fold_{result['fold']}"
        fold_dir.mkdir()
        _write_json(fold_dir / "metrics.json", result)
        write_atomically(
            fold_dir / "trials.csv",
            lambda path: study.trials_dataframe().to_csv(path, index=False),
        )
        print(
            f"Fold {result['fold']} complete: AP={result['average_precision']:.6f}, test n={result['n']}",
            flush=True,
        )

    folds, pooled = nested_cross_validate(
        X,
        y,
        categorical=tuple(definition.categorical_features),
        base_params=definition.base_params,
        config=config,
        on_fold=save_fold,
    )
    write_atomically(
        output / "fold_metrics.csv", lambda path: folds.to_csv(path, index=False)
    )
    _write_json(output / "metrics.json", pooled)
    _write_json(output / "status.json", {"status": "complete"})
    print(f"Nested evaluation complete: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
