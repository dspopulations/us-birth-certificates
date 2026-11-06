"""Build and validate a separate CDC natality pilot. Never replace the live DB."""

from __future__ import annotations

import argparse
from pathlib import Path

from dse_research_utils.environment.setup import init_script

from dspopulations_us_birth_certificates.ingest.database import import_year
from dspopulations_us_birth_certificates.ingest.layouts import ROOT, load_layout
from dspopulations_us_birth_certificates.ingest.published_validation import (
    load_controls,
    validate_published,
)
from dspopulations_us_birth_certificates.ingest.validation import compare_database


def main() -> int:
    init_script()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline", type=Path, default=ROOT / "data/us_births.db")
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--memory-limit", default="2GB")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Rebuild an incomplete or changed managed pilot",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Compare an already source-validated pilot",
    )
    args = parser.parse_args()
    layout = load_layout(args.year)
    mode = layout.source.get("validation_mode", "legacy_comparison")
    if mode == "published_controls":
        load_controls(layout)
    elif mode != "legacy_comparison":
        parser.error(f"Unsupported validation mode: {mode}")
    cache = args.cache or ROOT / "data/raw/natality" / str(args.year)
    output = (
        args.output or ROOT / "data/staging/natality" / str(args.year) / "us_births.db"
    )
    if not args.validate_only:
        result = import_year(
            layout,
            cache=cache,
            output=output,
            baseline=args.baseline,
            batch_size=args.batch_size,
            memory_limit=args.memory_limit,
            threads=args.threads,
            restart=args.restart,
        )
        print(
            f"Source checks passed for {result['rows']:,} records. Validation mode: {mode}.",
            flush=True,
        )
    validate = validate_published if mode == "published_controls" else compare_database
    report = validate(
        layout,
        output,
        args.baseline,
        memory_limit=args.memory_limit,
        threads=args.threads,
    )
    print(
        f"Validation {'passed' if report['passed'] else 'FAILED'}: {output.with_suffix('.validation.json')}"
    )
    if (
        mode == "published_controls"
        and not report["reference_coverage"]["all_reference_inputs_available"]
    ):
        print(
            f"External model inputs are incomplete. Coverage report: {output.with_suffix('.coverage.json')}"
        )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
