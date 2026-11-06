"""Import CDC natality files into a separate, validated DuckDB candidate."""

from __future__ import annotations

import argparse
from pathlib import Path

from dse_research_utils.environment.setup import init_script

from dspopulations_us_birth_certificates.ingest.combine import combine_years
from dspopulations_us_birth_certificates.ingest.database import import_year
from dspopulations_us_birth_certificates.ingest.layouts import ROOT, load_layout
from dspopulations_us_birth_certificates.ingest.published_validation import (
    load_controls,
    validate_published,
)
from dspopulations_us_birth_certificates.ingest.validation import compare_database
from dspopulations_us_birth_certificates.ingest.years import parse_years


def main() -> int:
    init_script()
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--year", type=int, help="Build one annual database")
    selection.add_argument(
        "--years", type=parse_years, help="Build and combine a range, e.g. 1989-2025"
    )
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--staging-dir", type=Path, default=ROOT / "data/staging/natality"
    )
    parser.add_argument("--baseline", type=Path, default=ROOT / "data/us_births.db")
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--memory-limit", default="2GB")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Rebuild incomplete or changed managed candidates",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate existing annual databases without importing again",
    )
    parser.add_argument(
        "--combine-only",
        action="store_true",
        help="Combine existing validated annual databases",
    )
    args = parser.parse_args()
    if args.combine_only and (args.year is not None or args.validate_only):
        parser.error(
            "--combine-only requires --years and cannot be used with --validate-only"
        )
    years = args.years or [args.year]
    layouts = [load_layout(year) for year in years]
    for layout in layouts:
        mode = layout.source.get("validation_mode", "legacy_comparison")
        if mode == "published_controls":
            load_controls(layout)
        elif mode != "legacy_comparison":
            parser.error(f"Unsupported validation mode: {mode}")
    annual = {}
    for layout in layouts:
        year = layout.year
        mode = layout.source.get("validation_mode", "legacy_comparison")
        cache = (
            args.cache / str(year) if args.years and args.cache else args.cache
        ) or ROOT / "data/raw/natality" / str(year)
        output = (
            args.output if args.year is not None else None
        ) or args.staging_dir / str(year) / "us_births.db"
        annual[year] = output
        if args.combine_only:
            continue
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
                f"Source checks passed for {year}: {result['rows']:,} records.",
                flush=True,
            )
        validate = (
            validate_published if mode == "published_controls" else compare_database
        )
        report = validate(
            layout,
            output,
            args.baseline,
            memory_limit=args.memory_limit,
            threads=args.threads,
        )
        print(
            f"Validation {'passed' if report['passed'] else 'FAILED'}: {output.with_suffix('.validation.json')}",
            flush=True,
        )
        if not report["passed"]:
            return 1
        if (
            mode == "published_controls"
            and not report["reference_coverage"]["all_reference_inputs_available"]
        ):
            print(
                f"External model inputs are incomplete. Coverage report: {output.with_suffix('.coverage.json')}"
            )
    if args.years:
        output = args.output or args.staging_dir / "us_births.db"
        result = combine_years(
            annual,
            output,
            args.baseline,
            memory_limit=args.memory_limit,
            threads=args.threads,
            restart=args.restart,
        )
        print(f"Combined validation passed: {result['rows']:,} records in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
