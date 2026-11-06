"""Export selected CDC import years to optional local Parquet files."""

import argparse
from pathlib import Path

from dse_research_utils.environment.setup import init_script

from dspopulations_us_birth_certificates.ingest.export import export_parquet
from dspopulations_us_birth_certificates.ingest.years import parse_years


def main() -> None:
    init_script()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database", type=Path, default=Path("data/staging/natality/us_births.db")
    )
    parser.add_argument("--years", type=parse_years, required=True)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path, help="One merged Parquet file")
    destination.add_argument(
        "--annual-dir", type=Path, help="Write us_births_<year>.parquet files"
    )
    parser.add_argument(
        "--all-columns",
        action="store_true",
        help="Include derived columns and reporting flags",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--memory-limit", default="2GB")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    outputs = (
        [(args.output, args.years)]
        if args.output
        else [
            (args.annual_dir / f"us_births_{year}.parquet", [year])
            for year in args.years
        ]
    )
    for output, years in outputs:
        rows = export_parquet(
            args.database,
            output,
            years,
            all_columns=args.all_columns,
            overwrite=args.overwrite,
            memory_limit=args.memory_limit,
            threads=args.threads,
        )
        print(f"Exported {rows:,} records to {output}")


if __name__ == "__main__":
    main()
