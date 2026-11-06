"""Export the historical 2014-2024 modelling subset from the CDC database."""

from pathlib import Path

from dse_research_utils.environment.setup import init_script

from dspopulations_us_birth_certificates.ingest.export import export_parquet


def merge_years():
    """Keep the established output name without loading whole years into pandas."""
    return export_parquet(
        Path("data/staging/natality/us_births.db"),
        Path("data/us_births_all.parquet"),
        list(range(2014, 2025)),
        overwrite=True,
    )


def main():
    init_script()
    merge_years()


if __name__ == "__main__":
    main()
