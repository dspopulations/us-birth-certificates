"""Tests for annual parquet source discovery in ``scripts/combine_parquet.py``."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import combine_parquet  # noqa: E402


def test_annual_parquet_paths_excludes_combined_outputs(tmp_path: Path) -> None:
    for name in (
        "us_births_2023.parquet",
        "us_births_2024.parquet",
        "us_births_all.parquet",
        "us_births_combined.parquet",
        "us_births.parquet",
    ):
        (tmp_path / name).touch()

    paths = combine_parquet._annual_parquet_paths(tmp_path)

    assert [p.name for p in paths] == [
        "us_births_2023.parquet",
        "us_births_2024.parquet",
    ]
