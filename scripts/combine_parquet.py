"""Combine Parquet files."""

import pathlib
import re

import polars as pl

_ANNUAL_PARQUET_RE = re.compile(r"^us_births_(\d{4})\.parquet$")


def _annual_parquet_paths(src_dir: pathlib.Path) -> list[pathlib.Path]:
    """Return annual source parquet files, excluding combined outputs."""
    return sorted(
        p
        for p in src_dir.glob("us_births_*.parquet")
        if _ANNUAL_PARQUET_RE.match(p.name)
    )


def combine_all() -> None:
    src_dir = pathlib.Path("data")
    out_parquet = src_dir / "us_births_combined.parquet"
    out_parquet.unlink(missing_ok=True)

    paths = _annual_parquet_paths(src_dir)

    if not paths:
        raise FileNotFoundError(f"No input Parquet files found in {src_dir.resolve()}")

    print(f"Combining {len(paths)} Parquet files...")

    lfs = [pl.scan_parquet(p) for p in paths]
    combined = pl.concat(lfs, how="diagonal_relaxed")
    combined.sink_parquet(out_parquet.as_posix())

    print(f"Wrote: {out_parquet}")


if __name__ == "__main__":
    combine_all()
