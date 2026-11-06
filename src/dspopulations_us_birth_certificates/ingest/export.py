"""Optional Parquet exports from a read-only natality database."""

from pathlib import Path

import duckdb

from dspopulations_us_birth_certificates.file_io import write_atomically
from dspopulations_us_birth_certificates.variables import IMPORTED_VARS

from .layouts import ROOT, identifier


def export_parquet(
    database: Path,
    output: Path,
    years: list[int],
    *,
    all_columns: bool = False,
    overwrite: bool = False,
    memory_limit: str = "2GB",
    threads: int = 2,
) -> int:
    """Write selected years in bounded memory and replace only after checking."""
    for protected in (
        database,
        ROOT / "data/us_births.db",
        ROOT / "data/us_births_temp.db",
    ):
        if output.resolve() == protected.resolve() or (
            output.exists() and protected.exists() and output.samefile(protected)
        ):
            raise ValueError(f"Export would replace a protected database: {protected}")
    if output.suffix != ".parquet" or not years or threads < 1:
        raise ValueError(
            "A .parquet output, selected years and positive threads are required"
        )
    if output.exists() and not overwrite:
        raise FileExistsError(f"Export already exists: {output}")
    selected = sorted(set(years))
    if any(type(year) is not int or not 1989 <= year <= 2025 for year in selected):
        raise ValueError("Export years must be integers between 1989 and 2025")
    where = "year IN (" + ",".join(str(year) for year in selected) + ")"
    projection = "*" if all_columns else ", ".join(identifier(n) for n in IMPORTED_VARS)
    output.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(database), read_only=True) as con:
        con.execute("SET memory_limit = ?", [memory_limit])
        con.execute("SET threads = ?", [threads])
        con.execute("SET preserve_insertion_order = false")
        counts = dict(
            con.execute(
                f"SELECT year, count(*) FROM us_births WHERE {where} GROUP BY year"
            ).fetchall()
        )
        if set(counts) != set(selected):
            raise ValueError(
                f"Database is missing requested years: {sorted(set(selected) - set(counts))}"
            )

        def write(temporary: Path) -> None:
            con.execute(
                f"COPY (SELECT {projection} FROM us_births WHERE {where}) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",
                [str(temporary)],
            )
            stored = con.execute(
                "SELECT count(*) FROM read_parquet(?)", [str(temporary)]
            ).fetchone()[0]
            if stored != sum(counts.values()):
                raise ValueError("Parquet row count differs from the source database")

        write_atomically(output, write)
    return sum(counts.values())
