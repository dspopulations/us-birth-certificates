"""Legacy Parquet database preparation. The CDC pilot never calls compaction."""

import pathlib
import shutil

import duckdb

from dspopulations_us_birth_certificates.preparation import (
    _add_id_column,
    _retype_columns,
    derive_columns,
)


def _compact_database(out_db_temp: pathlib.Path, out_db: pathlib.Path) -> None:
    """Copy the working DB into the final compacted DB, then reset the temp copy."""
    out_db.unlink(missing_ok=True)

    print("Copying from temp_db to db...")

    duckdb.execute(
        f"""
        ATTACH '{out_db_temp.as_posix()}' AS temp_db;
        ATTACH '{out_db.as_posix()}' AS db;
        COPY FROM DATABASE temp_db TO db;
        DETACH temp_db;
        DETACH db;
        """
    )

    print("Copying from db to temp_db...")

    out_db_temp.unlink(missing_ok=True)

    # copy out_db file to out_db_temp overwriting
    shutil.copy2(out_db.as_posix(), out_db_temp.as_posix())


def combine_all() -> None:
    src_dir = pathlib.Path("data")
    out_db_temp = src_dir / "us_births_temp.db"
    out_db = src_dir / "us_births.db"

    con = duckdb.connect(out_db_temp.as_posix())

    print("--------------------------------------------------------------")
    print(f"Preparing DuckDB '{out_db_temp}'...")
    print("--------------------------------------------------------------")

    try:
        _retype_columns(con)
        _add_id_column(con)
        derive_columns(con)
    finally:
        print("Closing connection...")
        con.close()

    _compact_database(out_db_temp, out_db)


if __name__ == "__main__":
    combine_all()
