"""Create the intermediate DuckDB from data/us_births.parquet.

Writes to ``data/us_births_temp.db``. The final ``data/us_births.db``
is produced by ``scripts/duckdb_prepare.py``, which reads the temp
DB, runs the column-derivation pipeline, then COPYs FROM DATABASE
into ``us_births.db``. The full pipeline is:

    import_parquet -> combine_parquet -> prepare_parquet
    -> duckdb_create -> duckdb_prepare
"""

import pathlib

import duckdb


def create_temp_db() -> None:
    src_dir = pathlib.Path("data")
    source_parquet = src_dir / "us_births.parquet"
    out_db_temp = src_dir / "us_births_temp.db"
    out_db_temp.unlink(missing_ok=True)

    con = duckdb.connect(out_db_temp.as_posix())

    print("--------------------------------------------------------------")
    print(f"Importing Parquet file into DuckDB '{out_db_temp}'...")
    print("--------------------------------------------------------------")

    try:
        print(f"Reading Parquet file '{source_parquet}'...")

        con.execute(
            """
            CREATE TABLE us_births AS
            SELECT *
            FROM read_parquet(?)
            """,
            [source_parquet.as_posix()],
        )

    finally:
        con.close()


if __name__ == "__main__":
    create_temp_db()
