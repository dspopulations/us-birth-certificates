"""Exports data to SPSS format and zips SPSS files."""

import zipfile
from datetime import datetime

import pandas as pd
import pyreadstat as stats
from dse_research_utils.environment.setup import init_script


def read_for_spss(source):
    """Keep the CDC decimal BMI column numeric in the SPSS output."""
    df = pd.read_parquet(source)
    if "bmi" in df:
        # pandas reads DuckDB DECIMAL as Python Decimal objects. Without this
        # conversion, pyreadstat writes BMI as a string variable in SPSS.
        df["bmi"] = df["bmi"].astype("float64")
    return df


def export_spss():
    "Exports data to SPSS format."

    now = datetime.now().strftime("%Y%m%d_%H%M")

    sources = [
        "data/us_births_2014.parquet",
        "data/us_births_2015.parquet",
        "data/us_births_2016.parquet",
        "data/us_births_2017.parquet",
        "data/us_births_2018.parquet",
        "data/us_births_2019.parquet",
        "data/us_births_2020.parquet",
        "data/us_births_2021.parquet",
        "data/us_births_2022.parquet",
        "data/us_births_all.parquet",
    ]

    for source in sources:
        dest = source.replace(".parquet", f"_{now}.sav")
        zip_dest = source.replace(".parquet", f"_{now}.zip")

        print(f"Reading {source}...")

        df = read_for_spss(source)

        print(f"Exporting to {dest}...")

        stats.write_sav(df, dest)

        print(f"Compressing to {zip_dest}...")

        with zipfile.ZipFile(
            zip_dest, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as myzip:
            myzip.write(dest)


def main():
    init_script()
    export_spss()


if __name__ == "__main__":
    main()
