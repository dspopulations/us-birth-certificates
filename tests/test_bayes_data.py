"""Tests for ``bayes.data.load_cells`` on a synthetic DuckDB fixture."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from dspopulations_us_birth_certificates.bayes.data import load_cells


@pytest.fixture
def synthetic_duckdb(tmp_path: Path) -> Path:
    """Write a tiny DuckDB file with an ``us_births`` table and return its path.

    Pre-populates ``ds_pred_missing`` at the 1.5× year×month multiplier so
    the ``recorded_plus_predicted`` path has what it needs — this matches
    what ``scripts/fit_model.py`` writes after computing predictions.
    """
    db_path = tmp_path / "synthetic.db"
    con = duckdb.connect(str(db_path))
    rows = []
    birth_id = 0
    for year in (2020, 2021):
        for month in (1, 2):
            for i in range(2):
                birth_id += 1
                rows.append(
                    {
                        "id": birth_id,
                        "year": year,
                        "dob_mm": month,
                        "down_ind": 1,
                        "p_ds_lb_pred_01": 0.01,
                        "mage_c": 30 + i,
                        "ds_pred_missing": False,
                    }
                )
            for i in range(18):
                birth_id += 1
                rows.append(
                    {
                        "id": birth_id,
                        "year": year,
                        "dob_mm": month,
                        "down_ind": 0,
                        "p_ds_lb_pred_01": 0.09 - 0.001 * i,
                        "mage_c": 20 + i,
                        "ds_pred_missing": i < 3,
                    }
                )
    df = pd.DataFrame(rows)
    con.register("_births_df", df)
    con.execute("CREATE TABLE us_births AS SELECT * FROM _births_df")
    con.unregister("_births_df")
    con.close()
    return db_path


def test_load_cells_recorded_schema(synthetic_duckdb: Path) -> None:
    cells = load_cells(
        outcome="recorded",
        dims=("year", "mage_c"),
        year_range=(2020, 2021),
        db_path=synthetic_duckdb,
    )
    assert list(cells.columns) == ["year", "mage_c", "n_cell", "y_cell"]
    assert cells["n_cell"].dtype.name == "int64"
    assert cells["y_cell"].dtype.name == "int64"
    assert int(cells["n_cell"].sum()) == 80
    assert int(cells["y_cell"].sum()) == 8


def test_load_cells_prepends_year_dim(synthetic_duckdb: Path) -> None:
    cells = load_cells(
        outcome="recorded",
        dims=("mage_c",),
        year_range=(2020, 2021),
        db_path=synthetic_duckdb,
    )
    assert "year" in cells.columns
    assert cells.columns[0] == "year"


def test_load_cells_recorded_plus_predicted(
    synthetic_duckdb: Path,
) -> None:
    cells = load_cells(
        outcome="recorded_plus_predicted",
        dims=("year", "mage_c"),
        year_range=(2020, 2021),
        db_path=synthetic_duckdb,
    )
    # 2 recorded + 3 predicted per year×month, 4 cells → 20 cases; exposure 80.
    assert int(cells["n_cell"].sum()) == 80
    assert int(cells["y_cell"].sum()) == 20


def test_load_cells_rejects_empty_dims(synthetic_duckdb: Path) -> None:
    with pytest.raises(ValueError, match="dims must be non-empty"):
        load_cells(
            outcome="recorded",
            dims=(),
            year_range=(2020, 2021),
            db_path=synthetic_duckdb,
        )
