"""Tests for ``selection.data.prepare_cells`` on a synthetic DuckDB.

Mirrors ``tests/test_bayes_data.py``: the fixture builds a tiny in-memory
(-ish) DuckDB with only the columns ``prepare_cells`` reads, so the tests
don't depend on the 7 GB real database being present.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from dspopulations_us_birth_certificates.selection import prepare_cells
from dspopulations_us_birth_certificates.selection.data import (
    DEFAULT_COLUMNS,
    EDU_MAP,
    EDU_UNKNOWN_IDX,
    PAYER_MAP,
    PAYER_UNKNOWN_IDX,
    RACE_MAP,
    RACE_UNKNOWN_IDX,
    _largest_remainder_round,
)


def _make_row(
    *,
    year: int,
    mage_c: int,
    mracehisp_c: int | None = 1,
    meduc: int | None = 3,
    pay_rec: int | None = 2,
    gestrec10: int | None = 7,
    ca_cchd: str | None = "N",
    ab_nicu: str | None = "N",
    ab_aven1: str | None = "N",
    down_ind: int | None = 0,
) -> dict:
    return {
        "year": year,
        "mage_c": mage_c,
        "mracehisp_c": mracehisp_c,
        "meduc": meduc,
        "pay_rec": pay_rec,
        "gestrec10": gestrec10,
        "ca_cchd": ca_cchd,
        "ab_nicu": ab_nicu,
        "ab_aven1": ab_aven1,
        "down_ind": down_ind,
    }


@pytest.fixture
def tiny_db(tmp_path: Path) -> Path:
    """Write a minimal ``us_births`` DuckDB for schema-level tests."""
    rows: list[dict] = []
    # A spread of age bands (covers every AGE index 0-6).
    ages_by_bin = {
        0: 18,
        1: 22,
        2: 27,
        3: 32,
        4: 37,
        5: 42,
        6: 47,
    }
    for year in (2016, 2020, 2024):
        for age in ages_by_bin.values():
            for race in (1, 2, 3, 4, 5, None):
                for edu in (1, 2, 3, 4, 5, 6, 7, 8, 9, None):
                    for pay in (1, 2, 3, 4, None):
                        rows.append(
                            _make_row(
                                year=year,
                                mage_c=age,
                                mracehisp_c=race,
                                meduc=edu,
                                pay_rec=pay,
                            )
                        )
    # Add a handful of DS-positive rows and unknown-gestation / unknown-flag
    # rows that should be dropped.
    rows.append(_make_row(year=2020, mage_c=40, down_ind=1))
    rows.append(
        _make_row(year=2020, mage_c=40, gestrec10=99, down_ind=1)
    )  # dropped
    rows.append(
        _make_row(year=2020, mage_c=40, ca_cchd="U", down_ind=1)
    )  # dropped
    rows.append(_make_row(year=2015, mage_c=40))  # out of range
    rows.append(_make_row(year=2025, mage_c=40))  # out of range
    rows.append(_make_row(year=2020, mage_c=None))  # mage null
    df = pd.DataFrame(rows)

    db_path = tmp_path / "tiny.db"
    con = duckdb.connect(str(db_path))
    con.register("_rows", df)
    con.execute("CREATE TABLE us_births AS SELECT * FROM _rows")
    con.unregister("_rows")
    con.close()
    return db_path


def test_prepare_cells_column_schema(tiny_db: Path) -> None:
    con = duckdb.connect(str(tiny_db), read_only=True)
    try:
        cells = prepare_cells(con, year_range=(2016, 2024))
    finally:
        con.close()

    expected = {
        "year_idx",
        "age_idx",
        "race_idx",
        "edu_idx",
        "payer_idx",
        "preterm",
        "cchd",
        "nicu",
        "aven",
        "N_cell",
        "R_cell",
    }
    assert expected.issubset(cells.columns)
    assert "region_idx" not in cells.columns
    assert "n_region" not in cells.attrs
    assert cells.attrs["n_year"] == 9
    assert cells.attrs["year_range"] == (2016, 2024)


def test_largest_remainder_round_preserves_total() -> None:
    # Many sub-0.5 values: independent per-cell rounding floors them all to 0 and
    # loses the total; largest-remainder must preserve the rounded grand total.
    x = np.array([0.3, 0.3, 0.4, 2.4, 0.6])  # sum 4.0
    out = _largest_remainder_round(x)
    assert out.sum() == round(x.sum()) == 4
    assert out.dtype == np.int64
    assert (out >= np.floor(x)).all() and (out <= np.ceil(x)).all()
    # integer input returned unchanged
    assert (_largest_remainder_round(np.array([1.0, 2.0, 3.0])) == [1, 2, 3]).all()


def test_prepare_cells_prob_sum(tmp_path: Path) -> None:
    """predictions_column gives calibrated expected DS = recorded + sum(prob | unrec)."""
    rows = [
        _make_row(year=2020, mage_c=30, down_ind=1),  # recorded -> contributes 1
        _make_row(year=2020, mage_c=30, down_ind=1),  # recorded -> 1
        _make_row(year=2020, mage_c=30, down_ind=0),  # unrecorded -> prob 0.7
        _make_row(year=2020, mage_c=30, down_ind=0),  # unrecorded -> prob 0.3
    ]
    df = pd.DataFrame(rows)
    df["p_ds_lb_pred_14"] = [0.9, 0.1, 0.7, 0.3]  # recorded preds ignored (CASE -> 1)
    db = tmp_path / "p.db"
    con = duckdb.connect(str(db))
    con.register("_r", df)
    con.execute("CREATE TABLE us_births AS SELECT * FROM _r")
    con.unregister("_r")
    con.close()

    con = duckdb.connect(str(db), read_only=True)
    try:
        cells = prepare_cells(
            con, year_range=(2020, 2020), predictions_column="p_ds_lb_pred_14"
        )
        # expected = 2 recorded + (0.7 + 0.3 over unrecorded) = 3
        assert cells["R_cell"].sum() == 3
        assert (cells["R_cell"] <= cells["N_cell"]).all()
        with pytest.raises(ValueError, match="at most one"):
            prepare_cells(
                con,
                predictions_column="p_ds_lb_pred_14",
                missing_flag_column="ds_pred_missing_14",
            )
    finally:
        con.close()


def test_prepare_cells_year_filter(tiny_db: Path) -> None:
    """Year filter drops out-of-range rows."""
    con = duckdb.connect(str(tiny_db), read_only=True)
    try:
        cells = prepare_cells(con, year_range=(2020, 2020))
    finally:
        con.close()
    # All year_idx values should be 0 (2020 - 2020).
    assert set(cells["year_idx"].unique()) == {0}


def test_prepare_cells_drops_unknown_gestation_and_flags(
    tiny_db: Path,
) -> None:
    """Rows with unknown gestation / ca_cchd / ab_nicu / ab_aven1 are dropped."""
    con = duckdb.connect(str(tiny_db), read_only=True)
    try:
        cells = prepare_cells(con, year_range=(2016, 2024))
    finally:
        con.close()
    for col in ("preterm", "cchd", "nicu", "aven"):
        assert set(cells[col].unique()).issubset({0, 1})


def test_prepare_cells_emits_unknown_levels(tiny_db: Path) -> None:
    """Unknown race/edu/payer map to their sentinel indices."""
    con = duckdb.connect(str(tiny_db), read_only=True)
    try:
        cells = prepare_cells(con, year_range=(2016, 2024))
    finally:
        con.close()
    assert RACE_UNKNOWN_IDX in set(cells["race_idx"].unique())
    assert EDU_UNKNOWN_IDX in set(cells["edu_idx"].unique())
    assert PAYER_UNKNOWN_IDX in set(cells["payer_idx"].unique())


def test_prepare_cells_aggregates_counts(tmp_path: Path) -> None:
    """Two identical births aggregate into one cell with N_cell=2."""
    df = pd.DataFrame(
        [
            _make_row(year=2020, mage_c=30),
            _make_row(year=2020, mage_c=30),
            _make_row(year=2020, mage_c=30, down_ind=1),
        ]
    )
    db_path = tmp_path / "agg.db"
    con = duckdb.connect(str(db_path))
    con.register("_rows", df)
    con.execute("CREATE TABLE us_births AS SELECT * FROM _rows")
    con.unregister("_rows")
    con.close()

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        cells = prepare_cells(con, year_range=(2016, 2024))
    finally:
        con.close()
    # All three rows share a covariate profile → one cell with N=3, R=1.
    assert len(cells) == 1
    assert int(cells["N_cell"].iloc[0]) == 3
    assert int(cells["R_cell"].iloc[0]) == 1


def test_prepare_cells_preterm_derivation(tmp_path: Path) -> None:
    """gestrec10 1-5 -> preterm=1; 6-10 -> preterm=0; 99 dropped."""
    df = pd.DataFrame(
        [
            _make_row(year=2020, mage_c=30, gestrec10=3),  # preterm
            _make_row(year=2020, mage_c=30, gestrec10=7),  # term
            _make_row(year=2020, mage_c=30, gestrec10=99),  # dropped
        ]
    )
    db_path = tmp_path / "preterm.db"
    con = duckdb.connect(str(db_path))
    con.register("_rows", df)
    con.execute("CREATE TABLE us_births AS SELECT * FROM _rows")
    con.unregister("_rows")
    con.close()

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        cells = prepare_cells(con, year_range=(2016, 2024))
    finally:
        con.close()
    assert len(cells) == 2
    assert set(cells["preterm"].unique()) == {0, 1}


def test_age_binning_covers_all_indices(tmp_path: Path) -> None:
    """Every age index 0-6 is reachable from a single representative age."""
    df = pd.DataFrame(
        [
            _make_row(year=2020, mage_c=18),  # <20 -> 0
            _make_row(year=2020, mage_c=22),  # 20-24 -> 1
            _make_row(year=2020, mage_c=27),  # 25-29 -> 2
            _make_row(year=2020, mage_c=32),  # 30-34 -> 3
            _make_row(year=2020, mage_c=37),  # 35-39 -> 4
            _make_row(year=2020, mage_c=42),  # 40-44 -> 5
            _make_row(year=2020, mage_c=47),  # 45+ -> 6
        ]
    )
    db_path = tmp_path / "ages.db"
    con = duckdb.connect(str(db_path))
    con.register("_rows", df)
    con.execute("CREATE TABLE us_births AS SELECT * FROM _rows")
    con.unregister("_rows")
    con.close()

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        cells = prepare_cells(con, year_range=(2016, 2024))
    finally:
        con.close()
    assert sorted(cells["age_idx"].unique().tolist()) == [0, 1, 2, 3, 4, 5, 6]


def test_column_alias_override(tmp_path: Path) -> None:
    """``columns=`` override lets callers handle schema drift without SQL edits."""
    df = pd.DataFrame([_make_row(year=2020, mage_c=30)])
    df = df.rename(columns={"ca_cchd": "cchd_renamed"})

    db_path = tmp_path / "alias.db"
    con = duckdb.connect(str(db_path))
    con.register("_rows", df)
    con.execute("CREATE TABLE us_births AS SELECT * FROM _rows")
    con.unregister("_rows")
    con.close()

    override = {**DEFAULT_COLUMNS, "ca_cchd": "cchd_renamed"}
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        cells = prepare_cells(con, year_range=(2016, 2024), columns=override)
    finally:
        con.close()
    assert len(cells) == 1


def test_code_maps_are_complete() -> None:
    """Each map covers every non-Unknown raw code its SQL branches on."""
    assert set(RACE_MAP) == {1, 2, 3, 4, 5}
    assert set(EDU_MAP) == set(range(1, 9))
    assert set(PAYER_MAP) == {1, 2, 3, 4}
    assert np.array_equal(
        sorted(RACE_MAP.values()), sorted({0, 1, 2, 3, 4})
    )
