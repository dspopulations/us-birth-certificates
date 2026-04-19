"""Unit tests for outcome SQL builders in ``bayes.outcomes``."""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from dspopulations_us_birth_certificates.bayes.outcomes import (
    build_outcome_sql,
    outcome_spec_from_name,
    recorded_plus_predicted_spec,
    recorded_spec,
)


@pytest.fixture
def synthetic_births_con() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with a tiny, predictable ``us_births`` table.

    Layout: 2 years × 2 months × 20 births, with 2 recorded DS cases per
    year×month and 18 non-recorded. ``ds_pred_missing`` is pre-populated
    to flag the top 3 non-recorded (ceil(1.5 × 2) = 3) per year×month by
    ``p_ds_lb_pred_01`` — this mimics what ``scripts/fit_model.py`` does
    after populating ``p_ds_lb_pred_01``.
    """
    con = duckdb.connect(":memory:")
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
            # 18 non-recorded; descending p_ds_lb_pred_01 so the top-3 (by rank)
            # are the first three i in {0, 1, 2}.
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
    return con


def test_recorded_spec_has_no_params() -> None:
    spec = recorded_spec()
    assert spec.name == "recorded"
    assert spec.params == {}


def test_recorded_plus_predicted_spec_has_no_params() -> None:
    spec = recorded_plus_predicted_spec()
    assert spec.name == "recorded_plus_predicted"
    assert spec.params == {}


def test_outcome_spec_from_name_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown outcome name"):
        outcome_spec_from_name("nonsense")


def test_recorded_sql_counts_match_down_ind(
    synthetic_births_con: duckdb.DuckDBPyConnection,
) -> None:
    sql = build_outcome_sql(
        recorded_spec(),
        year_range=(2020, 2021),
        extra_columns=("mage_c",),
    )
    result = synthetic_births_con.execute(
        f"SELECT SUM(is_case) AS cases, COUNT(*) AS total FROM ({sql})"
    ).fetchone()
    # 2 years × 2 months × 2 recorded = 8 recorded cases; 80 total births.
    assert result == (8, 80)


def test_recorded_plus_predicted_reads_column(
    synthetic_births_con: duckdb.DuckDBPyConnection,
) -> None:
    # Fixture pre-flags top 3 non-recorded per year×month. Per cell:
    # 2 recorded + 3 predicted = 5 cases. 2 years × 2 months × 5 = 20 cases,
    # 80 total births.
    sql = build_outcome_sql(
        recorded_plus_predicted_spec(),
        year_range=(2020, 2021),
        extra_columns=("mage_c",),
    )
    result = synthetic_births_con.execute(
        f"SELECT SUM(is_case) AS cases, COUNT(*) AS total FROM ({sql})"
    ).fetchone()
    assert result == (20, 80)


def test_recorded_plus_predicted_respects_year_range(
    synthetic_births_con: duckdb.DuckDBPyConnection,
) -> None:
    # Restrict to 2020 only: 2 months × (2 recorded + 3 predicted) = 10 cases,
    # 40 total births.
    sql = build_outcome_sql(
        recorded_plus_predicted_spec(),
        year_range=(2020, 2020),
        extra_columns=("mage_c",),
    )
    result = synthetic_births_con.execute(
        f"SELECT SUM(is_case) AS cases, COUNT(*) AS total FROM ({sql})"
    ).fetchone()
    assert result == (10, 40)


def test_build_outcome_sql_rejects_unknown_spec() -> None:
    from dspopulations_us_birth_certificates.bayes.outcomes import OutcomeSpec

    bad = OutcomeSpec(name="bogus", params={})
    with pytest.raises(ValueError, match="Unknown outcome spec"):
        build_outcome_sql(bad, year_range=(2020, 2021), extra_columns=())
