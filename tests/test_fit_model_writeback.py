"""Tests for prediction writeback in ``scripts/fit_model.py``."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "fit_model.py"


def _load_fit_model_module():
    spec = importlib.util.spec_from_file_location("fit_model_cli", SCRIPT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fit_model_cli"] = mod
    spec.loader.exec_module(mod)
    return mod


class _FakeBooster:
    best_iteration = 1

    def predict(self, X, num_iteration=None):  # noqa: ANN001
        return np.array([0.1, 0.9, 0.2], dtype=float)


def test_write_predictions_clears_stale_values_before_flagging(tmp_path: Path) -> None:
    mod = _load_fit_model_module()
    db_path = tmp_path / "births.db"
    existing = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "year": [2024, 2024, 2024, 2023],
            "dob_mm": [1, 1, 1, 1],
            "down_ind": [1, 0, 0, 0],
            "f1": [0.0, 0.0, 0.0, 0.0],
            "p_test_pred": [0.4, 0.3, 0.2, 0.99],
            "test_missing": [False, False, False, True],
        }
    )
    con = duckdb.connect(str(db_path))
    con.register("_rows", existing)
    con.execute("CREATE TABLE us_births AS SELECT * FROM _rows")
    con.unregister("_rows")
    con.close()

    current = existing.loc[:2, ["id", "f1"]].copy()
    mod.write_predictions_to_duckdb(
        current,
        _FakeBooster(),
        ["f1"],
        [],
        db_path,
        predictions_column="p_test_pred",
        missing_flag_column="test_missing",
    )

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        got = con.execute(
            """
            SELECT id, p_test_pred, test_missing
            FROM us_births
            ORDER BY id
            """
        ).fetchdf()
    finally:
        con.close()

    assert got.loc[got["id"] == 4, "p_test_pred"].isna().all()
    assert not bool(got.loc[got["id"] == 4, "test_missing"].iloc[0])
    assert got.loc[got["id"].isin([2, 3]), "test_missing"].tolist() == [
        True,
        True,
    ]
