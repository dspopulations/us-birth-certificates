"""Tests for the read-only core reduction measurement/anchor audit."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_core_reduction_assumptions.py"


def _load_audit_module():
    spec = importlib.util.spec_from_file_location(
        "audit_core_reduction_assumptions_cli", SCRIPT_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["audit_core_reduction_assumptions_cli"] = module
    spec.loader.exec_module(module)
    return module


AUDIT = _load_audit_module()


def test_age_band_indices_match_core_boundaries() -> None:
    ages = np.array([19, 20, 24, 25, 29, 30, 34, 35, 39, 40, 44, 45])
    assert AUDIT.age_band_indices(ages).tolist() == [
        0,
        1,
        1,
        2,
        2,
        3,
        3,
        4,
        4,
        5,
        5,
        6,
    ]


def test_measurement_tables_report_cp_morris_and_false_positive_scales() -> None:
    counts = pd.DataFrame(
        {
            "age": [19, 20, 45],
            "births": [25_000, 50_000, 25_000],
            "confirmed": [15, 40, 25],
            "pending": [5, 10, 5],
        }
    )
    exact, summary = AUDIT.measurement_anchor_tables(
        counts,
        false_positive_rate=7.8e-5,
        target_fp_share_among_recorded=0.078,
    )

    overall = summary.loc[summary["scope"] == "overall"].iloc[0]
    assert overall["births"] == 100_000
    assert overall["confirmed"] == 80
    assert overall["pending"] == 20
    assert overall["recorded_cp"] == 100
    assert overall["confirmed_share_cp"] == pytest.approx(0.8)
    assert overall["pending_share_cp"] == pytest.approx(0.2)
    assert overall["expected_fp_all_births_upper_approx"] == pytest.approx(7.8)
    assert overall["implied_fp_share_recorded_all_births_approx"] == pytest.approx(
        0.078
    )
    assert overall["calibrated_f_all_births_for_target_share"] == pytest.approx(7.8e-5)
    assert overall["calibrated_f_natural_non_ds_proxy_for_target_share"] > 7.8e-5

    expected_exact = (
        counts["births"]
        * AUDIT.get_ds_lb_nt_probability_array(counts["age"].to_numpy())
    ).sum()
    assert overall["exact_expected_ds"] == pytest.approx(expected_exact)
    assert exact["band_expected_ds"].sum() == pytest.approx(overall["band_expected_ds"])
    assert set(summary.loc[summary["scope"] == "age_band", "age_label"]) == {
        "<20",
        "20-24",
        "45+",
    }


def test_calibrated_false_positive_rate_requires_population_rate() -> None:
    calibrated = AUDIT.calibrated_false_positive_rate(
        target_share_among_recorded=0.078,
        recorded_count=100,
        non_ds_exposure=100_000,
    )
    assert calibrated == pytest.approx(7.8e-5)

    with pytest.raises(ValueError, match="non_ds_exposure"):
        AUDIT.calibrated_false_positive_rate(
            target_share_among_recorded=0.078,
            recorded_count=100,
            non_ds_exposure=0,
        )


def test_identified_product_table_uses_exact_observation_identity() -> None:
    cells = pd.DataFrame(
        {
            "age_idx": [0, 0, 1],
            "N_cell": [10_000, 10_000, 10_000],
            "R_cell": [5, 5, 8],
        }
    )
    theta = np.array([0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007])
    ppc = pd.DataFrame(
        {
            "age_idx": [0, 1],
            "observed": [10, 8],
            "predicted_mean": [8.0, 9.0],
            "predicted_lo": [5.0, 6.0],
            "predicted_hi": [11.0, 12.0],
        }
    )
    table = AUDIT.identified_product_age_table(
        cells,
        theta_age=theta,
        false_positive_rate=1e-4,
        target_fp_share_among_recorded=0.10,
        ppc_by_age=ppc,
    )

    age0 = table.loc[table["age_idx"] == 0].iloc[0]
    assert age0["age_model"] == "band"
    assert age0["scope"] == "age_band"
    assert age0["age_label"] == "<20"
    assert not age0["maternal_age_endpoint_capped"]
    assert age0["observed_rate"] == pytest.approx(5e-4)
    assert age0["identified_eta_times_s_minus_f"] == pytest.approx(0.4)
    assert age0["implied_fp_share_recorded_all_births_approx"] == pytest.approx(0.2)
    assert age0["calibrated_f_all_births_for_target_share"] == pytest.approx(5e-5)
    assert age0["observed_minus_predicted"] == pytest.approx(2.0)

    overall = table.loc[table["scope"] == "overall"].iloc[0]
    expected_product = (18 - 30_000 * 1e-4) / (20_000 * theta[0] + 10_000 * theta[1])
    assert overall["identified_eta_times_s_minus_f"] == pytest.approx(expected_product)


def test_identified_product_table_supports_exact_age_indices_above_six() -> None:
    cells = pd.DataFrame(
        {
            "age_idx": [0, 0, 7, 8],
            "maternal_age": [12, 12, 40, 50],
            "N_cell": [4_000, 6_000, 10_000, 10_000],
            "R_cell": [2, 3, 20, 35],
        }
    )
    table = AUDIT.identified_product_age_table(
        cells,
        false_positive_rate=1e-4,
        age_model="single_year",
    )

    exact = table.loc[table["scope"] == "exact_age"].set_index("maternal_age")
    assert set(exact.index) == {12.0, 40.0, 50.0}
    assert exact.loc[12.0, "age_label"] == "10-12"
    assert exact.loc[50.0, "age_label"] == "50+"
    assert exact.loc[12.0, "maternal_age_endpoint_capped"]
    assert exact.loc[50.0, "maternal_age_endpoint_capped"]
    assert not exact.loc[40.0, "maternal_age_endpoint_capped"]
    assert exact.loc[50.0, "theta_age"] == pytest.approx(
        AUDIT.get_ds_lb_nt_probability_array(np.array([50]))[0]
    )
    expected_product = (35 / 10_000 - 1e-4) / exact.loc[50.0, "theta_age"]
    assert exact.loc[50.0, "identified_eta_times_s_minus_f"] == pytest.approx(
        expected_product
    )


def test_single_year_config_requires_represented_maternal_age() -> None:
    cells = pd.DataFrame({"age_idx": [7], "N_cell": [10], "R_cell": [1]})
    with pytest.raises(ValueError, match="cells.maternal_age"):
        AUDIT.identified_product_age_table(cells, age_model="single_year")


def test_cli_writes_machine_readable_audit_without_database_writes(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "births.db"
    rows = pd.DataFrame(
        {
            "year": [2020, 2020, 2020, 2021, 2021, 2021, 2019],
            "mage_c": [19, 20, 20, 45, 45, 30, 30],
            "ca_down_c": ["C", "P", "N", "C", "P", "U", "C"],
            "down_ind": [1, 1, 0, 1, 1, None, 1],
        }
    )
    con = duckdb.connect(str(db_path))
    con.register("_rows", rows)
    con.execute("CREATE TABLE us_births AS SELECT * FROM _rows")
    con.unregister("_rows")
    con.close()

    fit_dir = tmp_path / "fit"
    tables_dir = fit_dir / "tables"
    tables_dir.mkdir(parents=True)
    cells = pd.DataFrame(
        {
            "year_idx": [0, 0, 1],
            "age_idx": [0, 1, 2],
            "maternal_age": [12, 20, 50],
            "N_cell": [1, 2, 2],
            "R_cell": [1, 1, 2],
        }
    )
    cells.to_parquet(fit_dir / "cells.parquet", index=False)
    (fit_dir / "config.json").write_text(
        json.dumps(
            {
                "year_range": [2020, 2021],
                "age_model": "single_year",
                "priors": {
                    "theta_lb_age": AUDIT.CURRENT_BAND_THETA.tolist(),
                    "false_positive_rate": 7.8e-5,
                },
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "age_idx": [0, 1, 2],
            "observed": [1, 1, 2],
            "predicted_mean": [0.8, 1.1, 1.5],
        }
    ).to_csv(tables_dir / "core_ppc_by_age.csv", index=False)

    out_dir = tmp_path / "audit"
    assert (
        AUDIT.main(
            [
                "--duckdb-path",
                str(db_path),
                "--fit-dir",
                str(fit_dir),
                "--target-fp-share",
                "0.078",
                "--output-dir",
                str(out_dir),
            ]
        )
        == 0
    )

    audit_csv = pd.read_csv(out_dir / "audit.csv")
    assert set(audit_csv["section"]) == {
        "measurement_by_exact_age",
        "measurement_summary",
        "identified_product_by_age",
    }
    payload = json.loads((out_dir / "audit.json").read_text(encoding="utf-8"))
    assert payload["metadata"]["year_range"] == [2020, 2021]
    assert payload["metadata"]["target_fp_share_among_recorded"] == 0.078
    assert payload["metadata"]["identified_product_age_model"] == "single_year"
    assert (
        "capped representatives" in payload["metadata"]["maternal_age_endpoint_warning"]
    )
    assert (
        "population and era" in payload["metadata"]["false_positive_calibration_caveat"]
    )
    assert set(payload["sections"]) == {
        "measurement_by_exact_age",
        "measurement_summary",
        "identified_product_by_age",
    }
    identified = payload["sections"]["identified_product_by_age"]
    exact_labels = {
        row["age_label"] for row in identified if row["scope"] == "exact_age"
    }
    assert {"10-12", "20", "50+"} <= exact_labels
    measurement = payload["sections"]["measurement_summary"]
    overall = next(row for row in measurement if row["scope"] == "overall")
    assert overall["births_age_known"] == 6
    assert overall["births"] == 5
    assert overall["excluded_unknown_down_ind"] == 1

    check = duckdb.connect(str(db_path), read_only=True)
    try:
        assert check.execute("SHOW TABLES").fetchall() == [("us_births",)]
        assert check.execute("SELECT COUNT(*) FROM us_births").fetchone() == (7,)
    finally:
        check.close()


def test_cli_rejects_invalid_false_positive_inputs() -> None:
    with pytest.raises(ValueError, match="false_positive_rate"):
        AUDIT.resolve_false_positive_rate({}, 1.0)
    with pytest.raises(ValueError, match="table identifier"):
        AUDIT.load_exact_age_counts("does-not-matter.db", table="bad; DROP TABLE")
