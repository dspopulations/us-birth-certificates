"""Tests for the read-only DSP004 race-surveillance audit."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import pandas as pd
import pytest
import xarray as xr

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_core_race_surveillance.py"


def _load_audit_module():
    spec = importlib.util.spec_from_file_location(
        "audit_core_race_surveillance_cli", SCRIPT_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


AUDIT = _load_audit_module()

YEARS = tuple(range(2016, 2025))
SOURCE_YEARS = tuple(range(2014, 2025))
AGES = (12, 50)
THETA = np.array([0.001, 0.02])
N_BY_YEAR_AGE = np.array(
    [[1000 - 40 * year_idx, 100 + 40 * year_idx] for year_idx in range(len(YEARS))]
)
R_BY_YEAR_AGE = np.array([[1, 3 + year_idx] for year_idx in range(len(YEARS))])
RACE_PERCENT = np.array([40, 20, 10, 10, 10, 5, 5])


def _config() -> dict[str, object]:
    return {
        "model_id": "DSP004",
        "age_model": "single_year",
        "theta_model": "morris_double_logistic_by_age_code",
        "recording_model": "constant",
        "reduction_model": "year",
        "recorded_definition": "confirmed_or_pending",
        "year_range": [2016, 2024],
        "age_endpoint_convention": {
            "12": "10-12; Morris evaluated at age 12",
            "50": "50+; Morris evaluated at age 50",
        },
        "priors": {
            "theta_lb_age_used": False,
            "false_positive_rate": 0.000078,
            "reduction_error_correlation": 0.0,
            "reduction_calibration_shift_logit": 0.0,
        },
    }


def _saved_cells() -> pd.DataFrame:
    rows = []
    for year_idx, _year in enumerate(YEARS):
        for age_idx, age in enumerate(AGES):
            rows.append(
                {
                    "year_idx": year_idx,
                    "age_idx": age_idx,
                    "maternal_age": age,
                    "N_cell": int(N_BY_YEAR_AGE[year_idx, age_idx]),
                    "R_cell": int(R_BY_YEAR_AGE[year_idx, age_idx]),
                }
            )
    return pd.DataFrame(rows)


def _posterior_idata(
    cells: pd.DataFrame,
    *,
    divergent: bool = False,
):
    rng = np.random.default_rng(104729)
    n_chain = 4
    n_draw = 600
    eta = np.clip(
        rng.normal(
            loc=np.linspace(0.8, 0.6, len(YEARS)),
            scale=0.035,
            size=(n_chain, n_draw, len(YEARS)),
        ),
        0.01,
        0.99,
    )
    rho = 1.0 - eta
    year_idx = cells["year_idx"].to_numpy(dtype=int)
    age_idx = cells["age_idx"].to_numpy(dtype=int)
    p_ds = eta[:, :, year_idx] * THETA[age_idx][None, None, :]
    natural_year = np.array(
        [
            np.sum(
                N_BY_YEAR_AGE[year_idx_value] * THETA,
                dtype=float,
            )
            for year_idx_value in range(len(YEARS))
        ]
    )
    true_year = eta * natural_year[None, None, :]
    posterior = xr.Dataset(
        {
            "eta_year": (("chain", "draw", "year"), eta),
            "rho_year": (("chain", "draw", "year"), rho),
            "recording_s": (
                ("chain", "draw"),
                np.clip(
                    rng.normal(0.4, 0.02, size=(n_chain, n_draw)),
                    0.01,
                    0.99,
                ),
            ),
            "recording_s_year": (
                ("chain", "draw", "year"),
                np.clip(
                    rng.normal(
                        0.4,
                        0.02,
                        size=(n_chain, n_draw, len(YEARS)),
                    ),
                    0.01,
                    0.99,
                ),
            ),
            "p_ds_lb": (("chain", "draw", "cell"), p_ds),
            "true_count_year": (("chain", "draw", "year"), true_year),
            "true_count_total": (
                ("chain", "draw"),
                true_year.sum(axis=2),
            ),
        },
        coords={
            "chain": np.arange(n_chain),
            "draw": np.arange(n_draw),
            "year": np.arange(len(YEARS)),
            "cell": np.arange(len(cells)),
        },
    )
    constant_data = xr.Dataset(
        {"theta_lb_age": (("age",), THETA)},
        coords={"age": list(AGES)},
    )
    divergences = np.zeros((n_chain, n_draw), dtype=bool)
    divergences[0, 0] = divergent
    sample_stats = xr.Dataset(
        {"diverging": (("chain", "draw"), divergences)},
        coords={"chain": np.arange(n_chain), "draw": np.arange(n_draw)},
    )
    return xr.DataTree.from_dict(
        {
            "posterior": posterior,
            "constant_data": constant_data,
            "sample_stats": sample_stats,
        }
    )


def _write_fit(run_dir: Path, *, divergent: bool = False) -> None:
    run_dir.mkdir()
    cells = _saved_cells()
    (run_dir / "config.json").write_text(json.dumps(_config()), encoding="utf-8")
    (run_dir / "run_config.json").write_text(
        json.dumps({"name": "reporting"}), encoding="utf-8"
    )
    cells.to_parquet(run_dir / "cells.parquet", index=False)
    idata = _posterior_idata(cells, divergent=divergent)
    idata.to_netcdf(run_dir / "idata.nc")
    AUDIT.az.summary(
        idata,
        var_names=list(AUDIT.HEALTH_SUMMARY_VARIABLES),
        ci_prob=AUDIT.DEFAULT_ETI_PROB,
        ci_kind="hdi",
        round_to="none",
    ).to_csv(run_dir / "summary.csv")


def _race_cells() -> pd.DataFrame:
    rows = []
    for year_idx, year in enumerate(YEARS):
        for age_idx, age in enumerate(AGES):
            n_total = int(N_BY_YEAR_AGE[year_idx, age_idx])
            for race_idx, percent in enumerate(RACE_PERCENT):
                rows.append(
                    {
                        "year": year,
                        "year_idx": year_idx,
                        "maternal_age": age,
                        "race_idx": race_idx,
                        "race": AUDIT.RACE_LEVELS[race_idx],
                        "N_cell": n_total * int(percent) // 100,
                        "R_cell": (
                            int(R_BY_YEAR_AGE[year_idx, age_idx])
                            if race_idx == 0
                            else 0
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _surveillance_source() -> pd.DataFrame:
    source_labels = {1: "nhw", 2: "nhb", 3: "ai/an", 4: "as/pi", 5: "his"}
    rows = []
    for year in SOURCE_YEARS:
        for code in range(1, 6):
            births = 10_000 + 100 * code + 10 * (year - YEARS[0])
            recorded = 10 + code
            recording_fraction = 0.5
            estimated_true = recorded / recording_fraction
            raw = 8.0 + code + 0.25 * (year - YEARS[0])
            rows.append(
                {
                    "year": year,
                    "race": source_labels[code],
                    "mracehisp_c": code,
                    "recorded_bc": recorded,
                    "births_bc": births,
                    "bc_prev_per10k": 1e4 * recorded / births,
                    "recording_frac_g": recording_fraction,
                    "est_true_count": estimated_true,
                    "est_true_prev_per10k": 1e4 * estimated_true / births,
                    "surveillance_prev_per10k": (
                        raw if year in (2016, 2018) else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def fit(tmp_path: Path):
    run_dir = tmp_path / "fit"
    _write_fit(run_dir)
    return AUDIT.load_fit_input(run_dir)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("model_id", "DSP005"),
        ("age_model", "band"),
        ("theta_model", "age_band_lookup"),
        ("recording_model", "year"),
        ("reduction_model", "year_age"),
        ("recorded_definition", "confirmed_only"),
    ],
)
def test_fit_config_is_dsp004_specific(field: str, bad_value: str) -> None:
    config = _config()
    config[field] = bad_value

    with pytest.raises(ValueError, match=field):
        AUDIT._validate_fit_config(config)


def test_load_fit_input_fails_closed_on_health_gate(tmp_path: Path) -> None:
    run_dir = tmp_path / "fit"
    _write_fit(run_dir)
    loaded = AUDIT.load_fit_input(run_dir)
    assert loaded.fit_health["health_gate_passed"]

    unhealthy_dir = tmp_path / "unhealthy-fit"
    _write_fit(unhealthy_dir, divergent=True)

    with pytest.raises(ValueError, match="scientific health gate"):
        AUDIT.load_fit_input(unhealthy_dir)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("reduction_error_correlation", 0.5),
        ("reduction_calibration_shift_logit", -0.2),
    ],
)
def test_fit_config_requires_unshifted_independent_baseline(
    field: str,
    bad_value: float,
) -> None:
    config = _config()
    priors = config["priors"]
    assert isinstance(priors, dict)
    priors[field] = bad_value

    with pytest.raises(ValueError, match=field):
        AUDIT._validate_fit_config(config)


def test_fit_config_requires_frozen_age_and_false_positive_contract() -> None:
    config = _config()
    priors = config["priors"]
    assert isinstance(priors, dict)
    priors["false_positive_rate"] = 0.0
    with pytest.raises(ValueError, match="false_positive_rate"):
        AUDIT._validate_fit_config(config)

    config = _config()
    priors = config["priors"]
    assert isinstance(priors, dict)
    priors["theta_lb_age_used"] = True
    with pytest.raises(ValueError, match="theta_lb_age_used"):
        AUDIT._validate_fit_config(config)

    config = _config()
    config["year_range"] = [2016, 2023]
    with pytest.raises(ValueError, match="2016-2024"):
        AUDIT._validate_fit_config(config)

    config = _config()
    config["age_endpoint_convention"] = {"12": "changed", "50": "changed"}
    with pytest.raises(ValueError, match="exact-age contract"):
        AUDIT._validate_fit_config(config)


def test_load_fit_input_requires_reporting_profile(tmp_path: Path) -> None:
    run_dir = tmp_path / "fit"
    _write_fit(run_dir)
    (run_dir / "run_config.json").write_text(
        json.dumps({"name": "development"}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="reporting-profile fit"):
        AUDIT.load_fit_input(run_dir)


def test_load_fit_input_rejects_cached_summary_drift(tmp_path: Path) -> None:
    run_dir = tmp_path / "fit"
    _write_fit(run_dir)
    summary_path = run_dir / "summary.csv"
    summary = pd.read_csv(summary_path, index_col=0)
    summary.loc["recording_s", "mean"] += 0.01
    summary.to_csv(summary_path)

    with pytest.raises(ValueError, match="cached summary column 'mean'"):
        AUDIT.load_fit_input(run_dir)


def test_race_cohort_query_preserves_endpoints_filters_and_mapping(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "births.db"
    source = pd.DataFrame(
        {
            "year": [2016] * 7 + [2015],
            "mage_c": [10, 11, 12, 50, 54, None, 30, 30],
            "mracehisp_c": [1, 6, None, 5, 2, 1, 1, 1],
            "down_ind": [1, 0, 0, 1, 0, 1, None, 1],
        }
    )
    con = duckdb.connect(str(db_path))
    con.register("_source", source)
    con.execute("CREATE TABLE us_births AS SELECT * FROM _source")
    con.unregister("_source")
    con.close()

    cells = AUDIT.load_race_age_year_cells(db_path, year_range=(2016, 2016))

    assert cells["N_cell"].sum() == 5
    assert cells["R_cell"].sum() == 2
    assert set(cells["maternal_age"]) == {12, 50}
    assert set(cells["race_idx"]) == {0, 1, 4, 5, 6}
    assert cells.loc[cells["maternal_age"] == 12, "N_cell"].sum() == 3
    assert cells.loc[cells["maternal_age"] == 50, "N_cell"].sum() == 2

    with pytest.raises(ValueError, match="table identifier"):
        AUDIT.load_race_age_year_cells(
            db_path,
            year_range=(2016, 2016),
            table="us_births;DROP_TABLE",
        )


def test_reconstruction_is_exact_and_excludes_unsupported_races(fit) -> None:
    race_cells = _race_cells()
    reconciliation = AUDIT.reconcile_saved_cells(fit, race_cells)
    assert (reconciliation["N_cell_difference"] == 0).all()
    assert (reconciliation["R_cell_difference"] == 0).all()

    internal, identities = AUDIT.reconstruct_internal_accounting(fit, race_cells)

    assert max(identities.values()) < 1e-12
    unsupported = internal[internal["race_idx"].isin([5, 6])]
    assert not unsupported["named_surveillance_supported"].any()
    assert unsupported["model_named_case_share"].isna().all()
    named = internal[internal["named_surveillance_supported"]]
    assert named.groupby("year")[
        "model_named_case_share"
    ].sum().to_numpy() == pytest.approx(np.ones(len(YEARS)))

    first = internal.query("year == 2016 and race_idx == 0").iloc[0]
    expected_natural = 0.4 * np.sum(N_BY_YEAR_AGE[0] * THETA)
    assert first["natural_expected_ds"] == pytest.approx(expected_natural)
    eta_2016_mean = float(fit.idata.posterior["eta_year"].sel(year=0).mean())
    assert first["model_true_count_mean"] == pytest.approx(
        expected_natural * eta_2016_mean
    )


def test_reconstruction_rejects_cohort_and_posterior_identity_drift(fit) -> None:
    race_cells = _race_cells()
    bad_cells = race_cells.copy()
    bad_cells.loc[0, "N_cell"] += 1
    with pytest.raises(ValueError, match="does not reproduce saved DSP004 cells"):
        AUDIT.reconcile_saved_cells(fit, bad_cells)

    fit.idata.posterior["p_ds_lb"].values[0, 0, 0] += 1e-4
    with pytest.raises(ValueError, match="stored p_ds_lb identity failed"):
        AUDIT.reconstruct_internal_accounting(fit, race_cells)


def test_surveillance_loader_never_substitutes_filled_values(tmp_path: Path) -> None:
    path = tmp_path / "surveillance.csv"
    source = _surveillance_source()
    source.to_csv(path, index=False)

    direct, inventory = AUDIT.load_surveillance(path, years=(2016, 2018))

    assert len(direct) == 10
    assert set(direct["race_idx"]) == set(range(5))
    assert direct["surveillance_prev_per10k"].notna().all()
    assert not direct["filled_estimate_used"].any()
    assert set(direct["source_aggregation_operator"]) == {"pooled_count_ratio"}
    pooled_white_2018 = direct.query("year == 2018 and race_idx == 0").iloc[0]
    assert pooled_white_2018["source_window_years"] == "2016,2017,2018,2019,2020"
    assert pooled_white_2018["source_window_births_bc"] == 50_600
    assert pooled_white_2018["source_window_recorded_bc"] == 55
    assert pooled_white_2018["source_native_implied_true_count"] == pytest.approx(
        50_600 * 9.5 / 1e4
    )
    assert not inventory["filled_values_eligible_for_primary"].any()
    assert inventory.set_index("year").loc[2017, "raw_surveillance_rows"] == 0

    with pytest.raises(ValueError, match="not an allowed fallback"):
        AUDIT.load_surveillance(path, years=(2016, 2017))

    source.loc[0, "est_true_count"] += 1.0
    source.to_csv(path, index=False)
    with pytest.raises(ValueError, match="source algebra failed for est_true_count"):
        AUDIT.load_surveillance(path, years=(2016, 2018))


def test_surveillance_loader_requires_frozen_source_race_mapping(
    tmp_path: Path,
) -> None:
    path = tmp_path / "surveillance.csv"
    source = _surveillance_source()
    source.loc[source["mracehisp_c"] == 3, "race"] = "as/pi"
    source.to_csv(path, index=False)

    with pytest.raises(ValueError, match="race labels.*frozen code mapping"):
        AUDIT.load_surveillance(path, years=(2016, 2018))


def _centered_inputs(fit, tmp_path: Path):
    internal, _identities = AUDIT.reconstruct_internal_accounting(fit, _race_cells())
    source_path = tmp_path / "surveillance.csv"
    _surveillance_source().to_csv(source_path, index=False)
    surveillance, _inventory = AUDIT.load_surveillance(source_path, years=(2016, 2018))
    support = AUDIT.centered_window_support_table(internal, surveillance)
    return internal, surveillance, support


def test_centred_window_membership_and_race_specific_coverage(
    fit,
    tmp_path: Path,
) -> None:
    internal, surveillance, support = _centered_inputs(fit, tmp_path)

    window_2016 = support[support["label_year"] == 2016]
    assert set(window_2016["expected_fit_years"]) == {"2014,2015,2016,2017,2018"}
    assert set(window_2016["supported_fit_years"]) == {"2016,2017,2018"}
    assert set(window_2016["missing_fit_years"]) == {"2014,2015"}
    assert set(window_2016["supported_year_count"]) == {3}
    assert window_2016["coverage_fraction"].to_numpy() == pytest.approx(np.full(5, 0.6))
    assert not window_2016["window_complete"].any()

    window_2018 = support[support["label_year"] == 2018]
    assert set(window_2018["expected_fit_years"]) == {"2016,2017,2018,2019,2020"}
    assert set(window_2018["supported_fit_years"]) == {"2016,2017,2018,2019,2020"}
    assert set(window_2018["supported_year_count"]) == {5}
    assert window_2018["window_complete"].all()
    assert not window_2018["supported_fit_years"].str.contains("2021").any()

    missing_race_year = internal[
        ~((internal["year"] == 2019) & (internal["race_idx"] == 1))
    ]
    missing_support = AUDIT.centered_window_support_table(
        missing_race_year, surveillance
    )
    affected = missing_support.query("label_year == 2018 and race_idx == 1").iloc[0]
    assert not affected["window_complete"]
    assert affected["missing_fit_years"] == "2019"


def test_pooled_window_formula_is_exact(fit) -> None:
    fit.idata.posterior["eta_year"].values[...] = 1.0
    annual_prevalence = np.array([10.0, 10.0, 10.0, 10.0, 20.0])
    annual_births = np.array([100.0, 100.0, 100.0, 100.0, 600.0])
    rows = []
    for year_position, year in enumerate(range(2016, 2021)):
        for race_idx in range(len(AUDIT.RACE_LEVELS)):
            births = annual_births[year_position] if race_idx == 0 else 100.0
            prevalence = annual_prevalence[year_position] if race_idx == 0 else 10.0
            rows.append(
                {
                    "year": year,
                    "race_idx": race_idx,
                    "N_cell": births,
                    "R_cell": 0.0,
                    "natural_expected_ds": births * prevalence / 1e4,
                }
            )
    internal = pd.DataFrame(rows)

    pooled = AUDIT._pooled_window_model_draws(
        fit,
        internal,
        label_year=2018,
    )

    assert pooled["prevalence"][:, 0] == pytest.approx(16.0)
    assert pooled["count"][:, 0] == pytest.approx(1.6)
    assert 14.0 / pooled["prevalence"][0, 0] == pytest.approx(0.875)
    assert not hasattr(AUDIT, "CENTERED_AGGREGATION_CONSTRUCTIONS")


def test_centered_comparison_excludes_partial_window_and_stays_blocked(
    fit,
    tmp_path: Path,
) -> None:
    internal, surveillance, support = _centered_inputs(fit, tmp_path)
    comparison = AUDIT.build_centered_window_comparison(
        fit, internal, surveillance, support
    )
    denominator = AUDIT.pooled_window_denominator_reconciliation(internal, surveillance)
    composition = AUDIT.centered_composition_summary_table(comparison)
    uncertainty = AUDIT.centered_uncertainty_table(fit, internal, surveillance, support)
    decision = AUDIT.centered_audit_decision(
        comparison, composition, uncertainty, support, denominator
    )

    assert len(comparison) == 10
    partial = comparison[comparison["label_year"] == 2016]
    assert not partial["source_aligned_evaluable"].any()
    assert partial["model_prevalence_per10k_mean"].isna().all()
    assert set(partial["comparison_status"]) == {
        "not_estimable_missing_2014_2015_eta_draws"
    }
    complete = comparison[comparison["label_year"] == 2018]
    assert complete["source_aligned_evaluable"].all()
    assert set(complete["source_aggregation_operator"]) == {"pooled_count_ratio"}
    assert set(complete["model_aggregation_construction"]) == {
        "pooled_birth_weighted_prevalence"
    }
    assert not complete["filled_estimate_used"].any()
    expected_native = (
        complete["source_window_births_bc"] * complete["surveillance_prev_per10k"] / 1e4
    )
    np.testing.assert_allclose(
        complete["source_native_implied_true_count"], expected_native
    )
    assert len(composition) == 1
    assert len(uncertainty) == len(AUDIT.UNCERTAINTY_SCENARIOS)
    assert uncertainty["model_posterior_covariance_included"].all()
    assert denominator["group_denominator_material"].any()
    assert set(denominator["diagnostic_role"]) == {
        "pooled_window_denominator_mapping_not_calibration_ready"
    }
    assert not denominator.query("label_year == 2016")[
        "window_denominator_evaluable"
    ].any()
    assert denominator.query("label_year == 2018")["window_denominator_evaluable"].all()

    assert decision["descriptive_signal_for_time_invariant_race_layer"] is None
    assert decision["temporal_replication_evaluable"] is False
    assert decision["cross_window_transport_evaluable"] is False
    assert decision["complete_centred_window_label_years"] == [2018]
    assert decision["partial_centred_window_label_years"] == [2016]
    assert decision["source_aggregation_operator_confirmed"] == "pooled_count_ratio"
    assert decision["pooled_source_counts_available"] is True
    assert decision["pooled_source_counts_calibration_eligible"] is False
    assert decision["calibration_eligible"] is False
    assert decision["time_invariant_race_layer_fit_authorized"] is False
    assert decision["race_by_year_layer_fit_authorized"] is False
    assert decision["absolute_race_scale_fit_authorized"] is False


def test_centered_decision_reports_pooled_signal_but_never_authorizes_fit() -> None:
    composition = pd.DataFrame(
        {
            "tv_material_at_0_05": [True],
            "wrms_material_at_log_1_10": [False],
        }
    )
    comparison = pd.DataFrame(
        [
            {
                "source_aligned_evaluable": True,
                "race_idx": race_idx,
                "source_vs_model_relative_rate_ratio_to_white_mean": ratio,
            }
            for race_idx, ratio in enumerate((1.0, 1.20, 2.0, 0.80, 1.0))
        ]
    )
    uncertainty = pd.DataFrame(
        {
            "scenario": ["moderate"],
            "exceeds_conditional_reference": [True],
        }
    )
    support = pd.DataFrame(
        [
            {
                "label_year": label,
                "race_idx": race,
                "window_complete": complete,
            }
            for label, complete in ((2016, False), (2018, True))
            for race in range(5)
        ]
    )
    denominator = pd.DataFrame(
        {
            "group_denominator_material": [False],
            "named_total_denominator_material": [False],
        }
    )

    decision = AUDIT.centered_audit_decision(
        comparison, composition, uncertainty, support, denominator
    )

    assert decision["single_complete_window_pooled_descriptive_signal"] is True
    components = decision["pooled_descriptive_signal_components"]
    assert components["robust_non_aian_relative_rate_groups"] == [
        "NH Black",
        "NH Asian/Pacific Islander",
    ]
    assert "NH AIAN" not in components["robust_non_aian_relative_rate_groups"]
    assert decision["descriptive_signal_for_time_invariant_race_layer"] is None
    assert decision["calibration_eligible"] is False
    assert decision["time_invariant_race_layer_fit_authorized"] is False

    no_size = composition.copy()
    no_size[["tv_material_at_0_05", "wrms_material_at_log_1_10"]] = False
    no_signal = AUDIT.centered_audit_decision(
        comparison, no_size, uncertainty, support, denominator
    )
    assert no_signal["single_complete_window_pooled_descriptive_signal"] is False


def test_run_audit_rejects_non_frozen_surveillance_years(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="direct-surveillance years"):
        AUDIT.run_audit(
            fit_dir=tmp_path / "unused-fit",
            db_path=tmp_path / "unused.db",
            surveillance_path=tmp_path / "unused.csv",
            years=(2014, 2016),
            output_dir=tmp_path / "unused-output",
        )


def test_end_to_end_audit_is_read_only_and_writes_provenance(
    tmp_path: Path,
) -> None:
    fit_dir = tmp_path / "fit"
    _write_fit(fit_dir)

    records: list[pd.DataFrame] = []
    code_by_race_idx = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: None, 6: 6}
    for row in _race_cells().itertuples(index=False):
        n = int(row.N_cell)
        recorded = int(row.R_cell)
        records.append(
            pd.DataFrame(
                {
                    "year": np.full(n, row.year, dtype=int),
                    "mage_c": np.full(n, row.maternal_age, dtype=int),
                    "mracehisp_c": [code_by_race_idx[row.race_idx]] * n,
                    "down_ind": np.r_[
                        np.ones(recorded, dtype=int),
                        np.zeros(n - recorded, dtype=int),
                    ],
                }
            )
        )
    births = pd.concat(records, ignore_index=True)
    db_path = tmp_path / "births.db"
    con = duckdb.connect(str(db_path))
    con.register("_births", births)
    con.execute("CREATE TABLE us_births AS SELECT * FROM _births")
    con.unregister("_births")
    con.close()
    db_hash_before = AUDIT._sha256(db_path)

    surveillance_path = tmp_path / "surveillance.csv"
    _surveillance_source().to_csv(surveillance_path, index=False)
    output_dir = tmp_path / "audit"
    paths, decision = AUDIT.run_audit(
        fit_dir=fit_dir,
        db_path=db_path,
        surveillance_path=surveillance_path,
        output_dir=output_dir,
        invocation="synthetic end-to-end test",
    )

    assert AUDIT._sha256(db_path) == db_hash_before
    assert len(paths) == 14
    assert all(path.is_file() for path in paths.values())
    assert decision["calibration_eligible"] is False

    payload = json.loads((output_dir / "audit.json").read_text(encoding="utf-8"))
    metadata = payload["metadata"]
    assert metadata["duckdb_access"] == "read_only"
    assert metadata["duckdb_table"] == "us_births"
    assert metadata["invocation"] == "synthetic end-to-end test"
    assert metadata["source_definition"] == {
        "window_width_years": 5,
        "window_alignment": "centred",
        "label_role": "centre_year",
        "race_basis": "maternal",
        "confirmed_by": "project lead",
        "confirmed_date": "2026-08-03",
        "aggregation_operator": {
            "status": "confirmed",
            "name": "pooled_count_ratio",
            "definition": (
                "five-year true-case numerator divided by five-year birth denominator"
            ),
        },
        "birth_denominator_components": (
            "sum of annual race-specific births_bc over the centred window"
        ),
        "window_overlap": {
            "label_years": [2016, 2018],
            "shared_years": [2016, 2017, 2018],
            "fraction_of_each_window": 0.6,
            "jaccard_fraction": 3 / 7,
            "independent_validation": False,
        },
    }
    assert metadata["complete_centred_window_label_years"] == [2018]
    assert metadata["partial_centred_window_label_years"] == [2016]
    assert "cross_year_transport" not in payload["sections"]
    assert "centered_window_support" in payload["sections"]
    assert "centered_window_comparison" in payload["sections"]
    assert "source_pooled_window_denominator_mapping" in payload["sections"]
    assert "source_annual_category_mapping" not in payload["sections"]
    assert metadata["input_hashes"]["audit_script_sha256"] == AUDIT._sha256(SCRIPT_PATH)
    assert metadata["input_hashes"]["protocol_note_sha256"] == AUDIT._sha256(
        AUDIT.PROTOCOL_PATH
    )
    assert metadata["git"]["commit"]
    assert metadata["runtime"]["python"]

    manifest = json.loads(
        (output_dir / "race_surveillance_audit_config.json").read_text(encoding="utf-8")
    )
    assert len(manifest["artefact_paths"]) == 14
    assert manifest["metadata"]["source_window_alignment_status"] == (
        "confirmed_centred"
    )
    assert manifest["metadata"]["source_race_basis_status"] == "confirmed_maternal"
    assert manifest["metadata"]["source_aggregation_operator_status"] == (
        "confirmed_pooled_count_ratio"
    )
