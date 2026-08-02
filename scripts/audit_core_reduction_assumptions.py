"""Audit measurement and external-anchor assumptions in the core DS model.

The audit is deliberately read-only.  It can inspect the project DuckDB for
confirmed/pending Down-syndrome certificate codes and the exact maternal-age
distribution, and can optionally inspect a core-model ``cells.parquet`` plus
``config.json``.  It does not fit a model and does not update the database.

Outputs are a wide ``audit.csv`` and a structured ``audit.json``.  The latter
retains separate measurement and identified-product sections so downstream
reporting does not accidentally present the identified product as recording
sensitivity or prenatal reduction.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from dspopulations_us_birth_certificates.chance import (
    get_ds_lb_nt_probability_array,
)

DEFAULT_DB_PATH = Path("data/us_births.db")
DEFAULT_OUTPUT_ROOT = Path("output/core_reduction_assumption_audit")
DEFAULT_YEAR_RANGE = (2016, 2024)
DEFAULT_FALSE_POSITIVE_RATE = 7.8e-5
AGE_LEVELS = ("<20", "20-24", "25-29", "30-34", "35-39", "40-44", "45+")
AGE_BIN_EDGES = np.array([20, 25, 30, 35, 40, 45], dtype=int)
# Explicit snapshot of the seven-band approximation under audit. A supplied fit
# config overrides it, so the audit describes the fitted assumptions even if the
# live model registry is being edited or has since changed.
CURRENT_BAND_THETA = np.array(
    [0.00066, 0.00070, 0.00084, 0.00148, 0.00472, 0.01522, 0.03071]
)
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def _validate_probability(value: float, *, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value < 1.0:
        raise ValueError(f"{name} must be finite and lie in [0, 1)")
    return value


def calibrated_false_positive_rate(
    *,
    target_share_among_recorded: float,
    recorded_count: float,
    non_ds_exposure: float,
) -> float:
    """Convert a target false-coded share among recorded flags to model ``f``.

    ``f`` is a probability per non-DS birth, not a share among recorded flags.
    The conversion is population-specific because it also depends on the
    recorded-flag rate in the audited population and era.
    """
    target = _validate_probability(
        target_share_among_recorded, name="target_share_among_recorded"
    )
    recorded_count = float(recorded_count)
    non_ds_exposure = float(non_ds_exposure)
    if not np.isfinite(recorded_count) or recorded_count < 0.0:
        raise ValueError("recorded_count must be finite and non-negative")
    if not np.isfinite(non_ds_exposure) or non_ds_exposure <= 0.0:
        raise ValueError("non_ds_exposure must be finite and positive")
    calibrated = target * recorded_count / non_ds_exposure
    if calibrated >= 1.0:
        raise ValueError("the requested calibration implies f >= 1")
    return calibrated


def _parse_year_range(value: str) -> tuple[int, int]:
    try:
        start_text, end_text = value.split("-", maxsplit=1)
        start, end = int(start_text), int(end_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("years must have the form YYYY-YYYY") from exc
    if start > end:
        raise argparse.ArgumentTypeError("the first year must not exceed the last")
    return start, end


def age_band_indices(ages: pd.Series | np.ndarray) -> np.ndarray:
    """Map exact maternal ages to the core model's seven age bands."""
    values = np.asarray(ages, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("maternal ages must be finite")
    return np.searchsorted(AGE_BIN_EDGES, values, side="right").astype("int8")


def load_exact_age_counts(
    db_path: Path | str,
    *,
    year_range: tuple[int, int] = DEFAULT_YEAR_RANGE,
    table: str = "us_births",
) -> pd.DataFrame:
    """Read exact-age birth and C/P counts from DuckDB without modifying it."""
    if not _IDENTIFIER.fullmatch(table):
        raise ValueError(f"invalid DuckDB table identifier: {table!r}")
    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    start, end = year_range
    con = duckdb.connect(str(path), read_only=True)
    try:
        frame = con.execute(
            f"""
            SELECT
                CAST(mage_c AS INTEGER) AS age,
                COUNT(*) AS births_age_known,
                COUNT(*) FILTER (WHERE down_ind IS NOT NULL) AS births,
                COUNT(*) FILTER (WHERE down_ind IS NULL) AS excluded_unknown_down_ind,
                COUNT(*) FILTER (WHERE UPPER(ca_down_c) = 'C') AS confirmed,
                COUNT(*) FILTER (WHERE UPPER(ca_down_c) = 'P') AS pending
            FROM {table}
            WHERE year BETWEEN ? AND ?
              AND mage_c IS NOT NULL
            GROUP BY age
            ORDER BY age
            """,
            [start, end],
        ).df()
    finally:
        con.close()
    if frame.empty:
        raise ValueError(f"no rows found in {table} for {start}-{end}")
    for column in (
        "age",
        "births_age_known",
        "births",
        "excluded_unknown_down_ind",
        "confirmed",
        "pending",
    ):
        frame[column] = frame[column].astype("int64")
    return frame


def _add_measurement_metrics(
    frame: pd.DataFrame,
    *,
    false_positive_rate: float,
    target_fp_share_among_recorded: float | None,
) -> pd.DataFrame:
    out = frame.copy()
    out["recorded_cp"] = out["confirmed"] + out["pending"]
    recorded = out["recorded_cp"].replace(0, np.nan)
    births = out["births"].replace(0, np.nan)
    out["confirmed_share_cp"] = out["confirmed"] / recorded
    out["pending_share_cp"] = out["pending"] / recorded
    out["recorded_cp_share_births"] = out["recorded_cp"] / births
    out["recorded_cp_per_10k"] = 1e4 * out["recorded_cp"] / births
    out["model_eligible_share_age_known"] = out["births"] / out[
        "births_age_known"
    ].replace(0, np.nan)
    out["band_minus_exact_expected_ds"] = (
        out["band_expected_ds"] - out["exact_expected_ds"]
    )
    out["band_vs_exact_relative_difference"] = out[
        "band_minus_exact_expected_ds"
    ] / out["exact_expected_ds"].replace(0.0, np.nan)
    non_ds_proxy = np.maximum(out["births"] - out["exact_expected_ds"], 0.0)
    out["false_positive_rate_per_non_ds_birth"] = false_positive_rate
    out["expected_fp_all_births_upper_approx"] = out["births"] * false_positive_rate
    out["expected_fp_natural_non_ds_proxy"] = non_ds_proxy * false_positive_rate
    out["implied_fp_share_recorded_all_births_approx"] = (
        out["expected_fp_all_births_upper_approx"] / recorded
    )
    out["implied_fp_share_recorded_natural_non_ds_proxy"] = (
        out["expected_fp_natural_non_ds_proxy"] / recorded
    )
    if target_fp_share_among_recorded is not None:
        target = _validate_probability(
            target_fp_share_among_recorded,
            name="target_fp_share_among_recorded",
        )
        out["target_fp_share_among_recorded_cp"] = target
        out["calibrated_f_all_births_for_target_share"] = (
            target * out["recorded_cp"] / births
        )
        out["calibrated_f_natural_non_ds_proxy_for_target_share"] = (
            target * out["recorded_cp"] / pd.Series(non_ds_proxy).replace(0.0, np.nan)
        )
    return out


def measurement_anchor_tables(
    exact_age_counts: pd.DataFrame,
    *,
    false_positive_rate: float = DEFAULT_FALSE_POSITIVE_RATE,
    band_theta: np.ndarray = CURRENT_BAND_THETA,
    target_fp_share_among_recorded: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return exact-age and band/overall measurement-anchor audit tables."""
    false_positive_rate = _validate_probability(
        false_positive_rate, name="false_positive_rate"
    )
    required = {"age", "births", "confirmed", "pending"}
    missing = required - set(exact_age_counts.columns)
    if missing:
        raise ValueError(f"exact_age_counts is missing columns: {sorted(missing)}")
    theta = np.asarray(band_theta, dtype=float)
    if theta.shape != (len(AGE_LEVELS),) or not np.all((theta > 0.0) & (theta < 1.0)):
        raise ValueError(f"band_theta must contain {len(AGE_LEVELS)} probabilities")

    exact = exact_age_counts.loc[:, ["age", "births", "confirmed", "pending"]].copy()
    if "births_age_known" in exact_age_counts:
        exact["births_age_known"] = exact_age_counts["births_age_known"]
    else:
        exact["births_age_known"] = exact["births"]
    if "excluded_unknown_down_ind" in exact_age_counts:
        exact["excluded_unknown_down_ind"] = exact_age_counts[
            "excluded_unknown_down_ind"
        ]
    else:
        exact["excluded_unknown_down_ind"] = exact["births_age_known"] - exact["births"]
    for column in (
        "age",
        "births_age_known",
        "births",
        "excluded_unknown_down_ind",
        "confirmed",
        "pending",
    ):
        exact[column] = pd.to_numeric(exact[column], errors="raise")
    count_columns = [
        "births_age_known",
        "births",
        "excluded_unknown_down_ind",
        "confirmed",
        "pending",
    ]
    if (exact[count_columns] < 0).any().any():
        raise ValueError("birth and C/P counts must be non-negative")
    if not (
        exact["births"] + exact["excluded_unknown_down_ind"]
        == exact["births_age_known"]
    ).all():
        raise ValueError(
            "births + excluded_unknown_down_ind must equal births_age_known"
        )
    if ((exact["confirmed"] + exact["pending"]) > exact["births"]).any():
        raise ValueError("confirmed + pending cannot exceed births")

    exact["age_idx"] = age_band_indices(exact["age"])
    exact["age_label"] = [AGE_LEVELS[idx] for idx in exact["age_idx"]]
    exact["exact_theta"] = get_ds_lb_nt_probability_array(
        exact["age"].to_numpy(dtype=float)
    )
    exact["band_theta"] = theta[exact["age_idx"].to_numpy(dtype=int)]
    exact["exact_expected_ds"] = exact["births"] * exact["exact_theta"]
    exact["band_expected_ds"] = exact["births"] * exact["band_theta"]
    exact.insert(0, "scope", "exact_age")
    exact = _add_measurement_metrics(
        exact,
        false_positive_rate=false_positive_rate,
        target_fp_share_among_recorded=target_fp_share_among_recorded,
    )

    additive = [
        "births_age_known",
        "births",
        "excluded_unknown_down_ind",
        "confirmed",
        "pending",
        "exact_expected_ds",
        "band_expected_ds",
    ]
    band = exact.groupby("age_idx", as_index=False, observed=True)[additive].sum()
    band["age_label"] = [AGE_LEVELS[int(idx)] for idx in band["age_idx"]]
    band["age"] = np.nan
    band["exact_theta"] = band["exact_expected_ds"] / band["births"]
    band["band_theta"] = band["band_expected_ds"] / band["births"]
    band.insert(0, "scope", "age_band")
    band = _add_measurement_metrics(
        band,
        false_positive_rate=false_positive_rate,
        target_fp_share_among_recorded=target_fp_share_among_recorded,
    )

    overall_values = exact[additive].sum()
    overall = pd.DataFrame(
        [
            {
                "scope": "overall",
                "age_idx": -1,
                "age_label": "Overall",
                "age": np.nan,
                **overall_values.to_dict(),
                "exact_theta": overall_values["exact_expected_ds"]
                / overall_values["births"],
                "band_theta": overall_values["band_expected_ds"]
                / overall_values["births"],
            }
        ]
    )
    overall = _add_measurement_metrics(
        overall,
        false_positive_rate=false_positive_rate,
        target_fp_share_among_recorded=target_fp_share_among_recorded,
    )
    summary = pd.concat([overall, band], ignore_index=True, sort=False)
    return exact, summary


def load_config(path: Path | str | None) -> dict[str, Any]:
    """Load a core-model JSON configuration, or return an empty configuration."""
    if path is None:
        return {}
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    value = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{config_path} must contain a JSON object")
    return value


def _config_priors(config: dict[str, Any]) -> dict[str, Any]:
    priors = config.get("priors", {})
    return priors if isinstance(priors, dict) else {}


def resolve_false_positive_rate(
    config: dict[str, Any], override: float | None = None
) -> float:
    """Resolve ``f`` from an explicit override, fit config, or project default."""
    if override is not None:
        return _validate_probability(override, name="false_positive_rate")
    value = _config_priors(config).get(
        "false_positive_rate", DEFAULT_FALSE_POSITIVE_RATE
    )
    return _validate_probability(value, name="false_positive_rate")


def resolve_band_theta(config: dict[str, Any]) -> np.ndarray:
    """Resolve the seven band-level theta values represented by a fitted config."""
    values = _config_priors(config).get("theta_lb_age", CURRENT_BAND_THETA)
    theta = np.asarray(values, dtype=float)
    if theta.shape != (len(AGE_LEVELS),) or not np.all((theta > 0.0) & (theta < 1.0)):
        raise ValueError(
            f"config priors.theta_lb_age must contain {len(AGE_LEVELS)} probabilities"
        )
    return theta


def _maternal_age_label(age: int) -> str:
    if age == 12:
        return "10-12"
    if age == 50:
        return "50+"
    return str(age)


def _uses_single_year_age(cells: pd.DataFrame, age_model: str | None) -> bool:
    if age_model not in {None, "band", "single_year"}:
        raise ValueError("age_model must be 'band', 'single_year', or None")
    return "maternal_age" in cells.columns or age_model == "single_year"


def identified_product_age_table(
    cells: pd.DataFrame,
    *,
    theta_age: np.ndarray = CURRENT_BAND_THETA,
    false_positive_rate: float = DEFAULT_FALSE_POSITIVE_RATE,
    target_fp_share_among_recorded: float | None = None,
    ppc_by_age: pd.DataFrame | None = None,
    age_model: str | None = None,
) -> pd.DataFrame:
    """Summarise the age-specific product identified by certificate counts.

    In the core observation model,

    ``p_recorded = f + theta_age * eta * (s - f)``.

    Consequently ``(p_recorded - f) / theta_age`` estimates the combined
    ``eta * (s - f)`` product.  It does not identify ``eta`` or ``s`` separately.
    """
    false_positive_rate = _validate_probability(
        false_positive_rate, name="false_positive_rate"
    )
    required = {"age_idx", "N_cell", "R_cell"}
    missing = required - set(cells.columns)
    if missing:
        raise ValueError(f"cells is missing columns: {sorted(missing)}")
    single_year = _uses_single_year_age(cells, age_model)
    if single_year:
        if "maternal_age" not in cells.columns:
            raise ValueError(
                "single-year age diagnostics require cells.maternal_age; age_idx is "
                "only a coordinate and cannot recover the represented age"
            )
        grouped = (
            cells.groupby(["age_idx", "maternal_age"], as_index=False, observed=True)[
                ["N_cell", "R_cell"]
            ]
            .sum()
            .sort_values("age_idx")
            .reset_index(drop=True)
        )
        if (
            grouped["age_idx"].duplicated().any()
            or grouped["maternal_age"].duplicated().any()
        ):
            raise ValueError(
                "exact-age age_idx and maternal_age values must map one-to-one"
            )
        ages = grouped["maternal_age"].to_numpy(dtype=float)
        if not np.isfinite(ages).all() or not np.equal(ages, np.floor(ages)).all():
            raise ValueError("maternal_age values must be finite integers")
        maternal_ages = ages.astype(int)
        grouped["maternal_age"] = maternal_ages
        grouped["age_model"] = "single_year"
        grouped["age_label"] = [_maternal_age_label(age) for age in maternal_ages]
        grouped["maternal_age_endpoint_capped"] = np.isin(maternal_ages, (12, 50))
        grouped["theta_age"] = get_ds_lb_nt_probability_array(maternal_ages)
    else:
        theta = np.asarray(theta_age, dtype=float)
        if theta.shape != (len(AGE_LEVELS),) or not np.all(
            (theta > 0.0) & (theta < 1.0)
        ):
            raise ValueError(f"theta_age must contain {len(AGE_LEVELS)} probabilities")
        grouped = (
            cells.groupby("age_idx", as_index=False, observed=True)[
                ["N_cell", "R_cell"]
            ]
            .sum()
            .sort_values("age_idx")
            .reset_index(drop=True)
        )
        indices = grouped["age_idx"].to_numpy(dtype=int)
        if not np.all((0 <= indices) & (indices < len(theta))):
            raise ValueError(
                "cells contains an age_idx outside the configured theta vector"
            )
        grouped["maternal_age"] = np.nan
        grouped["age_model"] = "band"
        grouped["age_label"] = [AGE_LEVELS[idx] for idx in indices]
        grouped["maternal_age_endpoint_capped"] = False
        grouped["theta_age"] = theta[indices]
    if grouped.empty:
        raise ValueError("cells contains no age groups")
    if (grouped["N_cell"] <= 0).any() or (grouped["R_cell"] < 0).any():
        raise ValueError("cells must have positive N_cell and non-negative R_cell")

    grouped["scope"] = "exact_age" if single_year else "age_band"
    grouped["false_positive_rate_per_non_ds_birth"] = false_positive_rate
    grouped["observed_rate"] = grouped["R_cell"] / grouped["N_cell"]
    grouped["expected_fp_all_births_upper_approx"] = (
        grouped["N_cell"] * false_positive_rate
    )
    grouped["implied_fp_share_recorded_all_births_approx"] = grouped[
        "expected_fp_all_births_upper_approx"
    ] / grouped["R_cell"].replace(0.0, np.nan)
    grouped["observed_minus_expected_fp"] = (
        grouped["R_cell"] - grouped["N_cell"] * false_positive_rate
    )
    grouped["identified_eta_times_s_minus_f"] = (
        grouped["observed_rate"] - false_positive_rate
    ) / grouped["theta_age"]
    grouped["naive_eta_times_s"] = grouped["observed_rate"] / grouped["theta_age"]

    overall_n = float(grouped["N_cell"].sum())
    overall_r = float(grouped["R_cell"].sum())
    natural_expected = float((grouped["N_cell"] * grouped["theta_age"]).sum())
    overall = pd.DataFrame(
        [
            {
                "age_idx": -1,
                "N_cell": overall_n,
                "R_cell": overall_r,
                "scope": "overall",
                "age_label": "Overall",
                "maternal_age": np.nan,
                "age_model": "single_year" if single_year else "band",
                "maternal_age_endpoint_capped": False,
                "theta_age": natural_expected / overall_n,
                "false_positive_rate_per_non_ds_birth": false_positive_rate,
                "observed_rate": overall_r / overall_n,
                "expected_fp_all_births_upper_approx": (
                    overall_n * false_positive_rate
                ),
                "implied_fp_share_recorded_all_births_approx": (
                    overall_n * false_positive_rate / overall_r
                    if overall_r > 0.0
                    else np.nan
                ),
                "observed_minus_expected_fp": (
                    overall_r - overall_n * false_positive_rate
                ),
                "identified_eta_times_s_minus_f": (
                    overall_r - overall_n * false_positive_rate
                )
                / natural_expected,
                "naive_eta_times_s": overall_r / natural_expected,
            }
        ]
    )
    result = pd.concat([overall, grouped], ignore_index=True, sort=False)

    if target_fp_share_among_recorded is not None:
        target = _validate_probability(
            target_fp_share_among_recorded,
            name="target_fp_share_among_recorded",
        )
        result["target_fp_share_among_recorded"] = target
        result["calibrated_f_all_births_for_target_share"] = (
            target * result["R_cell"] / result["N_cell"]
        )

    if ppc_by_age is not None:
        expected = {"age_idx", "observed", "predicted_mean"}
        missing_ppc = expected - set(ppc_by_age.columns)
        if missing_ppc:
            raise ValueError(f"ppc_by_age is missing columns: {sorted(missing_ppc)}")
        ppc = ppc_by_age.copy()
        ppc["age_idx"] = ppc["age_idx"].astype(int)
        keep = [
            column
            for column in (
                "age_idx",
                "observed",
                "predicted_mean",
                "predicted_lo",
                "predicted_hi",
                "observed_in_interval",
            )
            if column in ppc.columns
        ]
        result = result.merge(ppc[keep], on="age_idx", how="left", validate="1:1")
        result["observed_minus_predicted"] = (
            result["observed"] - result["predicted_mean"]
        )
        result["observed_minus_predicted_relative"] = result[
            "observed_minus_predicted"
        ] / result["predicted_mean"].replace(0.0, np.nan)
    return result


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def write_audit_outputs(
    output_dir: Path | str,
    *,
    metadata: dict[str, Any],
    exact_age: pd.DataFrame | None = None,
    measurement_summary: pd.DataFrame | None = None,
    identified_product: pd.DataFrame | None = None,
) -> tuple[Path, Path]:
    """Write the audit as one combined CSV and one structured JSON file."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sections: dict[str, list[dict[str, Any]]] = {}
    csv_frames: list[pd.DataFrame] = []
    for name, frame in (
        ("measurement_by_exact_age", exact_age),
        ("measurement_summary", measurement_summary),
        ("identified_product_by_age", identified_product),
    ):
        if frame is None:
            continue
        sections[name] = _json_records(frame)
        csv_frame = frame.copy()
        csv_frame.insert(0, "section", name)
        csv_frames.append(csv_frame)
    if not csv_frames:
        raise ValueError("the audit contains no sections to write")

    combined = pd.concat(csv_frames, ignore_index=True, sort=False)
    csv_path = out_dir / "audit.csv"
    json_path = out_dir / "audit.json"
    combined.to_csv(csv_path, index=False)
    payload = {"metadata": metadata, "sections": sections}
    json_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    return csv_path, json_path


def _resolve_fit_paths(
    ns: argparse.Namespace,
) -> tuple[Path | None, Path | None, Path | None]:
    cells = ns.cells
    config = ns.config
    ppc = ns.ppc_by_age
    if ns.fit_dir is not None:
        cells = cells or ns.fit_dir / "cells.parquet"
        config = config or ns.fit_dir / "config.json"
        candidate = ns.fit_dir / "tables" / "core_ppc_by_age.csv"
        if ppc is None and candidate.is_file():
            ppc = candidate
    return cells, config, ppc


def _resolve_year_range(
    explicit: tuple[int, int] | None, config: dict[str, Any]
) -> tuple[int, int]:
    if explicit is not None:
        return explicit
    value = config.get("year_range", DEFAULT_YEAR_RANGE)
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or int(value[0]) > int(value[1])
    ):
        raise ValueError("config year_range must contain [start, end]")
    return int(value[0]), int(value[1])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--duckdb-path",
        type=Path,
        default=None,
        help=(
            "Read-only project DuckDB. If omitted, data/us_births.db is used when "
            "present; otherwise a supplied cells parquet is required."
        ),
    )
    parser.add_argument("--table", default="us_births")
    parser.add_argument("--fit-dir", type=Path, default=None)
    parser.add_argument("--cells", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--ppc-by-age", type=Path, default=None)
    parser.add_argument("--years", type=_parse_year_range, default=None)
    parser.add_argument(
        "--false-positive-rate",
        type=float,
        default=None,
        help="Override f; otherwise use config.json or the project default.",
    )
    parser.add_argument(
        "--target-fp-share",
        type=float,
        default=None,
        help=(
            "Optional target false-positive share among recorded C/P flags. "
            "The audit reports the population-specific f that would imply it."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for audit.csv and audit.json.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    ns = parse_args(argv)
    cells_path, config_path, ppc_path = _resolve_fit_paths(ns)
    config = load_config(config_path)
    year_range = _resolve_year_range(ns.years, config)
    false_positive_rate = resolve_false_positive_rate(config, ns.false_positive_rate)
    theta = resolve_band_theta(config)

    db_path = ns.duckdb_path
    if db_path is None and DEFAULT_DB_PATH.is_file():
        db_path = DEFAULT_DB_PATH
    if db_path is None and cells_path is None:
        raise SystemExit(
            "No audit input found: supply --duckdb-path, --cells, or --fit-dir."
        )

    exact_age: pd.DataFrame | None = None
    measurement_summary: pd.DataFrame | None = None
    if db_path is not None:
        age_counts = load_exact_age_counts(
            db_path, year_range=year_range, table=ns.table
        )
        exact_age, measurement_summary = measurement_anchor_tables(
            age_counts,
            false_positive_rate=false_positive_rate,
            band_theta=theta,
            target_fp_share_among_recorded=ns.target_fp_share,
        )

    identified: pd.DataFrame | None = None
    identified_age_model: str | None = None
    if cells_path is not None:
        if not cells_path.is_file():
            raise FileNotFoundError(cells_path)
        cells = pd.read_parquet(cells_path)
        ppc = None
        if ppc_path is not None:
            if not ppc_path.is_file():
                raise FileNotFoundError(ppc_path)
            ppc = pd.read_csv(ppc_path)
        identified = identified_product_age_table(
            cells,
            theta_age=theta,
            false_positive_rate=false_positive_rate,
            target_fp_share_among_recorded=ns.target_fp_share,
            ppc_by_age=ppc,
            age_model=config.get("age_model"),
        )
        identified_age_model = str(
            identified.loc[identified["scope"] != "overall", "age_model"].iloc[0]
        )

    output_dir = ns.output_dir or (
        DEFAULT_OUTPUT_ROOT / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    metadata = {
        "year_range": list(year_range),
        "false_positive_rate": false_positive_rate,
        "false_positive_rate_definition": "probability of a DS flag per non-DS birth",
        "target_fp_share_among_recorded": ns.target_fp_share,
        "false_positive_share_definition": (
            "expected false-positive flags divided by observed confirmed-plus-pending flags"
        ),
        "false_positive_calibration_caveat": (
            "Converting a validation false-coded share to f requires the recorded-flag "
            "rate and non-DS exposure from the same validation population and era; "
            "values calibrated on this audit cohort are sensitivity scenarios, not "
            "transported validation estimates."
        ),
        "project_default_f_derivation_warning": (
            "The project default 7.8e-5 historically combined a 7.8% false-coded "
            "share with an assumed recorded rate of 1e-3. The audit reports both "
            "scales and does not treat that conversion as transportable."
        ),
        "theta_lb_age": theta.tolist(),
        "duckdb_path": str(db_path) if db_path is not None else None,
        "cells_path": str(cells_path) if cells_path is not None else None,
        "config_path": str(config_path) if config_path is not None else None,
        "ppc_by_age_path": str(ppc_path) if ppc_path is not None else None,
        "identified_product_age_model": identified_age_model,
        "maternal_age_endpoint_warning": (
            "Maternal ages 12 and 50 are capped representatives: 12 denotes ages "
            "10-12 and 50 denotes ages 50+. Morris theta is evaluated at the "
            "representative ages 12 and 50."
            if identified_age_model == "single_year"
            else None
        ),
        "identified_product_definition": "eta * (s - f) = (observed_rate - f) / theta_age",
    }
    csv_path, json_path = write_audit_outputs(
        output_dir,
        metadata=metadata,
        exact_age=exact_age,
        measurement_summary=measurement_summary,
        identified_product=identified,
    )

    print(f"Core reduction assumption audit: {year_range[0]}-{year_range[1]}")
    print(f"false-positive rate f per non-DS birth: {false_positive_rate:.8g}")
    if measurement_summary is not None:
        columns = [
            "age_label",
            "births_age_known",
            "births",
            "excluded_unknown_down_ind",
            "confirmed",
            "pending",
            "confirmed_share_cp",
            "pending_share_cp",
            "exact_expected_ds",
            "band_expected_ds",
            "expected_fp_all_births_upper_approx",
            "implied_fp_share_recorded_all_births_approx",
        ]
        if ns.target_fp_share is not None:
            columns.extend(
                [
                    "target_fp_share_among_recorded_cp",
                    "calibrated_f_natural_non_ds_proxy_for_target_share",
                ]
            )
        print("\nMeasurement and anchor summary")
        print(measurement_summary[columns].to_string(index=False))
        print(
            "\nCalibration warning: a false-coded share among recorded flags is not f; "
            "conversion depends on the validation population and era."
        )
    if identified is not None:
        columns = [
            "age_label",
            "N_cell",
            "R_cell",
            "observed_rate",
            "identified_eta_times_s_minus_f",
        ]
        if ns.target_fp_share is not None:
            columns.extend(
                [
                    "target_fp_share_among_recorded",
                    "calibrated_f_all_births_for_target_share",
                ]
            )
        print("\nIdentified-product diagnostic")
        print(identified[columns].to_string(index=False))
        if identified_age_model == "single_year":
            print(
                "\nExact-age warning: 12 represents ages 10-12 and 50 represents "
                "ages 50+; Morris theta uses the representative ages 12 and 50."
            )
    print(f"\nCSV:  {csv_path}")
    print(f"JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
